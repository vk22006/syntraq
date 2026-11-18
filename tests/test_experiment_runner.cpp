//
// tests/test_experiment_runner.cpp
//
// Unit tests for the SyntraQ Reproducible Experiment Framework.
// Validates:
//  - ExperimentConfig and ExperimentBatch grid generation
//  - Headless execution and reproducible metric capture across seeds
//  - Statistical aggregation (mean, std dev, 95% confidence interval bounds)
//  - Machine-readable serialization (CSV and JSON export)
//  - Controller and routing comparison under identical conditions
//

#include "syntraq/experiments/experiment_runner.hpp"
#include "syntraq/core/config.hpp"
#include "syntraq/world/road_network.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace syntraq;

namespace {

RoadNetwork make_test_corridor() {
    RoadNetwork net;
    const auto i0 = net.add_intersection({   0.0f, 0.0f }, "West");
    const auto i1 = net.add_intersection({ 200.0f, 0.0f }, "Center");
    const auto i2 = net.add_intersection({ 400.0f, 0.0f }, "East");
    const auto i3 = net.add_intersection({ 200.0f, 150.0f }, "North");

    net.add_road(i0, i1, 200.0f, 13.89f, 1, "WestToCenter");
    net.add_road(i1, i2, 200.0f, 13.89f, 1, "CenterToEast");
    net.add_road(i3, i1, 150.0f, 13.89f, 1, "NorthToCenter");
    return net;
}

} // namespace

class ExperimentRunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "syntraq_test_experiments";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(test_dir_, ec);
    }

    fs::path test_dir_;
};

TEST_F(ExperimentRunnerTest, ExperimentConfigDefaultsAndSimConfigConversion) {
    ExperimentConfig exp;
    exp.experiment_id    = "test_01";
    exp.scenario         = TrafficScenario::Medium;
    exp.controller       = SignalControllerType::Adaptive;
    exp.routing_mode     = RoutingStrategyType::TrafficAware;
    exp.seed             = 12345u;
    exp.duration_s       = 45.0f;
    exp.traffic_density  = 1.5f;
    exp.spawn_interval_s = 1.33f;
    exp.max_vehicles     = 52;
    exp.model_version    = "none";

    const Config cfg = exp.to_simulation_config();
    EXPECT_EQ(cfg.seed, 12345u);
    EXPECT_FLOAT_EQ(cfg.sim_duration_s, 45.0f);
    EXPECT_TRUE(cfg.headless);
    EXPECT_EQ(cfg.scenario, TrafficScenario::Medium);
    EXPECT_EQ(cfg.signal_controller_type, SignalControllerType::Adaptive);
    EXPECT_EQ(cfg.routing_strategy, RoutingStrategyType::TrafficAware);
    EXPECT_FLOAT_EQ(cfg.vehicle_spawn_interval_s, 1.33f);
    EXPECT_EQ(cfg.max_vehicles, 52u);
}

TEST_F(ExperimentRunnerTest, BatchGridGenerationComputesCorrectCartesianProduct) {
    ExperimentBatch batch("GridTest");
    const std::vector<TrafficScenario> scenarios = { TrafficScenario::Low, TrafficScenario::Medium };
    const std::vector<SignalControllerType> controllers = { SignalControllerType::FixedTime, SignalControllerType::Adaptive };
    const std::vector<RoutingStrategyType> routings = { RoutingStrategyType::Static };
    const std::vector<uint32_t> seeds = { 101u, 202u, 303u };
    const std::vector<float> densities = { 1.0f };

    batch.add_grid(scenarios, controllers, routings, seeds, 60.0f, densities);

    // Expected size = 2 scenarios * 2 controllers * 1 routing * 3 seeds * 1 density = 12
    EXPECT_EQ(batch.size(), 12u);

    // Verify all configs have valid attributes
    for (const auto& exp : batch.experiments()) {
        EXPECT_FALSE(exp.experiment_id.empty());
        EXPECT_FALSE(exp.controller_name.empty());
        EXPECT_FALSE(exp.scenario_name.empty());
        EXPECT_FLOAT_EQ(exp.duration_s, 60.0f);
    }
}

TEST_F(ExperimentRunnerTest, SingleRunExecutionProducesValidMetrics) {
    ExperimentRunner::RunnerOptions opts;
    opts.verbose = false;
    opts.custom_network_factory = make_test_corridor;

    ExperimentRunner runner(opts);

    ExperimentConfig cfg;
    cfg.experiment_id    = "single_run_test";
    cfg.scenario         = TrafficScenario::Low;
    cfg.controller       = SignalControllerType::FixedTime;
    cfg.routing_mode     = RoutingStrategyType::Static;
    cfg.seed             = 42u;
    cfg.duration_s       = 20.0f;
    cfg.spawn_interval_s = 2.0f;
    cfg.max_vehicles     = 10;

    const ExperimentResult res = runner.run_single(cfg);

    EXPECT_TRUE(res.success);
    EXPECT_EQ(res.config.experiment_id, "single_run_test");
    EXPECT_GE(res.metrics.total_spawned, 1u);
    EXPECT_GT(res.execution_wall_time_s, 0.0f);
    EXPECT_GT(res.sim_fps, 0.0f);
    EXPECT_FALSE(res.to_csv_row().empty());
    EXPECT_FALSE(res.to_json_fragment().empty());
}

TEST_F(ExperimentRunnerTest, IdenticalConditionsProduceDeterministicResults) {
    ExperimentRunner::RunnerOptions opts;
    opts.verbose = false;
    opts.custom_network_factory = make_test_corridor;

    ExperimentRunner runner(opts);

    ExperimentConfig cfg;
    cfg.experiment_id    = "deterministic_test";
    cfg.scenario         = TrafficScenario::Medium;
    cfg.controller       = SignalControllerType::FixedTime;
    cfg.routing_mode     = RoutingStrategyType::Static;
    cfg.seed             = 999u;
    cfg.duration_s       = 25.0f;
    cfg.spawn_interval_s = 2.0f;
    cfg.max_vehicles     = 15;

    const ExperimentResult res1 = runner.run_single(cfg);
    const ExperimentResult res2 = runner.run_single(cfg);

    EXPECT_TRUE(res1.success);
    EXPECT_TRUE(res2.success);
    EXPECT_EQ(res1.metrics.total_spawned, res2.metrics.total_spawned);
    EXPECT_EQ(res1.metrics.total_completed, res2.metrics.total_completed);
    EXPECT_FLOAT_EQ(res1.metrics.avg_speed_kmh, res2.metrics.avg_speed_kmh);
    EXPECT_FLOAT_EQ(res1.metrics.avg_waiting_time_s, res2.metrics.avg_waiting_time_s);
    EXPECT_FLOAT_EQ(res1.metrics.throughput_vph, res2.metrics.throughput_vph);
}

TEST_F(ExperimentRunnerTest, BatchExecutionAndStatisticalAggregationWithCI) {
    ExperimentRunner::RunnerOptions opts;
    opts.verbose = false;
    opts.parallel = false;
    opts.custom_network_factory = make_test_corridor;

    ExperimentRunner runner(opts);

    ExperimentBatch batch("SmallSweep");
    // Run 3 seeds across FixedTime vs Adaptive
    batch.add_grid(
        { TrafficScenario::Low },
        { SignalControllerType::FixedTime, SignalControllerType::Adaptive },
        { RoutingStrategyType::Static },
        { 42u, 101u, 202u },
        15.0f // 15s per run for fast test execution
    );

    EXPECT_EQ(batch.size(), 6u);

    const auto results = runner.run_batch(batch);
    EXPECT_EQ(results.size(), 6u);
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);
    }

    // Aggregate into groups
    const auto summaries = ExperimentRunner::aggregate_results(results);
    EXPECT_EQ(summaries.size(), 2u); // FixedTime group and Adaptive group

    for (const auto& s : summaries) {
        EXPECT_EQ(s.sample_count, 3u);
        EXPECT_GE(s.speed_kmh.mean, 0.0);
        EXPECT_GE(s.wait_time_s.mean, 0.0);
        EXPECT_LE(s.wait_time_s.min_val, s.wait_time_s.mean);
        EXPECT_GE(s.wait_time_s.max_val, s.wait_time_s.mean);

        // Verify 95% Confidence Interval bounds surround or match mean
        EXPECT_LE(s.wait_time_s.ci95_lower, s.wait_time_s.mean + 1e-4);
        EXPECT_GE(s.wait_time_s.ci95_upper, s.wait_time_s.mean - 1e-4);
    }
}

TEST_F(ExperimentRunnerTest, CSVAndJSONSerializationIntegrity) {
    ExperimentRunner::RunnerOptions opts;
    opts.verbose = false;
    opts.custom_network_factory = make_test_corridor;

    ExperimentRunner runner(opts);

    ExperimentBatch batch("SerializationTest");
    batch.add_grid(
        { TrafficScenario::Low },
        { SignalControllerType::FixedTime },
        { RoutingStrategyType::Static },
        { 42u, 101u },
        10.0f
    );

    const auto results = runner.run_batch(batch);
    const auto summaries = ExperimentRunner::aggregate_results(results);

    const std::string runs_csv    = (test_dir_ / "runs.csv").string();
    const std::string summary_csv = (test_dir_ / "summary.csv").string();
    const std::string batch_json  = (test_dir_ / "batch.json").string();

    EXPECT_TRUE(ExperimentRunner::export_runs_csv(results, runs_csv));
    EXPECT_TRUE(ExperimentRunner::export_summary_csv(summaries, summary_csv));
    EXPECT_TRUE(ExperimentRunner::export_json("SerializationTest", results, summaries, batch_json));

    // Verify files exist and are non-empty
    EXPECT_TRUE(fs::exists(runs_csv));
    EXPECT_TRUE(fs::exists(summary_csv));
    EXPECT_TRUE(fs::exists(batch_json));

    EXPECT_GT(fs::file_size(runs_csv), 50u);
    EXPECT_GT(fs::file_size(summary_csv), 50u);
    EXPECT_GT(fs::file_size(batch_json), 100u);

    // Validate JSON structure
    std::ifstream jf(batch_json);
    std::string line;
    bool found_batch_name = false;
    bool found_summaries  = false;
    bool found_runs       = false;

    while (std::getline(jf, line)) {
        if (line.find("\"batch_name\": \"SerializationTest\"") != std::string::npos) found_batch_name = true;
        if (line.find("\"summaries\":") != std::string::npos) found_summaries = true;
        if (line.find("\"runs\":") != std::string::npos) found_runs = true;
    }

    EXPECT_TRUE(found_batch_name);
    EXPECT_TRUE(found_summaries);
    EXPECT_TRUE(found_runs);
}

