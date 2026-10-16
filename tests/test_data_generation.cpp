//
// tests/test_data_generation.cpp
//
// Unit tests for the structured training dataset generation subsystem:
//   - TrafficDataSample serialization & CSV schema correctness
//   - Multi-dimensional feature extraction (counts, queues, speeds, occupancy, signals)
//   - Lookahead ground truth target binding (+15s and +30s)
//   - Valid sample filtering without fabricated futures
//   - Simulation integration and headless dataset generation
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_data_collector.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/world/road_network.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace syntraq;

namespace {

TEST(DataGenerationTest, CsvHeaderContainsAllFeatureAndTargetColumns) {
    const std::string header = TrafficDataSample::csv_header();

    EXPECT_NE(header.find("run_id"), std::string::npos);
    EXPECT_NE(header.find("scenario_id"), std::string::npos);
    EXPECT_NE(header.find("vehicle_count"), std::string::npos);
    EXPECT_NE(header.find("queue_vehicle_count"), std::string::npos);
    EXPECT_NE(header.find("max_queue_length"), std::string::npos);
    EXPECT_NE(header.find("avg_queue_per_lane"), std::string::npos);
    EXPECT_NE(header.find("queue_ratio"), std::string::npos);
    EXPECT_NE(header.find("avg_speed_mps"), std::string::npos);
    EXPECT_NE(header.find("avg_speed_kmh"), std::string::npos);
    EXPECT_NE(header.find("speed_ratio"), std::string::npos);
    EXPECT_NE(header.find("road_density_vpm"), std::string::npos);
    EXPECT_NE(header.find("road_occupancy_pct"), std::string::npos);
    EXPECT_NE(header.find("signal_phase_index"), std::string::npos);
    EXPECT_NE(header.find("time_in_phase_s"), std::string::npos);
    EXPECT_NE(header.find("phase_elapsed_ratio"), std::string::npos);
    EXPECT_NE(header.find("recent_throughput_vph"), std::string::npos);
    EXPECT_NE(header.find("recent_queue_delta"), std::string::npos);
    EXPECT_NE(header.find("target_queue_15s"), std::string::npos);
    EXPECT_NE(header.find("target_congestion_15s"), std::string::npos);
    EXPECT_NE(header.find("target_queue_30s"), std::string::npos);
    EXPECT_NE(header.find("target_congestion_30s"), std::string::npos);
}

TEST(DataGenerationTest, SampleRowSerializationMatchesHeaderColumns) {
    TrafficDataSample sample;
    sample.run_id                 = "run_001";
    sample.scenario_id            = 2;
    sample.scenario_name          = "High";
    sample.seed                   = 42;
    sample.timestamp_s            = 10.0f;
    sample.vehicle_count          = 25;
    sample.queue_vehicle_count    = 8;
    sample.max_queue_length       = 4;
    sample.avg_speed_mps          = 8.5f;
    sample.target_queue_15s       = 12;
    sample.target_congestion_15s  = 0.48f;
    sample.has_target_15s         = true;
    sample.target_queue_30s       = 15;
    sample.target_congestion_30s  = 0.60f;
    sample.has_target_30s         = true;

    const std::string header = TrafficDataSample::csv_header();
    const std::string row    = sample.to_csv_row();

    // Count commas to verify equal column counts
    const auto header_commas = std::count(header.begin(), header.end(), ',');
    const auto row_commas    = std::count(row.begin(), row.end(), ',');
    EXPECT_EQ(header_commas, row_commas);
    EXPECT_GT(header_commas, 25);
}

TEST(DataGenerationTest, FeatureExtractionValuesAreWithinPhysicalBounds) {
    RoadNetwork net = make_test_map();
    TrafficDataCollector collector(/* interval = */ 1.0f);
    collector.set_run_metadata("test_run", TrafficScenario::Medium, "Medium", 42);

    std::vector<Vehicle> vehicles;
    // 3 active vehicles: 2 moving, 1 stopped
    Vehicle v1; v1.id = VehicleId{ 1 }; v1.current_road = RoadId{ 0 }; v1.speed_mps = 10.0f; v1.state = VehicleState::Moving; v1.length_m = 4.5f;
    Vehicle v2; v2.id = VehicleId{ 2 }; v2.current_road = RoadId{ 0 }; v2.speed_mps = 8.0f;  v2.state = VehicleState::Moving; v2.length_m = 4.5f;
    Vehicle v3; v3.id = VehicleId{ 3 }; v3.current_road = RoadId{ 0 }; v3.speed_mps = 0.0f;  v3.state = VehicleState::Stopped; v3.length_m = 4.5f;
    vehicles = { v1, v2, v3 };

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;

    collector.update(1.0f, net, vehicles, controllers, 3, 0, 2.0f);

    ASSERT_EQ(collector.samples().size(), 1u);
    const auto& s = collector.samples()[0];

    EXPECT_EQ(s.vehicle_count, 3u);
    EXPECT_EQ(s.queue_vehicle_count, 1u);
    EXPECT_NEAR(s.queue_ratio, 1.0f / 3.0f, 0.01f);
    EXPECT_NEAR(s.avg_speed_mps, 6.0f, 0.01f); // (10 + 8 + 0) / 3 = 6.0
    EXPECT_GT(s.avg_speed_kmh, 20.0f);
    EXPECT_LE(s.speed_ratio, 1.0f);
    EXPECT_GE(s.speed_ratio, 0.0f);
    EXPECT_GT(s.road_density_vpm, 0.0f);
    EXPECT_LE(s.road_occupancy_pct, 100.0f);
    EXPECT_GE(s.road_occupancy_pct, 0.0f);
}

TEST(DataGenerationTest, LookaheadTargetBindingAt15sAnd30s) {
    RoadNetwork net = make_test_map();
    TrafficDataCollector collector(/* sample_interval = */ 1.0f, /* h1 = */ 15.0f, /* h2 = */ 30.0f);
    collector.set_run_metadata("run_h", TrafficScenario::High, "High", 101);

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    std::vector<Vehicle> vehicles;

    // Sample 0 at t = 1.0s with 0 queue
    collector.update(1.0f, net, vehicles, controllers, 0, 0, 1.0f);
    ASSERT_EQ(collector.samples().size(), 1u);
    EXPECT_FALSE(collector.samples()[0].has_target_15s);
    EXPECT_FALSE(collector.samples()[0].has_target_30s);

    // At t = 16.0s (15s after sample 0): introduce 4 stopped vehicles
    for (uint32_t i = 0; i < 4; ++i) {
        Vehicle v;
        v.id = VehicleId{ i };
        v.current_road = RoadId{ 0 };
        v.speed_mps = 0.0f;
        v.state = VehicleState::Stopped;
        v.length_m = 4.5f;
        vehicles.push_back(v);
    }

    collector.update(16.0f, net, vehicles, controllers, 4, 0, 1.0f);

    // Sample 0 at t = 1.0s should now have target_15s bound to 4
    EXPECT_TRUE(collector.samples()[0].has_target_15s);
    EXPECT_EQ(collector.samples()[0].target_queue_15s, 4u);
    EXPECT_NEAR(collector.samples()[0].target_congestion_15s, 1.0f, 0.01f);
    // But target_30s is still pending
    EXPECT_FALSE(collector.samples()[0].has_target_30s);

    // At t = 31.0s (30s after sample 0): introduce 6 stopped vehicles
    for (uint32_t i = 4; i < 6; ++i) {
        Vehicle v;
        v.id = VehicleId{ i };
        v.current_road = RoadId{ 0 };
        v.speed_mps = 0.0f;
        v.state = VehicleState::Stopped;
        v.length_m = 4.5f;
        vehicles.push_back(v);
    }

    collector.update(31.0f, net, vehicles, controllers, 6, 0, 1.0f);

    // Sample 0 at t = 1.0s should now have target_30s bound to 6
    EXPECT_TRUE(collector.samples()[0].has_target_30s);
    EXPECT_EQ(collector.samples()[0].target_queue_30s, 6u);
    EXPECT_NEAR(collector.samples()[0].target_congestion_30s, 1.0f, 0.01f);
}

TEST(DataGenerationTest, SamplesWithoutFutureRetainPendingFlag) {
    RoadNetwork net = make_test_map();
    TrafficDataCollector collector(/* sample_interval = */ 1.0f, /* h1 = */ 15.0f, /* h2 = */ 30.0f);
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    std::vector<Vehicle> vehicles;

    // Simulate 20 seconds total
    for (float t = 1.0f; t <= 20.0f; t += 1.0f) {
        collector.update(t, net, vehicles, controllers, 0, 0, 1.0f);
    }
    collector.finalize();

    // Total samples = 20
    EXPECT_EQ(collector.samples().size(), 20u);

    // Samples from t = 1.0s to 5.0s reached t + 15s (within 20s), so they have target_15s
    EXPECT_TRUE(collector.samples()[0].has_target_15s);
    EXPECT_TRUE(collector.samples()[4].has_target_15s);
    // Samples from t = 6.0s onwards did not reach t + 15s
    EXPECT_FALSE(collector.samples()[5].has_target_15s);

    // NONE of the samples reached t + 30s because sim stopped at 20s
    for (const auto& s : collector.samples()) {
        EXPECT_FALSE(s.has_target_30s);
    }

    // Valid samples requiring 30s horizon should be 0
    EXPECT_EQ(collector.valid_samples(/* require_30s = */ true).size(), 0u);
    // Valid samples requiring 15s horizon should be 5 (t = 1..5)
    EXPECT_EQ(collector.valid_samples(/* require_30s = */ false).size(), 5u);
}

TEST(DataGenerationTest, ExportCsvWritesValidFormattedFile) {
    const std::string tmp_csv = "test_dataset_export.csv";
    if (fs::exists(tmp_csv)) {
        fs::remove(tmp_csv);
    }

    RoadNetwork net = make_test_map();
    TrafficDataCollector collector(/* sample_interval = */ 1.0f, /* h1 = */ 5.0f, /* h2 = */ 10.0f);
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    std::vector<Vehicle> vehicles;

    // Run 15 seconds to resolve both 5s and 10s horizons for early samples
    for (float t = 1.0f; t <= 15.0f; t += 1.0f) {
        collector.update(t, net, vehicles, controllers, 0, 0, 1.0f);
    }
    collector.finalize();

    const bool exported = collector.export_csv(tmp_csv, /* append = */ false, /* require_30s = */ true);
    EXPECT_TRUE(exported);
    EXPECT_TRUE(fs::exists(tmp_csv));

    std::ifstream in(tmp_csv);
    std::string line;
    size_t line_count = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            line_count++;
        }
    }
    in.close();

    // 1 header + 5 resolved samples (t = 1..5 reaches t + 10s <= 15s) = 6 lines
    EXPECT_EQ(line_count, 6u);

    fs::remove(tmp_csv);
}

TEST(DataGenerationTest, SimulationIntegrationDataCollection) {
    Config cfg;
    cfg.scenario               = TrafficScenario::Medium;
    cfg.seed                   = 42;
    cfg.enable_traffic_signals = true;
    cfg.headless               = true;

    Simulation sim{ cfg };
    sim.enable_data_collection(true, /* sample_interval_s = */ 1.0f);
    EXPECT_TRUE(sim.is_data_collection_enabled());

    // Run for 40 seconds (enough to resolve 30s horizon targets)
    sim.run_for(40.0f);
    sim.data_collector().finalize();

    const auto& samples = sim.data_collector().samples();
    EXPECT_GT(samples.size(), 35u);

    const auto valid_30s = sim.data_collector().valid_samples(/* require_30s = */ true);
    // Samples from t = 0..10s will have target_30s resolved by t = 40s
    EXPECT_GE(valid_30s.size(), 9u);

    for (const auto& s : valid_30s) {
        EXPECT_TRUE(s.has_target_15s);
        EXPECT_TRUE(s.has_target_30s);
        EXPECT_GE(s.target_congestion_30s, 0.0f);
        EXPECT_LE(s.target_congestion_30s, 1.0f);
    }
}

} // namespace
