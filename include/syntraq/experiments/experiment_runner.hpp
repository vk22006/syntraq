#pragma once

//
// syntraq/experiments/experiment_runner.hpp
//
// Reproducible Experiment Framework for SyntraQ.
// Supports running batch experiments across parametric grids:
//  - scenarios (Low, Medium, High, RushHour, EmergencyPriority, Custom)
//  - traffic density / spawn interval / vehicle caps
//  - controllers (FixedTime, Adaptive, ReinforcementLearning, Coordinated)
//  - routing modes (Static, TrafficAware)
//  - random seeds
//  - simulation duration
//  - model versions (ONNX model paths or tags)
//
// Supports parallel and sequential execution, exporting machine-readable
// per-run and aggregated statistics in both CSV and JSON formats.
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/simulation/simulation.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace syntraq {

// ── ExperimentConfig ─────────────────────────────────────────────────────────

/// Defines the precise parameters of an individual experiment run.
struct ExperimentConfig {
    std::string         experiment_id      { "exp_default" };
    std::string         name               { "Default Experiment" };
    TrafficScenario     scenario           { TrafficScenario::Medium };
    std::string         scenario_name      { "Medium Traffic" };
    float               traffic_density    { 1.0f };    ///< Multiplier on default scenario density (higher = more vehicles)
    float               spawn_interval_s   { 2.0f };    ///< Seconds between spawns (derived or custom)
    uint32_t            max_vehicles       { 35 };      ///< Max active vehicles cap
    SignalControllerType controller        { SignalControllerType::FixedTime };
    std::string         controller_name    { "FixedTime" };
    RoutingStrategyType routing_mode       { RoutingStrategyType::Static };
    std::string         routing_name       { "Static" };
    uint32_t            seed               { 42u };
    float               duration_s         { 180.0f };  ///< Simulation duration in seconds
    float               dt_seconds         { 0.1f };
    std::string         model_version      { "none" };  ///< RL policy model tag or file path
    std::string         model_path         { "" };      ///< Actual ONNX path if RL
    bool                enable_emergency   { false };   ///< Enable emergency preemption
    bool                enable_rerouting   { false };   ///< Enable emergency detour rerouting

    [[nodiscard]] Config to_simulation_config() const;
};

// ── ExperimentResult ────────────────────────────────────────────────────────

/// Machine-readable results from a single experiment execution.
struct ExperimentResult {
    ExperimentConfig config;
    RunResult        metrics;
    float            execution_wall_time_s{ 0.0f };
    float            sim_fps              { 0.0f };
    bool             success              { true };
    std::string      error_message        {};

    [[nodiscard]] static std::string csv_header();
    [[nodiscard]] std::string to_csv_row() const;
    [[nodiscard]] std::string to_json_fragment(int indent = 4) const;
};

// ── ExperimentAggregateStat ─────────────────────────────────────────────────

struct MetricStatistics {
    double mean      { 0.0 };
    double std_dev   { 0.0 };
    double min_val   { 0.0 };
    double max_val   { 0.0 };
    double ci95_lower{ 0.0 };
    double ci95_upper{ 0.0 };
};

/// Aggregated statistical record grouping identical configurations across multiple seeds.
struct ExperimentGroupSummary {
    std::string          group_key       {}; // e.g. "FixedTime__Medium__Static__v1"
    std::string          controller_name {};
    std::string          scenario_name   {};
    std::string          routing_name    {};
    std::string          model_version   {};
    float                traffic_density { 1.0f };
    float                duration_s      { 0.0f };
    size_t               sample_count    { 0 };
    std::vector<uint32_t> seeds           {};

    MetricStatistics speed_kmh           {};
    MetricStatistics wait_time_s         {};
    MetricStatistics travel_time_s       {};
    MetricStatistics max_queue           {};
    MetricStatistics avg_queue           {};
    MetricStatistics throughput_vph      {};
    MetricStatistics congestion_pct      {};

    // Emergency metrics (if applicable)
    MetricStatistics emergency_travel_time_s {};
    MetricStatistics normal_travel_time_s    {};

    [[nodiscard]] static std::string csv_header();
    [[nodiscard]] std::string to_csv_row() const;
    [[nodiscard]] std::string to_json_fragment(int indent = 4) const;
};

// ── ExperimentBatch ─────────────────────────────────────────────────────────

/// A collection of experiments to be executed together.
class ExperimentBatch {
public:
    ExperimentBatch() = default;
    explicit ExperimentBatch(std::string batch_name);

    void set_name(std::string name) { name_ = std::move(name); }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    void add_experiment(ExperimentConfig config);
    void clear() { experiments_.clear(); }

    [[nodiscard]] size_t size() const noexcept { return experiments_.size(); }
    [[nodiscard]] const std::vector<ExperimentConfig>& experiments() const noexcept { return experiments_; }
    [[nodiscard]] std::vector<ExperimentConfig>& experiments() noexcept { return experiments_; }

    /// Helper to generate a full parametric grid.
    void add_grid(const std::vector<TrafficScenario>&     scenarios,
                  const std::vector<SignalControllerType>& controllers,
                  const std::vector<RoutingStrategyType>&  routing_modes,
                  const std::vector<uint32_t>&             seeds,
                  float                                    duration_s = 180.0f,
                  const std::vector<float>&                density_multipliers = { 1.0f },
                  const std::string&                       model_version = "traffic_signal_dqn_v1",
                  const std::string&                       model_path = "models/traffic_signal_dqn.onnx");

private:
    std::string                   name_{ "DefaultBatch" };
    std::vector<ExperimentConfig> experiments_;
};

// ── ExperimentRunner ────────────────────────────────────────────────────────

class ExperimentRunner {
public:
    struct RunnerOptions {
        bool   parallel          { false }; ///< Multi-threaded parallel batch execution
        size_t num_threads       { 0 };     ///< 0 = auto-detect hardware concurrency
        bool   verbose           { true };  ///< Log progress to console
        std::function<RoadNetwork()> custom_network_factory{ nullptr };
    };

    ExperimentRunner() = default;
    explicit ExperimentRunner(RunnerOptions options);

    /// Execute a single experiment.
    [[nodiscard]] ExperimentResult run_single(const ExperimentConfig& config) const;

    /// Execute an entire batch of experiments.
    [[nodiscard]] std::vector<ExperimentResult> run_batch(const ExperimentBatch& batch) const;

    /// Aggregate raw experiment results across seeds into grouped statistical summaries.
    [[nodiscard]] static std::vector<ExperimentGroupSummary>
    aggregate_results(const std::vector<ExperimentResult>& results);

    /// Export raw runs to CSV.
    static bool export_runs_csv(const std::vector<ExperimentResult>& results,
                                const std::string&                  filepath);

    /// Export grouped summary with confidence intervals to CSV.
    static bool export_summary_csv(const std::vector<ExperimentGroupSummary>& summaries,
                                   const std::string&                        filepath);

    /// Export complete batch results (metadata, configurations, runs, aggregates) to JSON.
    static bool export_json(const std::string&                         batch_name,
                            const std::vector<ExperimentResult>&       results,
                            const std::vector<ExperimentGroupSummary>& summaries,
                            const std::string&                         filepath);

private:
    RunnerOptions options_{};
};

} // namespace syntraq

