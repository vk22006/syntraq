#pragma once

//
// syntraq/metrics/traffic_metrics.h
//
// Traffic metrics subsystem independent from rendering.
// Tracks vehicle counts, speeds, waiting times, travel times, queue lengths,
// throughput, and congestion ratio.
// Provides RunResult summary, time-series logging, and CSV export.
//

#include "syntraq/core/config.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/world/road_network.h"

#include <cstdint>
#include <string>
#include <vector>

namespace syntraq {

// ── MetricSnapshot ────────────────────────────────────────────────────────────

/// Instantaneous snapshot of traffic metrics captured at a specific simulation tick.
struct MetricSnapshot {
    uint64_t tick            { 0 };
    float    timestamp_s     { 0.0f };
    uint32_t active_vehicles { 0 };   ///< Currently moving or stopped on network
    uint32_t queued_vehicles { 0 };   ///< Vehicles with speed < 0.5 m/s or stopped
    uint32_t total_spawned   { 0 };   ///< Cumulative vehicles spawned
    uint32_t total_completed { 0 };   ///< Cumulative vehicles arrived at destination
    float    avg_speed_mps   { 0.0f }; ///< Average speed of active vehicles (m/s)
    float    avg_waiting_time_s{ 0.0f }; ///< Average wait time of active vehicles (s)
    float    avg_travel_time_s { 0.0f }; ///< Average travel time of active vehicles (s)
    uint32_t max_queue_length{ 0 };   ///< Longest queue on any single lane at this tick
    float    throughput_vph  { 0.0f }; ///< Cumulative throughput (vehicles per hour)
    float    congestion_ratio{ 0.0f }; ///< queued_vehicles / active_vehicles [0.0 .. 1.0]
};

// ── RunResult ─────────────────────────────────────────────────────────────────

/// Comprehensive end-of-run simulation metrics summary.
struct RunResult {
    TrafficScenario scenario        { TrafficScenario::Medium };
    std::string     scenario_name   { "Medium Traffic" };
    uint32_t        seed            { 42u };
    float           duration_s      { 0.0f };
    uint64_t        total_ticks     { 0 };

    // Vehicle counts
    uint32_t        total_spawned   { 0 };
    uint32_t        total_completed { 0 };
    uint32_t        active_remaining{ 0 };

    // Speed metrics
    float           avg_speed_mps   { 0.0f }; ///< Overall average travel speed (m/s)
    float           avg_speed_kmh   { 0.0f }; ///< Overall average travel speed (km/h)

    // Timing metrics
    float           avg_waiting_time_s{ 0.0f }; ///< Mean time vehicles spent queued/stopped (s)
    float           avg_travel_time_s { 0.0f }; ///< Mean total trip duration (s)

    // Queue metrics
    uint32_t        max_queue_length{ 0 };   ///< Peak queue length observed on any lane
    float           avg_queue_length{ 0.0f }; ///< Time-mean queued vehicle count across simulation

    // Throughput & Flow
    float           throughput_vps  { 0.0f }; ///< Completed vehicles per simulated second
    float           throughput_vph  { 0.0f }; ///< Completed vehicles per simulated hour (vps * 3600)
    float           congestion_ratio{ 0.0f }; ///< Time-mean fraction of active vehicles in queue

    /// Returns standard CSV header row for summary runs.
    [[nodiscard]] static std::string csv_header();

    /// Formats this RunResult as a single CSV row matching csv_header().
    [[nodiscard]] std::string to_csv_row() const;

    /// Exports summary result to a CSV file. If file does not exist, writes header first.
    /// If append is true and file exists, appends the row; otherwise overwrites.
    [[nodiscard]] bool export_csv(const std::string& filepath, bool append = false) const;
};

// ── TrafficMetricsCollector ───────────────────────────────────────────────────

/// Subsystem for gathering, aggregating, and exporting traffic metrics.
/// Runs independently of any rendering or GUI code.
class TrafficMetricsCollector {
public:
    TrafficMetricsCollector();

    /// Configure active scenario metadata.
    void set_scenario(TrafficScenario scenario, std::string scenario_name);

    /// Configure active RNG seed for repeatable recording.
    void set_seed(uint32_t seed);

    /// Record a simulation tick with active vehicles on the network.
    void record_tick(float                       dt,
                     const std::vector<Vehicle>& active_vehicles,
                     uint32_t                    total_spawned_cumulative,
                     uint64_t                    tick_number);

    /// Record vehicle arrival when reaching destination (called prior to despawn).
    void record_arrival(const Vehicle& arrived_vehicle);

    /// Reset all metric counters, accumulators, and time-series history.
    void reset();

    /// Latest tick snapshot.
    [[nodiscard]] const MetricSnapshot& current_snapshot() const noexcept { return current_snapshot_; }

    /// Full history of recorded snapshots.
    [[nodiscard]] const std::vector<MetricSnapshot>& timeseries() const noexcept { return timeseries_; }

    /// Compile cumulative metrics into final RunResult.
    [[nodiscard]] RunResult get_run_result() const;

    /// Export single-row summary CSV of the current run.
    [[nodiscard]] bool export_summary_csv(const std::string& filepath, bool append = false) const;

    /// Export full chronological time-series to CSV.
    [[nodiscard]] bool export_timeseries_csv(const std::string& filepath) const;

private:
    TrafficScenario scenario_     { TrafficScenario::Medium };
    std::string     scenario_name_{ "Medium Traffic" };
    uint32_t        seed_         { 42u };

    float           elapsed_s_    { 0.0f };
    uint64_t        total_ticks_  { 0 };
    uint32_t        total_spawned_{ 0 };
    uint32_t        total_completed_{ 0 };

    // Completed vehicle accumulators
    double          completed_travel_time_s_{ 0.0 };
    double          completed_wait_time_s_  { 0.0 };
    double          completed_distance_m_   { 0.0 };

    // Time-averaged fleet accumulators
    double          sum_speed_samples_      { 0.0 };
    uint64_t        speed_sample_count_     { 0 };
    double          sum_queued_vehicles_    { 0.0 };
    double          sum_congestion_ratio_   { 0.0 };
    uint32_t        peak_max_queue_         { 0 };

    MetricSnapshot              current_snapshot_{};
    std::vector<MetricSnapshot> timeseries_{};
};

} // namespace syntraq
