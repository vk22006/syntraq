#pragma once

//
// syntraq/metrics/traffic_data_collector.hpp
//
// Structured traffic dataset generation engine for machine learning.
// Samples simulation states at fixed intervals, extracts multi-dimensional features,
// and binds ground-truth future prediction targets (future queue lengths and congestion)
// across configurable lookahead horizons.
//

#include "syntraq/core/config.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/world/road_network.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace syntraq {

// ── TrafficDataSample ─────────────────────────────────────────────────────────

/// A single structured observation of the simulation at timestamp t,
/// linked to future ground-truth prediction targets at t + H.
struct TrafficDataSample {
    // ── Metadata ─────────────────────────────────────────────────────────────
    std::string run_id              { "run_001" };
    uint32_t    scenario_id         { 0 };     ///< 0: Low, 1: Med, 2: High, 3: RushHour
    std::string scenario_name       { "Medium" };
    uint32_t    seed                { 42u };
    float       timestamp_s         { 0.0f };  ///< Elapsed simulation time (s)
    uint32_t    sample_idx          { 0 };

    // ── Vehicle Count & Flow Features ────────────────────────────────────────
    uint32_t    vehicle_count       { 0 };     ///< Active vehicles currently in network
    uint32_t    total_spawned       { 0 };     ///< Cumulative vehicles spawned up to t
    uint32_t    total_arrived       { 0 };     ///< Cumulative vehicles arrived up to t
    float       spawn_rate_vpm      { 0.0f };  ///< Vehicle inflow rate (veh/min)

    // ── Queue Dynamics Features ──────────────────────────────────────────────
    uint32_t    queue_vehicle_count { 0 };     ///< Active vehicles stopped or < 0.5 m/s
    uint32_t    max_queue_length    { 0 };     ///< Longest queue on any single approach lane
    float       avg_queue_per_lane  { 0.0f };  ///< Mean queue length across active approaches
    float       queue_ratio         { 0.0f };  ///< queued_vehicles / max(1, vehicle_count)

    // ── Speed & Travel Time Features ─────────────────────────────────────────
    float       avg_speed_mps       { 0.0f };  ///< Instantaneous mean vehicle speed (m/s)
    float       avg_speed_kmh       { 0.0f };  ///< Instantaneous mean vehicle speed (km/h)
    float       speed_ratio         { 0.0f };  ///< avg_speed / free_flow_speed_limit (0.0 .. 1.0)
    float       avg_wait_time_s     { 0.0f };  ///< Mean accumulated stopped/wait time (s)
    float       avg_travel_time_s   { 0.0f };  ///< Mean total travel time (s)

    // ── Road Occupancy & Density Features ────────────────────────────────────
    float       road_density_vpm    { 0.0f };  ///< Density: active vehicles / total road length (veh/m)
    float       road_occupancy_pct  { 0.0f };  ///< Overall road network length occupancy %
    float       critical_occupancy_pct{ 0.0f }; ///< Highest single-road occupancy percentage

    // ── Signal Phase & Intersection State Features ───────────────────────────
    int32_t     signal_phase_index  { 0 };     ///< Active phase index
    float       time_in_phase_s     { 0.0f };  ///< Seconds elapsed in current signal phase
    float       phase_duration_s    { 0.0f };  ///< Nominal total duration for current phase
    float       phase_elapsed_ratio { 0.0f };  ///< time_in_phase / phase_duration [0.0 .. 1.0]
    int32_t     is_all_red          { 0 };     ///< 1 if clearance/all-red phase, 0 otherwise
    int32_t     green_approaches_count{ 0 };   ///< Number of approaches with active green signal

    // ── Recent Traffic Statistics (Temporal Window / Momentum) ───────────────
    float       recent_throughput_vph{ 0.0f }; ///< Rolling completed vehicles per hour
    float       recent_queue_delta  { 0.0f };  ///< Rate of queue change dQ/dt (veh/s)
    float       recent_speed_delta  { 0.0f };  ///< Speed trend dv/dt (m/s^2)

    // ── Prediction Targets (Future Ground Truth) ─────────────────────────────
    // Horizon 1: +15.0 seconds
    bool        has_target_15s      { false };
    uint32_t    target_queue_15s    { 0 };     ///< Ground truth queue count at t + 15s
    uint32_t    target_max_queue_15s{ 0 };     ///< Ground truth max queue at t + 15s
    float       target_congestion_15s{ 0.0f }; ///< Ground truth congestion ratio at t + 15s

    // Horizon 2: +30.0 seconds
    bool        has_target_30s      { false };
    uint32_t    target_queue_30s    { 0 };     ///< Ground truth queue count at t + 30s
    uint32_t    target_max_queue_30s{ 0 };     ///< Ground truth max queue at t + 30s
    float       target_congestion_30s{ 0.0f }; ///< Ground truth congestion ratio at t + 30s

    /// Standard CSV column header row
    [[nodiscard]] static std::string csv_header();

    /// Formats sample as a CSV row matching csv_header()
    [[nodiscard]] std::string to_csv_row() const;
};

// ── TrafficDataCollector ──────────────────────────────────────────────────────

/// Subsystem responsible for sampling simulation states, computing features,
/// and retroactively attaching future ground-truth prediction targets.
class TrafficDataCollector {
public:
    TrafficDataCollector(float sample_interval_s = 1.0f,
                         float horizon_1_s       = 15.0f,
                         float horizon_2_s       = 30.0f);

    /// Reset collector for a fresh run
    void reset();

    /// Configure active run metadata
    void set_run_metadata(std::string     run_id,
                          TrafficScenario scenario,
                          std::string     scenario_name,
                          uint32_t        seed);

    /// Update collector at each simulation tick.
    /// Periodically samples features at sample_interval_s and updates future targets.
    void update(float                                                           sim_time_s,
                const RoadNetwork&                                             network,
                const std::vector<Vehicle>&                                    vehicles,
                const std::unordered_map<IntersectionId, TrafficSignalController>& controllers,
                uint32_t                                                       total_spawned,
                uint32_t                                                       total_arrived,
                float                                                          spawn_interval_s);

    /// Finalize collection at simulation completion
    void finalize();

    /// Access all collected samples
    [[nodiscard]] const std::vector<TrafficDataSample>& samples() const noexcept { return samples_; }

    /// Access only samples that have resolved future targets
    [[nodiscard]] std::vector<TrafficDataSample> valid_samples(bool require_30s = true) const;

    /// Export collected samples to a structured CSV file
    [[nodiscard]] bool export_csv(const std::string& filepath,
                                  bool               append = false,
                                  bool               require_30s = true) const;

    [[nodiscard]] float sample_interval_s() const noexcept { return sample_interval_s_; }
    [[nodiscard]] float horizon_1_s() const noexcept { return horizon_1_s_; }
    [[nodiscard]] float horizon_2_s() const noexcept { return horizon_2_s_; }

private:
    float sample_interval_s_{ 1.0f };
    float horizon_1_s_      { 15.0f };
    float horizon_2_s_      { 30.0f };
    float last_sample_time_ { -999.0f };
    uint32_t next_sample_idx_{ 0 };

    std::string     run_id_       { "run_001" };
    TrafficScenario scenario_     { TrafficScenario::Medium };
    std::string     scenario_name_{ "Medium Traffic" };
    uint32_t        seed_         { 42u };

    // Rolling history for computing windowed statistics and deltas
    struct HistoryPoint {
        float    timestamp_s{ 0.0f };
        uint32_t queue_count{ 0 };
        float    avg_speed_mps{ 0.0f };
        uint32_t total_arrived{ 0 };
    };
    std::deque<HistoryPoint> rolling_history_;

    // All collected samples for this run
    std::vector<TrafficDataSample> samples_;

    // Indices of samples pending target resolution
    std::deque<size_t> pending_h1_indices_;
    std::deque<size_t> pending_h2_indices_;
};

} // namespace syntraq
