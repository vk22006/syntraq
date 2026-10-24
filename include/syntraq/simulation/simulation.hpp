#pragma once

//
// syntraq/simulation/simulation.h
//
// The Simulation class owns the discrete-time update loop and all world state.
// It is completely headless — no rendering, no windowing.
// This makes it independently testable.
//

#include "syntraq/core/config.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/world/road_network.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/metrics/traffic_data_collector.hpp"
#include "syntraq/vehicles/vehicle_movement_system.hpp"
#include "syntraq/vehicles/vehicle_spawner.hpp"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace syntraq {

/// Simulation state snapshot — what is known after each tick.
struct SimState {
    uint64_t        tick           { 0 };
    float           elapsed_s      { 0.0f };
    uint32_t        active_vehicles{ 0 };   ///< Vehicles currently on the network
    uint32_t        total_spawned  { 0 };   ///< Cumulative vehicles ever spawned
    uint32_t        total_arrived  { 0 };   ///< Cumulative vehicles that reached destination
    uint32_t        queued_vehicles{ 0 };   ///< Current vehicles stopped in queues
    uint32_t        max_queue_len  { 0 };   ///< Maximum queue depth on any single lane
    float           avg_speed_mps  { 0.0f }; ///< Average speed of active vehicles (m/s)
    float           avg_wait_time_s{ 0.0f }; ///< Average waiting time of vehicles (s)
    float           avg_travel_time_s{ 0.0f }; ///< Average travel time of vehicles (s)
    float           throughput_vph { 0.0f }; ///< Cumulative throughput (veh/h)
    float           congestion_ratio{ 0.0f }; ///< Fraction of active fleet in queue [0.0 .. 1.0]
    TrafficScenario scenario       { TrafficScenario::Medium };
    bool            running        { true };
};

/// Owns the simulation state and drives the update loop.
class Simulation {
public:
    explicit Simulation(Config cfg);
    Simulation(Config cfg, std::function<RoadNetwork()> network_factory);

    /// Advance the simulation by one fixed timestep (cfg_.dt_seconds).
    void tick();

    /// Advance simulation using real elapsed time with a fixed-timestep accumulator.
    void update(float dt_real);

    /// Convenience: run until elapsed_s >= duration_s (headless / tests).
    void run_for(float duration_s);

    /// Reset to t=0 with the same config.
    void reset();

    // ── Simulation controls ───────────────────────────────────────────────
    void set_paused(bool paused) noexcept;
    [[nodiscard]] bool is_paused() const noexcept;

    void set_time_scale(float scale) noexcept;
    [[nodiscard]] float time_scale() const noexcept;

    // ── Traffic Scenarios & Behavior ──────────────────────────────────────
    void set_scenario(TrafficScenario scenario, std::optional<uint32_t> custom_seed = std::nullopt);
    [[nodiscard]] TrafficScenario current_scenario() const noexcept { return cfg_.scenario; }
    [[nodiscard]] const TrafficScenarioConfig& scenario_config() const noexcept { return scenario_cfg_; }

    void set_spawn_interval(float interval_s);
    void set_max_vehicles(uint32_t max);
    void set_seed(uint32_t seed);

    void set_allowed_spawn_origins(std::vector<IntersectionId> origins);
    void set_allowed_spawn_destinations(std::vector<IntersectionId> destinations);
    [[nodiscard]] const std::vector<IntersectionId>& allowed_spawn_origins() const noexcept { return allowed_origins_; }
    [[nodiscard]] const std::vector<IntersectionId>& allowed_spawn_destinations() const noexcept { return allowed_destinations_; }

    [[nodiscard]] uint32_t queued_vehicle_count() const noexcept;
    [[nodiscard]] uint32_t max_queue_length() const noexcept;

    // ── Traffic Signals ───────────────────────────────────────────────────
    void set_signal_controller(TrafficSignalController controller);
    void set_signal_controller_type(SignalControllerType type);
    [[nodiscard]] SignalControllerType signal_controller_type() const noexcept {
        return cfg_.signal_controller_type;
    }

    [[nodiscard]] const TrafficSignalController* signal_controller(IntersectionId id) const noexcept;
    [[nodiscard]]       TrafficSignalController* signal_controller(IntersectionId id)       noexcept;

    [[nodiscard]] const std::unordered_map<IntersectionId, TrafficSignalController>&
        signal_controllers() const noexcept { return signal_controllers_; }

    // ── Routing Engine ───────────────────────────────────────────────────
    void set_route_provider(std::shared_ptr<const IRouteProvider> provider) {
        spawner_.set_route_provider(std::move(provider));
    }
    [[nodiscard]] const IRouteProvider& route_provider() const noexcept {
        return spawner_.route_provider();
    }

    // ── Traffic Metrics Subsystem ─────────────────────────────────────────
    [[nodiscard]] const TrafficMetricsCollector& metrics() const noexcept { return metrics_collector_; }
    [[nodiscard]]       TrafficMetricsCollector& metrics()       noexcept { return metrics_collector_; }
    [[nodiscard]] RunResult get_run_result() const { return metrics_collector_.get_run_result(); }
    [[nodiscard]] bool export_summary_csv(const std::string& filepath, bool append = false) const {
        return metrics_collector_.export_summary_csv(filepath, append);
    }
    [[nodiscard]] bool export_timeseries_csv(const std::string& filepath) const {
        return metrics_collector_.export_timeseries_csv(filepath);
    }

    // ── Training Data Collection Subsystem ────────────────────────────────
    void enable_data_collection(bool enable, float sample_interval_s = 1.0f);
    void set_run_id(std::string id);
    [[nodiscard]] const std::string& run_id() const noexcept { return run_id_; }
    [[nodiscard]] bool is_data_collection_enabled() const noexcept { return data_collection_enabled_; }
    [[nodiscard]] const TrafficDataCollector& data_collector() const noexcept { return data_collector_; }
    [[nodiscard]]       TrafficDataCollector& data_collector()       noexcept { return data_collector_; }
    [[nodiscard]] bool export_training_data_csv(const std::string& filepath,
                                                bool               append = false,
                                                bool               require_30s = true) const {
        return data_collector_.export_csv(filepath, append, require_30s);
    }

    // ── Read-only accessors ───────────────────────────────────────────────
    [[nodiscard]] const SimState&     state()    const noexcept { return state_; }
    [[nodiscard]] const Config&       config()   const noexcept { return cfg_; }
    [[nodiscard]] const RoadNetwork&  network()  const noexcept { return network_; }
    [[nodiscard]]       RoadNetwork&  network()        noexcept { return network_; }
    [[nodiscard]] const std::vector<Vehicle>& vehicles() const noexcept { return vehicles_; }

private:
    void despawn_arrived();
    void init_default_signals();

    Config                cfg_;
    TrafficScenarioConfig scenario_cfg_;
    SimState              state_;
    RoadNetwork           network_;
    std::unordered_map<IntersectionId, TrafficSignalController> signal_controllers_;
    std::vector<Vehicle>  vehicles_;
    VehicleMovementSystem movement_system_;
    VehicleSpawner        spawner_;
    TrafficMetricsCollector metrics_collector_;
    TrafficDataCollector  data_collector_;
    std::string           run_id_{ "run_001" };
    bool                  data_collection_enabled_{ false };
    float                 accumulator_{ 0.0f };
    float                 time_scale_ { 1.0f };
    std::function<RoadNetwork()> network_factory_{};
    std::vector<IntersectionId>  allowed_origins_{};
    std::vector<IntersectionId>  allowed_destinations_{};
};

} // namespace syntraq
