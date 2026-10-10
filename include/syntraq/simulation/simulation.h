#pragma once

//
// syntraq/simulation/simulation.h
//
// The Simulation class owns the discrete-time update loop and all world state.
// It is completely headless — no rendering, no windowing.
// This makes it independently testable.
//

#include "syntraq/core/config.h"
#include "syntraq/signals/traffic_signal_controller.h"
#include "syntraq/world/road_network.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/vehicles/vehicle_movement_system.h"
#include "syntraq/vehicles/vehicle_spawner.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace syntraq {

/// Simulation state snapshot — what is known after each tick.
struct SimState {
    uint64_t tick          { 0 };
    float    elapsed_s     { 0.0f };
    uint32_t active_vehicles{ 0 };   ///< Vehicles currently on the network
    uint32_t total_spawned { 0 };    ///< Cumulative vehicles ever spawned
    uint32_t total_arrived { 0 };    ///< Cumulative vehicles that reached destination
    bool     running       { true };
};

/// Owns the simulation state and drives the update loop.
class Simulation {
public:
    explicit Simulation(Config cfg);

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

    // ── Traffic Signals ───────────────────────────────────────────────────
    void set_signal_controller(TrafficSignalController controller);

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
    SimState              state_;
    RoadNetwork           network_;
    std::unordered_map<IntersectionId, TrafficSignalController> signal_controllers_;
    std::vector<Vehicle>  vehicles_;
    VehicleMovementSystem movement_system_;
    VehicleSpawner        spawner_;
    float                 accumulator_{ 0.0f };
    float                 time_scale_ { 1.0f };
};

} // namespace syntraq
