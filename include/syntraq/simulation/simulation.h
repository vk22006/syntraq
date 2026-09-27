#pragma once

//
// syntraq/simulation/simulation.h
//
// The Simulation class owns the discrete-time update loop and all world state.
// It is completely headless — no rendering, no windowing.
// This makes it independently testable.
//

#include "syntraq/core/config.h"
#include "syntraq/world/road_network.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/vehicles/vehicle_movement_system.h"
#include "syntraq/vehicles/vehicle_spawner.h"

#include <cstdint>
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

    /// Convenience: run until elapsed_s >= duration_s (headless / tests).
    void run_for(float duration_s);

    /// Reset to t=0 with the same config.
    void reset();

    // ── Read-only accessors ───────────────────────────────────────────────
    [[nodiscard]] const SimState&     state()    const noexcept { return state_; }
    [[nodiscard]] const Config&       config()   const noexcept { return cfg_; }
    [[nodiscard]] const RoadNetwork&  network()  const noexcept { return network_; }
    [[nodiscard]]       RoadNetwork&  network()        noexcept { return network_; }
    [[nodiscard]] const std::vector<Vehicle>& vehicles() const noexcept { return vehicles_; }

private:
    void despawn_arrived();

    Config                cfg_;
    SimState              state_;
    RoadNetwork           network_;
    std::vector<Vehicle>  vehicles_;
    VehicleMovementSystem movement_system_;
    VehicleSpawner        spawner_;
};

} // namespace syntraq
