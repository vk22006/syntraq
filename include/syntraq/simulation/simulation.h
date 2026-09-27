#pragma once

//
// syntraq/simulation/simulation.h
//
// The Simulation class owns the discrete-time update loop and the world state.
// It is completely headless — no rendering, no windowing.
// This makes it independently testable.
//

#include "syntraq/core/config.h"
#include "syntraq/world/road_network.h"
#include <cstdint>

namespace syntraq {

/// Simulation state snapshot — what is known after each tick.
struct SimState {
    uint64_t tick       { 0 };    ///< Number of ticks completed
    float    elapsed_s  { 0.0f }; ///< Total simulated time (seconds)
    bool     running    { true };
};

/// Owns the simulation state and drives the update loop.
/// The Renderer reads SimState and the RoadNetwork for display;
/// it never writes to Simulation.
class Simulation {
public:
    explicit Simulation(Config cfg);

    /// Advance the simulation by one fixed timestep (cfg_.dt_seconds).
    void tick();

    /// Convenience: run until elapsed_s >= duration_s (for headless/tests).
    void run_for(float duration_s);

    /// Read-only access to current state.
    [[nodiscard]] const SimState&     state()   const noexcept { return state_; }
    [[nodiscard]] const Config&       config()  const noexcept { return cfg_; }
    [[nodiscard]] const RoadNetwork&  network() const noexcept { return network_; }
    [[nodiscard]]       RoadNetwork&  network()       noexcept { return network_; }

    /// Reset the simulation to t=0 with the same config.
    void reset();

private:
    Config      cfg_;
    SimState    state_;
    RoadNetwork network_;
};

} // namespace syntraq
