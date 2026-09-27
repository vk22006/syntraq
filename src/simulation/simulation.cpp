//
// src/simulation/simulation.cpp
//

#include "syntraq/simulation/simulation.h"
#include "syntraq/world/road_network.h"

namespace syntraq {

Simulation::Simulation(Config cfg)
    : cfg_(std::move(cfg))
    , state_{}
    , network_(make_test_map())  // Default: load the built-in test map
{
}

void Simulation::tick() {
    if (!state_.running) return;

    // Advance time. Future milestones add vehicle movement, signal control, etc.
    state_.elapsed_s += cfg_.dt_seconds;
    state_.tick      += 1;
}

void Simulation::run_for(float duration_s) {
    reset();
    while (state_.elapsed_s < duration_s) {
        tick();
    }
}

void Simulation::reset() {
    state_   = SimState{};
    network_ = make_test_map();
}

} // namespace syntraq
