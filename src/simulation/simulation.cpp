//
// src/simulation/simulation.cpp
//

#include "syntraq/simulation/simulation.h"

namespace syntraq {

Simulation::Simulation(Config cfg)
    : cfg_(std::move(cfg))
    , state_{}
{
}

void Simulation::tick() {
    if (!state_.running) return;

    // Milestone 1: bare minimum — advance time and tick count.
    // Future milestones add: vehicle movement, signal control, routing, AI.
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
    state_ = SimState{};
}

} // namespace syntraq
