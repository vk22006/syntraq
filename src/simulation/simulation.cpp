//
// src/simulation/simulation.cpp
//

#include "syntraq/simulation/simulation.h"
#include "syntraq/world/road_network.h"

#include <algorithm>

namespace syntraq {

Simulation::Simulation(Config cfg)
    : cfg_        (std::move(cfg))
    , state_      {}
    , network_    (make_test_map())
    , spawner_    (cfg_.seed,
                   /* spawn_interval_s = */ 2.0f,
                   /* max_vehicles     = */ 40)
    , accumulator_(0.0f)
    , time_scale_ (1.0f)
{
}

void Simulation::tick() {
    if (!state_.running) return;

    // ── 1. Try to spawn a new vehicle ─────────────────────────────────────
    auto opt = spawner_.try_spawn(network_,
                                  static_cast<uint32_t>(vehicles_.size()),
                                  state_.elapsed_s);
    if (opt.has_value()) {
        vehicles_.push_back(std::move(*opt));
        state_.total_spawned++;
    }

    // ── 2. Move all vehicles ──────────────────────────────────────────────
    movement_system_.update(vehicles_, network_, cfg_.dt_seconds);

    // ── 3. Despawn arrived vehicles ───────────────────────────────────────
    despawn_arrived();

    // ── 4. Advance simulation clock ───────────────────────────────────────
    state_.elapsed_s      += cfg_.dt_seconds;
    state_.tick           += 1;
    state_.active_vehicles = static_cast<uint32_t>(vehicles_.size());
}

void Simulation::update(float dt_real) {
    if (!state_.running) return;

    accumulator_ += dt_real * time_scale_;

    // Cap maximum ticks per update to prevent spiral of death during stalls
    constexpr uint32_t kMaxTicksPerUpdate = 100;
    constexpr float    kEpsilon           = 1e-5f;
    uint32_t ticks_done = 0;

    while ((accumulator_ + kEpsilon) >= cfg_.dt_seconds && ticks_done < kMaxTicksPerUpdate) {
        tick();
        accumulator_ -= cfg_.dt_seconds;
        ticks_done++;
    }

    if (accumulator_ < 0.0f) {
        accumulator_ = 0.0f;
    }

    // Discard runaway backlog if stall limit was hit
    if (ticks_done >= kMaxTicksPerUpdate) {
        accumulator_ = 0.0f;
    }
}

void Simulation::run_for(float duration_s) {
    reset();
    while (state_.elapsed_s < duration_s) {
        tick();
    }
}

void Simulation::reset() {
    state_       = SimState{};
    network_     = make_test_map();
    vehicles_.clear();
    spawner_     = VehicleSpawner{ cfg_.seed, 2.0f, 40 };
    accumulator_ = 0.0f;
}

void Simulation::set_paused(bool paused) noexcept {
    state_.running = !paused;
    if (paused) {
        accumulator_ = 0.0f;
    }
}

bool Simulation::is_paused() const noexcept {
    return !state_.running;
}

void Simulation::set_time_scale(float scale) noexcept {
    if (scale >= 0.0f) {
        time_scale_ = scale;
    }
}

float Simulation::time_scale() const noexcept {
    return time_scale_;
}

void Simulation::despawn_arrived() {
    const auto before = static_cast<uint32_t>(vehicles_.size());

    vehicles_.erase(
        std::remove_if(vehicles_.begin(), vehicles_.end(),
                       [](const Vehicle& v) {
                           return v.state == VehicleState::Arrived;
                       }),
        vehicles_.end());

    const auto after   = static_cast<uint32_t>(vehicles_.size());
    state_.total_arrived += (before - after);
}

} // namespace syntraq
