//
// src/simulation/simulation.cpp
//

#include "syntraq/simulation/simulation.h"
#include "syntraq/world/road_network.h"

#include <algorithm>
#include <map>

namespace syntraq {

Simulation::Simulation(Config cfg)
    : cfg_          (std::move(cfg))
    , scenario_cfg_ (get_scenario_preset(cfg_.scenario, cfg_.seed))
    , state_        {}
    , network_      (make_test_map())
    , spawner_      (cfg_.seed,
                     cfg_.vehicle_spawn_interval_s,
                     cfg_.max_vehicles)
    , accumulator_  (0.0f)
    , time_scale_   (1.0f)
{
    // If config specified default scenario, reflect presets
    if (cfg_.scenario != TrafficScenario::Custom) {
        scenario_cfg_ = get_scenario_preset(cfg_.scenario, cfg_.seed);
        spawner_.spawn_interval_s = scenario_cfg_.spawn_interval_s;
        spawner_.max_vehicles     = scenario_cfg_.max_vehicles;
    }
    state_.scenario = cfg_.scenario;
    init_default_signals();
}

void Simulation::init_default_signals() {
    signal_controllers_.clear();
    if (!cfg_.enable_traffic_signals) return;

    for (const auto& [id, inter] : network_.intersections()) {
        const auto incoming = network_.incoming_roads(id);
        if (incoming.size() > 1) {
            auto c = TrafficSignalController::make_fixed_time(
                network_,
                id,
                cfg_.default_green_duration_s,
                cfg_.default_yellow_duration_s,
                cfg_.default_all_red_duration_s);
            if (c.is_valid()) {
                signal_controllers_[id] = std::move(c);
            }
        }
    }
}

void Simulation::tick() {
    if (!state_.running) return;

    // ── 1. Advance traffic signal controllers ─────────────────────────────
    for (auto& [id, controller] : signal_controllers_) {
        controller.tick(cfg_.dt_seconds);
    }

    // ── 2. Try to spawn a new vehicle (with lane occupancy checks) ────────
    auto opt = spawner_.try_spawn(network_,
                                  vehicles_,
                                  state_.elapsed_s);
    if (opt.has_value()) {
        vehicles_.push_back(std::move(*opt));
        state_.total_spawned++;
    }

    // ── 3. Move all vehicles (respecting traffic signals and car-following)
    movement_system_.update(vehicles_, network_, signal_controllers_, cfg_.dt_seconds);

    // ── 4. Despawn arrived vehicles ───────────────────────────────────────
    despawn_arrived();

    // ── 5. Advance simulation clock & update metrics ──────────────────────
    state_.elapsed_s       += cfg_.dt_seconds;
    state_.tick            += 1;
    state_.active_vehicles  = static_cast<uint32_t>(vehicles_.size());
    state_.queued_vehicles  = queued_vehicle_count();
    state_.max_queue_len    = max_queue_length();
    state_.scenario         = cfg_.scenario;
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
    state_                 = SimState{};
    state_.scenario        = cfg_.scenario;
    network_               = make_test_map();
    vehicles_.clear();
    spawner_               = VehicleSpawner{ scenario_cfg_.seed,
                                             scenario_cfg_.spawn_interval_s,
                                             scenario_cfg_.max_vehicles };
    accumulator_           = 0.0f;
    for (auto& [id, controller] : signal_controllers_) {
        controller.reset();
    }
}

void Simulation::set_scenario(TrafficScenario scenario, std::optional<uint32_t> custom_seed) {
    cfg_.scenario  = scenario;
    const uint32_t seed = custom_seed.value_or(cfg_.seed);
    scenario_cfg_  = get_scenario_preset(scenario, seed);

    cfg_.vehicle_spawn_interval_s = scenario_cfg_.spawn_interval_s;
    cfg_.max_vehicles             = scenario_cfg_.max_vehicles;
    cfg_.seed                     = scenario_cfg_.seed;

    spawner_.spawn_interval_s = scenario_cfg_.spawn_interval_s;
    spawner_.max_vehicles     = scenario_cfg_.max_vehicles;
    spawner_.set_seed(scenario_cfg_.seed);
    state_.scenario           = scenario;
}

void Simulation::set_spawn_interval(float interval_s) {
    if (interval_s > 0.0f) {
        cfg_.vehicle_spawn_interval_s   = interval_s;
        scenario_cfg_.spawn_interval_s  = interval_s;
        spawner_.spawn_interval_s       = interval_s;
    }
}

void Simulation::set_max_vehicles(uint32_t max) {
    cfg_.max_vehicles          = max;
    scenario_cfg_.max_vehicles = max;
    spawner_.max_vehicles      = max;
}

void Simulation::set_seed(uint32_t seed) {
    cfg_.seed          = seed;
    scenario_cfg_.seed = seed;
    spawner_.set_seed(seed);
}

uint32_t Simulation::queued_vehicle_count() const noexcept {
    uint32_t count = 0;
    for (const auto& v : vehicles_) {
        if (v.state == VehicleState::Stopped || (v.state == VehicleState::Moving && v.speed_mps < 0.5f)) {
            ++count;
        }
    }
    return count;
}

uint32_t Simulation::max_queue_length() const noexcept {
    std::map<std::pair<RoadId, uint32_t>, uint32_t> lane_queues;
    for (const auto& v : vehicles_) {
        if (v.state == VehicleState::Stopped || (v.state == VehicleState::Moving && v.speed_mps < 0.5f)) {
            lane_queues[{v.current_road, v.lane_index}]++;
        }
    }

    uint32_t max_len = 0;
    for (const auto& [lane, len] : lane_queues) {
        if (len > max_len) {
            max_len = len;
        }
    }
    return max_len;
}

void Simulation::set_signal_controller(TrafficSignalController controller) {
    const auto id = controller.intersection_id();
    signal_controllers_[id] = std::move(controller);
}

const TrafficSignalController* Simulation::signal_controller(IntersectionId id) const noexcept {
    const auto it = signal_controllers_.find(id);
    if (it != signal_controllers_.end()) {
        return &it->second;
    }
    return nullptr;
}

TrafficSignalController* Simulation::signal_controller(IntersectionId id) noexcept {
    const auto it = signal_controllers_.find(id);
    if (it != signal_controllers_.end()) {
        return &it->second;
    }
    return nullptr;
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
