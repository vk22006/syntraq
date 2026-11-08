//
// src/simulation/simulation.cpp
//

#include "syntraq/simulation/simulation.hpp"
#include "syntraq/routing/static_routing.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <filesystem>
#include <map>

namespace syntraq {

Simulation::Simulation(Config cfg)
    : Simulation(std::move(cfg), [] { return make_test_map(); })
{
}

Simulation::Simulation(Config cfg, std::function<RoadNetwork()> network_factory)
    : cfg_          (std::move(cfg))
    , scenario_cfg_ (get_scenario_preset(cfg_.scenario, cfg_.seed))
    , state_        {}
    , network_      (network_factory ? network_factory() : make_test_map())
    , spawner_      (cfg_.seed,
                     cfg_.vehicle_spawn_interval_s,
                     cfg_.max_vehicles)
    , accumulator_  (0.0f)
    , time_scale_   (1.0f)
    , network_factory_(std::move(network_factory))
{
    // If config specified default scenario, reflect presets
    if (cfg_.scenario != TrafficScenario::Custom) {
        scenario_cfg_ = get_scenario_preset(cfg_.scenario, cfg_.seed);
        spawner_.spawn_interval_s = scenario_cfg_.spawn_interval_s;
        spawner_.max_vehicles     = scenario_cfg_.max_vehicles;
    }
    state_.scenario = cfg_.scenario;
    metrics_collector_.set_scenario(cfg_.scenario, scenario_cfg_.name);
    metrics_collector_.set_seed(cfg_.seed);
    data_collector_.set_run_metadata("run_001", cfg_.scenario, scenario_cfg_.name, cfg_.seed);
    init_default_signals();
    init_default_routing();
}

void Simulation::init_default_signals() {
    signal_controllers_.clear();
    if (!cfg_.enable_traffic_signals) return;

    for (const auto& [id, inter] : network_.intersections()) {
        const auto incoming = network_.incoming_roads(id);
        if (incoming.size() > 1) {
            TrafficSignalController c;
            if (cfg_.signal_controller_type == SignalControllerType::Adaptive) {
                AdaptiveControllerConfig ac_cfg;
                ac_cfg.nominal_green_s = cfg_.default_green_duration_s;
                c = TrafficSignalController::make_adaptive(
                    network_,
                    id,
                    ac_cfg,
                    cfg_.default_yellow_duration_s,
                    cfg_.default_all_red_duration_s);
            } else if (cfg_.signal_controller_type == SignalControllerType::ReinforcementLearning) {
                RLControllerConfig rl_cfg;
                rl_cfg.min_green_s = cfg_.default_green_duration_s;
                rl_cfg.yellow_s    = cfg_.default_yellow_duration_s;
                rl_cfg.all_red_s   = cfg_.default_all_red_duration_s;
                rl_cfg.model_path  = cfg_.rl_model_path;

                if (!shared_rl_policy_ && cfg_.enable_rl_model) {
                    shared_rl_policy_ = std::make_shared<RLPolicyInference>();
                    std::string mpath = cfg_.rl_model_path;
                    if (!std::filesystem::exists(mpath)) {
                        if (std::filesystem::exists("../" + mpath)) mpath = "../" + mpath;
                        else if (std::filesystem::exists("../../" + mpath)) mpath = "../../" + mpath;
                    }
                    if (std::filesystem::exists(mpath)) {
                        shared_rl_policy_->loadModel(mpath);
                    }
                }
                c = TrafficSignalController::make_rl(network_, id, rl_cfg, shared_rl_policy_);
            } else if (cfg_.signal_controller_type == SignalControllerType::Coordinated) {
                CoordinatedControllerConfig coord_cfg;
                coord_cfg.nominal_green_s = cfg_.default_green_duration_s;
                c = TrafficSignalController::make_coordinated(
                    network_,
                    id,
                    coord_cfg,
                    cfg_.default_yellow_duration_s,
                    cfg_.default_all_red_duration_s);
            } else {
                c = TrafficSignalController::make_fixed_time(
                    network_,
                    id,
                    cfg_.default_green_duration_s,
                    cfg_.default_yellow_duration_s,
                    cfg_.default_all_red_duration_s);
            }
            if (c.is_valid()) {
                signal_controllers_[id] = std::move(c);
            }
        }
    }
    coordination_manager_.build_topology(network_, signal_controllers_);
}

void Simulation::tick() {
    if (!state_.running) return;

    // ── 1. Ingest approach queues & wait times, then advance signal controllers ───
    std::unordered_map<RoadId, uint32_t> road_queues;
    std::unordered_map<RoadId, float>    road_max_waits;
    for (const auto& v : vehicles_) {
        if (v.state == VehicleState::Arrived) continue;
        if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
            road_queues[v.current_road]++;
            if (v.wait_time_s > road_max_waits[v.current_road]) {
                road_max_waits[v.current_road] = v.wait_time_s;
            }
        }
    }

    // ── Exchange network-level coordination information among neighboring signals ──
    coordination_manager_.exchange_coordination_states(network_, signal_controllers_, road_queues, road_max_waits);

    for (auto& [id, controller] : signal_controllers_) {
        controller.observe_traffic(road_queues, road_max_waits);
        if (auto* rlc = controller.as<RLController>()) {
            if (rlc->is_decision_due()) {
                const auto obs = compute_rl_observation(id);
                rlc->receive_observation(obs);
                rlc->run_inference();
                const int act = rlc->produce_action();
                rlc->apply_action(act);
            }
        }
        controller.tick(cfg_.dt_seconds);
    }

    // ── Update real-time conditions for traffic-aware routing ───────────
    if (traffic_aware_router_) {
        traffic_aware_router_->update_from_simulation(network_, vehicles_);
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

    metrics_collector_.record_tick(cfg_.dt_seconds,
                                   vehicles_,
                                   state_.total_spawned,
                                   state_.tick);

    const auto& snap        = metrics_collector_.current_snapshot();
    state_.active_vehicles  = snap.active_vehicles;
    state_.queued_vehicles  = snap.queued_vehicles;
    state_.max_queue_len    = snap.max_queue_length;
    state_.avg_speed_mps    = snap.avg_speed_mps;
    state_.avg_wait_time_s  = snap.avg_waiting_time_s;
    state_.avg_travel_time_s= snap.avg_travel_time_s;
    state_.throughput_vph   = snap.throughput_vph;
    state_.congestion_ratio = snap.congestion_ratio;
    state_.scenario         = cfg_.scenario;

    if (data_collection_enabled_) {
        data_collector_.update(state_.elapsed_s,
                               network_,
                               vehicles_,
                               signal_controllers_,
                               state_.total_spawned,
                               state_.total_arrived,
                               scenario_cfg_.spawn_interval_s);
    }
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

    // Update measured simulation FPS over real-time window
    real_time_window_ += dt_real;
    ticks_in_window_  += ticks_done;
    if (real_time_window_ >= 0.20f) {
        state_.simulation_fps = static_cast<float>(ticks_in_window_) / real_time_window_;
        ticks_in_window_  = 0;
        real_time_window_ = 0.0f;
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
    network_               = network_factory_ ? network_factory_() : make_test_map();
    vehicles_.clear();
    spawner_               = VehicleSpawner{ scenario_cfg_.seed,
                                             scenario_cfg_.spawn_interval_s,
                                             scenario_cfg_.max_vehicles };
    if (!allowed_origins_.empty()) {
        spawner_.allowed_origins = allowed_origins_;
    }
    if (!allowed_destinations_.empty()) {
        spawner_.allowed_destinations = allowed_destinations_;
    }
    init_default_routing();
    accumulator_           = 0.0f;
    real_time_window_      = 0.0f;
    ticks_in_window_       = 0;
    for (auto& [id, controller] : signal_controllers_) {
        controller.reset();
    }
    coordination_manager_.build_topology(network_, signal_controllers_);
    metrics_collector_.reset();
    metrics_collector_.set_scenario(cfg_.scenario, scenario_cfg_.name);
    metrics_collector_.set_seed(scenario_cfg_.seed);
    data_collector_.reset();
    data_collector_.set_run_metadata(run_id_, cfg_.scenario, scenario_cfg_.name, scenario_cfg_.seed);
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

    metrics_collector_.set_scenario(scenario, scenario_cfg_.name);
    metrics_collector_.set_seed(scenario_cfg_.seed);
    data_collector_.set_run_metadata(run_id_, scenario, scenario_cfg_.name, scenario_cfg_.seed);
}

void Simulation::set_run_id(std::string id) {
    run_id_ = std::move(id);
    data_collector_.set_run_metadata(run_id_, cfg_.scenario, scenario_cfg_.name, scenario_cfg_.seed);
}

void Simulation::enable_data_collection(bool enable, float sample_interval_s) {
    data_collection_enabled_ = enable;
    if (enable) {
        data_collector_ = TrafficDataCollector(sample_interval_s);
        data_collector_.set_run_metadata(run_id_, cfg_.scenario, scenario_cfg_.name, scenario_cfg_.seed);
    }
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
    metrics_collector_.set_seed(seed);
}

void Simulation::set_allowed_spawn_origins(std::vector<IntersectionId> origins) {
    allowed_origins_ = std::move(origins);
    spawner_.allowed_origins = allowed_origins_;
}

void Simulation::set_allowed_spawn_destinations(std::vector<IntersectionId> destinations) {
    allowed_destinations_ = std::move(destinations);
    spawner_.allowed_destinations = allowed_destinations_;
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

void Simulation::set_signal_controller_type(SignalControllerType type) {
    cfg_.signal_controller_type = type;
    init_default_signals();
}

void Simulation::set_routing_strategy_type(RoutingStrategyType type) {
    cfg_.routing_strategy = type;
    init_default_routing();
}

void Simulation::init_default_routing() {
    if (cfg_.routing_strategy == RoutingStrategyType::TrafficAware) {
        traffic_aware_router_ = std::make_shared<TrafficAwareRouting>();
        spawner_.set_route_provider(traffic_aware_router_);
    } else {
        traffic_aware_router_.reset();
        spawner_.set_route_provider(std::make_shared<StaticRouting>());
    }
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

    for (const auto& v : vehicles_) {
        if (v.state == VehicleState::Arrived) {
            metrics_collector_.record_arrival(v);
        }
    }

    vehicles_.erase(
        std::remove_if(vehicles_.begin(), vehicles_.end(),
                       [](const Vehicle& v) {
                           return v.state == VehicleState::Arrived;
                       }),
        vehicles_.end());

    const auto after   = static_cast<uint32_t>(vehicles_.size());
    state_.total_arrived += (before - after);
}

std::vector<float> Simulation::compute_rl_observation(IntersectionId id) const {
    std::vector<float> obs(12, 0.0f);
    const auto it = signal_controllers_.find(id);
    if (it == signal_controllers_.end()) {
        return obs;
    }
    const auto& ctrl = it->second;
    if (ctrl.phases().size() < 2) {
        return obs;
    }

    const auto& p0_roads = ctrl.phases()[0].green_roads;
    const auto& p1_roads = ctrl.phases()[1].green_roads;

    float q_p0 = 0.0f, q_p1 = 0.0f;
    float app_p0 = 0.0f, app_p1 = 0.0f;
    float spd_sum_p0 = 0.0f, spd_sum_p1 = 0.0f;
    uint32_t cnt_p0 = 0, cnt_p1 = 0;
    float wait_sum_p0 = 0.0f, wait_sum_p1 = 0.0f;

    for (const auto& v : vehicles_) {
        if (v.state == VehicleState::Arrived) continue;

        const bool in_p0 = std::find(p0_roads.begin(), p0_roads.end(), v.current_road) != p0_roads.end();
        if (in_p0) {
            spd_sum_p0 += v.speed_mps;
            cnt_p0++;
            if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
                q_p0 += 1.0f;
                wait_sum_p0 += v.wait_time_s;
            } else {
                app_p0 += 1.0f;
            }
            continue;
        }

        const bool in_p1 = std::find(p1_roads.begin(), p1_roads.end(), v.current_road) != p1_roads.end();
        if (in_p1) {
            spd_sum_p1 += v.speed_mps;
            cnt_p1++;
            if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
                q_p1 += 1.0f;
                wait_sum_p1 += v.wait_time_s;
            } else {
                app_p1 += 1.0f;
            }
            continue;
        }
    }

    constexpr float kMaxQueueNorm = 20.0f;
    constexpr float kMaxSpeedNorm = 13.89f; // 50 km/h
    constexpr float kMaxWaitNorm  = 60.0f;  // 60 seconds

    const float spd_p0  = (cnt_p0 > 0) ? (spd_sum_p0 / static_cast<float>(cnt_p0)) : kMaxSpeedNorm;
    const float spd_p1  = (cnt_p1 > 0) ? (spd_sum_p1 / static_cast<float>(cnt_p1)) : kMaxSpeedNorm;
    const float wait_p0 = (q_p0 > 0.0f) ? (wait_sum_p0 / q_p0) : 0.0f;
    const float wait_p1 = (q_p1 > 0.0f) ? (wait_sum_p1 / q_p1) : 0.0f;

    obs[0] = std::clamp(q_p0 / kMaxQueueNorm, 0.0f, 1.0f);
    obs[1] = std::clamp(q_p1 / kMaxQueueNorm, 0.0f, 1.0f);
    obs[2] = std::clamp(app_p0 / kMaxQueueNorm, 0.0f, 1.0f);
    obs[3] = std::clamp(app_p1 / kMaxQueueNorm, 0.0f, 1.0f);
    obs[4] = std::clamp(spd_p0 / kMaxSpeedNorm, 0.0f, 1.0f);
    obs[5] = std::clamp(spd_p1 / kMaxSpeedNorm, 0.0f, 1.0f);

    obs[6] = (ctrl.current_phase_index() == 0) ? 0.0f : 1.0f;

    float max_green = 45.0f;
    if (const auto* rlc = ctrl.as<RLController>()) {
        max_green = rlc->config().max_green_s;
        obs[8] = rlc->is_in_transition() ? 1.0f : 0.0f;
        obs[9] = rlc->can_switch() ? 1.0f : 0.0f;
    } else {
        obs[8] = (ctrl.current_stage() != PhaseStage::Green) ? 1.0f : 0.0f;
        obs[9] = (ctrl.current_stage() == PhaseStage::Green && ctrl.stage_elapsed_s() >= 4.0f) ? 1.0f : 0.0f;
    }
    obs[7] = std::clamp(ctrl.stage_elapsed_s() / max_green, 0.0f, 1.0f);

    obs[10] = std::clamp(wait_p0 / kMaxWaitNorm, 0.0f, 1.0f);
    obs[11] = std::clamp(wait_p1 / kMaxWaitNorm, 0.0f, 1.0f);

    return obs;
}

} // namespace syntraq
