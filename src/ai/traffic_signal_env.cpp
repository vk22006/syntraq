//
// src/ai/traffic_signal_env.cpp
//

#include "syntraq/ai/traffic_signal_env.hpp"

#include <algorithm>
#include <cmath>

namespace syntraq {

TrafficSignalEnv::TrafficSignalEnv(EnvConfig config)
    : config_(std::move(config)) {
    setup_simulation(config_.seed);
}

void TrafficSignalEnv::setup_simulation(uint32_t seed) {
    Config sim_cfg;
    sim_cfg.dt_seconds                 = config_.sim_dt_s;
    sim_cfg.seed                       = seed;
    sim_cfg.scenario                   = config_.scenario;
    sim_cfg.enable_traffic_signals     = true;
    sim_cfg.signal_controller_type     = SignalControllerType::ReinforcementLearning;
    sim_cfg.default_green_duration_s   = config_.min_green_s;
    sim_cfg.default_yellow_duration_s  = config_.yellow_s;
    sim_cfg.default_all_red_duration_s = config_.all_red_s;

    if (config_.use_single_intersection_map) {
        sim_ = std::make_unique<Simulation>(sim_cfg, [] {
            return make_single_intersection_map();
        });
        config_.target_intersection = IntersectionId{ 0 };
        // Focus vehicles on approaching and crossing the central junction
        sim_->set_allowed_spawn_origins({
            IntersectionId{ 1 }, IntersectionId{ 2 },
            IntersectionId{ 3 }, IntersectionId{ 4 }
        });
        sim_->set_allowed_spawn_destinations({
            IntersectionId{ 1 }, IntersectionId{ 2 },
            IntersectionId{ 3 }, IntersectionId{ 4 }
        });
    } else {
        sim_ = std::make_unique<Simulation>(sim_cfg);
    }

    // Ensure target intersection has an RLSignalController
    auto* existing_ctrl = sim_->signal_controller(config_.target_intersection);
    if (!existing_ctrl || !existing_ctrl->is<RLSignalController>()) {
        RLControllerConfig rl_cfg;
        rl_cfg.min_green_s = config_.min_green_s;
        rl_cfg.max_green_s = config_.max_green_s;
        rl_cfg.yellow_s    = config_.yellow_s;
        rl_cfg.all_red_s   = config_.all_red_s;

        auto rl_c = RLSignalController::make_rl(sim_->network(), config_.target_intersection, rl_cfg);
        sim_->set_signal_controller(TrafficSignalController(std::move(rl_c)));
        existing_ctrl = sim_->signal_controller(config_.target_intersection);
    }

    rl_controller_ = existing_ctrl ? existing_ctrl->as<RLSignalController>() : nullptr;

    // Cache phase road IDs
    phase_0_roads_.clear();
    phase_1_roads_.clear();
    if (rl_controller_) {
        const auto& phases = rl_controller_->phases();
        if (phases.size() >= 1) {
            phase_0_roads_ = phases[0].green_roads;
        }
        if (phases.size() >= 2) {
            phase_1_roads_ = phases[1].green_roads;
        }
    }

    prev_total_arrived_ = sim_->state().total_arrived;
    last_result_        = StepResult{};
    last_result_.observation = get_observation();
    last_result_.elapsed_s   = sim_->state().elapsed_s;
}

Observation TrafficSignalEnv::reset(std::optional<uint32_t> seed) {
    if (seed.has_value()) {
        config_.seed = *seed;
    }
    setup_simulation(config_.seed);
    return last_result_.observation;
}

StepResult TrafficSignalEnv::step(int action) {
    bool invalid_action = false;

    if (rl_controller_) {
        if (action == 0) {
            // Action 0: Maintain current phase
            rl_controller_->maintain_phase();
        } else if (action == 1) {
            // Action 1: Switch phase
            const bool switched = rl_controller_->switch_phase();
            if (!switched) {
                invalid_action = true;
            }
        }
    }

    // Advance simulation by step_duration_s
    const int ticks = std::max(1, static_cast<int>(std::round(config_.step_duration_s / config_.sim_dt_s)));
    for (int i = 0; i < ticks; ++i) {
        sim_->tick();
    }

    // Extract approach traffic conditions
    float q_p0 = 0.0f, q_p1 = 0.0f;
    float app_p0 = 0.0f, app_p1 = 0.0f;
    float spd_p0 = 0.0f, spd_p1 = 0.0f;
    float wait_p0 = 0.0f, wait_p1 = 0.0f;
    extract_approach_metrics(q_p0, q_p1, app_p0, app_p1, spd_p0, spd_p1, wait_p0, wait_p1);

    const float total_q   = q_p0 + q_p1;
    const float total_app = app_p0 + app_p1;
    const float avg_wait  = (total_q > 0.0f) ? ((q_p0 * wait_p0 + q_p1 * wait_p1) / total_q) : 0.0f;
    const float cong_ratio= total_q / std::max(1.0f, total_q + total_app);

    const uint32_t current_arrived = sim_->state().total_arrived;
    const uint32_t departures      = (current_arrived >= prev_total_arrived_)
                                   ? (current_arrived - prev_total_arrived_)
                                   : 0;
    prev_total_arrived_ = current_arrived;

    // Compute composite reward:
    // Penalty on queue length, waiting time, and congestion; bonus on completed departures
    float penalty = (config_.queue_penalty_weight * total_q)
                  + (config_.wait_penalty_weight  * avg_wait)
                  + (config_.congestion_penalty_weight * cong_ratio);

    if (invalid_action) {
        penalty += config_.invalid_action_penalty;
    }

    const float reward = -penalty + (config_.throughput_reward_weight * static_cast<float>(departures));

    const bool truncated = (sim_->state().elapsed_s >= config_.episode_duration_s);
    const bool terminated = false;

    StepResult res;
    res.observation        = get_observation();
    res.reward             = reward;
    res.terminated         = terminated;
    res.truncated          = truncated;
    res.total_queue        = static_cast<uint32_t>(total_q);
    res.avg_wait_s         = avg_wait;
    res.congestion_ratio   = cong_ratio;
    res.step_departures    = departures;
    res.elapsed_s          = sim_->state().elapsed_s;
    res.cumulative_arrived = current_arrived;

    last_result_ = res;
    return res;
}

Observation TrafficSignalEnv::get_observation() const {
    Observation obs{};
    obs.fill(0.0f);

    float q_p0 = 0.0f, q_p1 = 0.0f;
    float app_p0 = 0.0f, app_p1 = 0.0f;
    float spd_p0 = 0.0f, spd_p1 = 0.0f;
    float wait_p0 = 0.0f, wait_p1 = 0.0f;
    extract_approach_metrics(q_p0, q_p1, app_p0, app_p1, spd_p0, spd_p1, wait_p0, wait_p1);

    // Normalization constants
    constexpr float kMaxQueueNorm   = 20.0f;
    constexpr float kMaxSpeedNorm   = 13.89f; // 50 km/h
    constexpr float kMaxWaitNorm    = 60.0f;  // 60 seconds

    // [0] Queue on Phase 0 approaches [0..1]
    obs[0] = std::clamp(q_p0 / kMaxQueueNorm, 0.0f, 1.0f);

    // [1] Queue on Phase 1 approaches [0..1]
    obs[1] = std::clamp(q_p1 / kMaxQueueNorm, 0.0f, 1.0f);

    // [2] Moving approaching vehicles on Phase 0 [0..1]
    obs[2] = std::clamp(app_p0 / kMaxQueueNorm, 0.0f, 1.0f);

    // [3] Moving approaching vehicles on Phase 1 [0..1]
    obs[3] = std::clamp(app_p1 / kMaxQueueNorm, 0.0f, 1.0f);

    // [4] Average speed on Phase 0 [0..1]
    obs[4] = std::clamp(spd_p0 / kMaxSpeedNorm, 0.0f, 1.0f);

    // [5] Average speed on Phase 1 [0..1]
    obs[5] = std::clamp(spd_p1 / kMaxSpeedNorm, 0.0f, 1.0f);

    if (rl_controller_) {
        // [6] Current phase (0.0: Phase 0, 1.0: Phase 1)
        obs[6] = (rl_controller_->current_phase_index() == 0) ? 0.0f : 1.0f;

        // [7] Time elapsed in current stage normalized by max_green_s [0..1]
        obs[7] = std::clamp(rl_controller_->stage_elapsed_s() / config_.max_green_s, 0.0f, 1.0f);

        // [8] Is clearing (Yellow or AllRed)
        obs[8] = rl_controller_->is_in_transition() ? 1.0f : 0.0f;

        // [9] Can switch (in Green and elapsed >= min_green_s)
        obs[9] = rl_controller_->can_switch() ? 1.0f : 0.0f;
    }

    // [10] Average wait time on Phase 0 [0..1]
    obs[10] = std::clamp(wait_p0 / kMaxWaitNorm, 0.0f, 1.0f);

    // [11] Average wait time on Phase 1 [0..1]
    obs[11] = std::clamp(wait_p1 / kMaxWaitNorm, 0.0f, 1.0f);

    return obs;
}

void TrafficSignalEnv::extract_approach_metrics(float& q_p0, float& q_p1,
                                                float& app_p0, float& app_p1,
                                                float& spd_p0, float& spd_p1,
                                                float& wait_p0, float& wait_p1) const {
    q_p0 = 0.0f; q_p1 = 0.0f;
    app_p0 = 0.0f; app_p1 = 0.0f;
    spd_p0 = 13.89f; spd_p1 = 13.89f;
    wait_p0 = 0.0f; wait_p1 = 0.0f;

    if (!sim_) return;

    float p0_spd_sum = 0.0f; uint32_t p0_veh_cnt = 0; float p0_wait_sum = 0.0f;
    float p1_spd_sum = 0.0f; uint32_t p1_veh_cnt = 0; float p1_wait_sum = 0.0f;

    for (const auto& v : sim_->vehicles()) {
        if (v.state == VehicleState::Arrived) continue;

        // Check Phase 0 approaches
        const bool in_p0 = std::find(phase_0_roads_.begin(), phase_0_roads_.end(), v.current_road) != phase_0_roads_.end();
        if (in_p0) {
            p0_spd_sum += v.speed_mps;
            p0_veh_cnt++;
            if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
                q_p0 += 1.0f;
                p0_wait_sum += v.wait_time_s;
            } else {
                app_p0 += 1.0f;
            }
            continue;
        }

        // Check Phase 1 approaches
        const bool in_p1 = std::find(phase_1_roads_.begin(), phase_1_roads_.end(), v.current_road) != phase_1_roads_.end();
        if (in_p1) {
            p1_spd_sum += v.speed_mps;
            p1_veh_cnt++;
            if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
                q_p1 += 1.0f;
                p1_wait_sum += v.wait_time_s;
            } else {
                app_p1 += 1.0f;
            }
        }
    }

    if (p0_veh_cnt > 0) {
        spd_p0 = p0_spd_sum / static_cast<float>(p0_veh_cnt);
    }
    if (q_p0 > 0.0f) {
        wait_p0 = p0_wait_sum / q_p0;
    }

    if (p1_veh_cnt > 0) {
        spd_p1 = p1_spd_sum / static_cast<float>(p1_veh_cnt);
    }
    if (q_p1 > 0.0f) {
        wait_p1 = p1_wait_sum / q_p1;
    }
}

} // namespace syntraq
