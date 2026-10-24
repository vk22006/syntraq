//
// src/ai/env_c_api.cpp
//

#include "syntraq/ai/env_c_api.h"
#include "syntraq/ai/traffic_signal_env.hpp"

#include <cstring>
#include <memory>

using namespace syntraq;

namespace {

TrafficScenario map_scenario_id(int32_t id) {
    switch (id) {
        case 0: return TrafficScenario::Low;
        case 1: return TrafficScenario::Medium;
        case 2: return TrafficScenario::High;
        case 3: return TrafficScenario::RushHour;
        default: return TrafficScenario::Custom;
    }
}

EnvConfig map_config(const SyntraqEnvConfigC* c) {
    EnvConfig cfg;
    if (!c) return cfg;

    if (c->step_duration_s > 0.0f)    cfg.step_duration_s = c->step_duration_s;
    if (c->sim_dt_s > 0.0f)           cfg.sim_dt_s = c->sim_dt_s;
    if (c->episode_duration_s > 0.0f) cfg.episode_duration_s = c->episode_duration_s;
    cfg.target_intersection           = static_cast<IntersectionId>(c->target_intersection);
    cfg.use_single_intersection_map   = (c->use_single_intersection_map != 0);
    cfg.scenario                      = map_scenario_id(c->scenario_id);
    cfg.seed                          = c->seed;
    if (c->min_green_s > 0.0f)        cfg.min_green_s = c->min_green_s;
    if (c->max_green_s > 0.0f)        cfg.max_green_s = c->max_green_s;
    if (c->yellow_s >= 0.0f)          cfg.yellow_s = c->yellow_s;
    if (c->all_red_s >= 0.0f)         cfg.all_red_s = c->all_red_s;
    cfg.queue_penalty_weight          = c->queue_penalty_weight;
    cfg.wait_penalty_weight           = c->wait_penalty_weight;
    cfg.congestion_penalty_weight     = c->congestion_penalty_weight;
    cfg.throughput_reward_weight      = c->throughput_reward_weight;
    cfg.invalid_action_penalty        = c->invalid_action_penalty;

    return cfg;
}

} // namespace

extern "C" {

void* syntraq_env_create(const SyntraqEnvConfigC* config) {
    try {
        EnvConfig cfg = map_config(config);
        auto env = std::make_unique<TrafficSignalEnv>(std::move(cfg));
        return env.release();
    } catch (...) {
        return nullptr;
    }
}

void syntraq_env_destroy(void* handle) {
    if (!handle) return;
    delete static_cast<TrafficSignalEnv*>(handle);
}

void syntraq_env_reset(void* handle, uint32_t seed, float* out_obs, int32_t obs_len) {
    if (!handle || !out_obs || obs_len < static_cast<int32_t>(kObsDim)) return;
    auto* env = static_cast<TrafficSignalEnv*>(handle);
    Observation obs = env->reset(seed);
    std::memcpy(out_obs, obs.data(), sizeof(float) * kObsDim);
}

int32_t syntraq_env_step(void*    handle,
                         int32_t  action,
                         float*   out_obs,
                         int32_t  obs_len,
                         float*   out_reward,
                         int32_t* out_terminated,
                         int32_t* out_truncated) {
    if (!handle || !out_obs || obs_len < static_cast<int32_t>(kObsDim)) return -1;
    auto* env = static_cast<TrafficSignalEnv*>(handle);
    try {
        StepResult res = env->step(action);
        std::memcpy(out_obs, res.observation.data(), sizeof(float) * kObsDim);
        if (out_reward)     *out_reward     = res.reward;
        if (out_terminated) *out_terminated = res.terminated ? 1 : 0;
        if (out_truncated)  *out_truncated  = res.truncated ? 1 : 0;
        return 0;
    } catch (...) {
        return -1;
    }
}

void syntraq_env_get_info(void* handle, float* out_metrics, int32_t metrics_len) {
    if (!handle || !out_metrics || metrics_len < 6) return;
    auto* env = static_cast<TrafficSignalEnv*>(handle);
    const auto& res = env->last_step_result();
    out_metrics[0] = static_cast<float>(res.total_queue);
    out_metrics[1] = res.avg_wait_s;
    out_metrics[2] = res.congestion_ratio;
    out_metrics[3] = static_cast<float>(res.step_departures);
    out_metrics[4] = res.elapsed_s;
    out_metrics[5] = static_cast<float>(res.cumulative_arrived);
}

} // extern "C"
