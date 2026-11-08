#ifndef SYNTRAQ_ENV_C_API_H
#define SYNTRAQ_ENV_C_API_H

//
// syntraq/ai/env_c_api.h
//
// Pure C ABI interface for the SyntraQ Reinforcement Learning Environment.
// Enables zero-overhead in-memory simulation step execution from Python (via ctypes).
//

#include <stdint.h>

#ifdef _WIN32
  #ifdef SYNTRAQ_ENV_EXPORTS
    #define SYNTRAQ_ENV_API __declspec(dllexport)
  #else
    #define SYNTRAQ_ENV_API __declspec(dllimport)
  #endif
#else
  #define SYNTRAQ_ENV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SyntraqEnvConfigC {
    float    step_duration_s;
    float    sim_dt_s;
    float    episode_duration_s;
    uint32_t target_intersection;
    int32_t  use_single_intersection_map; // 1 = true, 0 = false
    int32_t  scenario_id;                // 0: Light, 1: Medium, 2: Heavy, 3: RushHour, 4: Custom
    uint32_t seed;
    float    min_green_s;
    float    max_green_s;
    float    yellow_s;
    float    all_red_s;
    float    queue_penalty_weight;
    float    wait_penalty_weight;
    float    congestion_penalty_weight;
    float    throughput_reward_weight;
    float    invalid_action_penalty;
} SyntraqEnvConfigC;

/// Create a new RL environment instance. Pass NULL to use default configuration.
SYNTRAQ_ENV_API void* syntraq_env_create(const SyntraqEnvConfigC* config);

/// Destroy an environment instance.
SYNTRAQ_ENV_API void  syntraq_env_destroy(void* handle);

/// Reset the environment with a given seed.
/// Fills out_obs (must be pre-allocated array of at least 12 floats).
SYNTRAQ_ENV_API void  syntraq_env_reset(void* handle, uint32_t seed, float* out_obs, int32_t obs_len);

/// Execute one RL action step:
///   action: 0 = Maintain phase, 1 = Switch phase.
/// Fills out_obs (12 floats), out_reward, out_terminated (0 or 1), out_truncated (0 or 1).
/// Returns 0 on success, -1 on error.
SYNTRAQ_ENV_API int32_t syntraq_env_step(void*    handle,
                                         int32_t  action,
                                         float*   out_obs,
                                         int32_t  obs_len,
                                         float*   out_reward,
                                         int32_t* out_terminated,
                                         int32_t* out_truncated);

/// Retrieve telemetry info metrics:
/// out_metrics: array of floats:
///   [0] total_queue
///   [1] avg_wait_s
///   [2] congestion_ratio
///   [3] step_departures
///   [4] elapsed_s
///   [5] cumulative_arrived
SYNTRAQ_ENV_API void syntraq_env_get_info(void* handle, float* out_metrics, int32_t metrics_len);

#ifdef __cplusplus
}
#endif

#endif // SYNTRAQ_ENV_C_API_H
