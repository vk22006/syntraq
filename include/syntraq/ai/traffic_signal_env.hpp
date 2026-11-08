#pragma once

//
// syntraq/ai/traffic_signal_env.hpp
//
// Single-intersection reinforcement learning environment for traffic signal control.
// Formulates the traffic signal decision problem as a Markov Decision Process (MDP):
//
//   State (Observation): Compact 12-dimensional normalized vector [0.0 .. 1.0]:
//     [0] queue_phase_0       Normalized queue on Phase 0 (North-South) approaches
//     [1] queue_phase_1       Normalized queue on Phase 1 (East-West) approaches
//     [2] approaching_phase_0 Normalized moving approaching vehicles on Phase 0
//     [3] approaching_phase_1 Normalized moving approaching vehicles on Phase 1
//     [4] avg_speed_phase_0   Normalized average vehicle speed on Phase 0 [0..1]
//     [5] avg_speed_phase_1   Normalized average vehicle speed on Phase 1 [0..1]
//     [6] current_phase       Active phase index (0.0: Phase 0, 1.0: Phase 1)
//     [7] time_in_phase       Elapsed time in current stage / max_green_s [0..1]
//     [8] is_clearing         1.0 if Yellow or AllRed clearance transition, 0.0 otherwise
//     [9] can_switch          1.0 if Green and elapsed >= min_green_s, 0.0 otherwise
//    [10] avg_wait_phase_0    Normalized average vehicle waiting time on Phase 0
//    [11] avg_wait_phase_1    Normalized average vehicle waiting time on Phase 1
//
//   Actions: Discrete(2)
//     0: Maintain current phase (continue current green or ongoing clearance)
//     1: Switch phase (initiate transition via safe yellow/all-red clearance)
//
//   Reward Function:
//     Penalizes queue lengths, waiting time, and congestion ratio.
//     Rewards completed vehicle throughput departures.
//     R_t = - (w_queue * Q + w_wait * W + w_cong * C) + w_thr * delta_Departures
//

#include "syntraq/core/config.hpp"
#include "syntraq/signals/rl_signal_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace syntraq {

// ── Environment Configuration ────────────────────────────────────────────────

struct EnvConfig {
    float           step_duration_s            { 1.0f };   ///< Duration of one RL action step in seconds
    float           sim_dt_s                   { 0.1f };   ///< Internal simulation physics timestep (seconds)
    float           episode_duration_s         { 300.0f }; ///< Total episode duration in seconds
    IntersectionId  target_intersection        { IntersectionId{ 0 } }; ///< Target intersection to control
    bool            use_single_intersection_map{ true };   ///< Use dedicated 4-way intersection map
    TrafficScenario scenario                   { TrafficScenario::Medium };
    uint32_t        seed                       { 42 };

    // Safety timings
    float           min_green_s                { 4.0f };
    float           max_green_s                { 45.0f };
    float           yellow_s                   { 3.0f };
    float           all_red_s                  { 2.0f };

    // Reward / penalty weights
    float           queue_penalty_weight       { 0.2f };
    float           wait_penalty_weight        { 0.05f };
    float           congestion_penalty_weight  { 1.0f };
    float           throughput_reward_weight   { 1.0f };
    float           invalid_action_penalty     { 0.05f };
};

// ── Observation Vector ───────────────────────────────────────────────────────

constexpr size_t kObsDim = 12;
using Observation = std::array<float, kObsDim>;

// ── Step Result ──────────────────────────────────────────────────────────────

struct StepResult {
    Observation observation{};
    float       reward{ 0.0f };
    bool        terminated{ false };
    bool        truncated{ false };

    // Telemetry & metrics
    uint32_t    total_queue{ 0 };
    float       avg_wait_s{ 0.0f };
    float       congestion_ratio{ 0.0f };
    uint32_t    step_departures{ 0 };
    float       elapsed_s{ 0.0f };
    uint32_t    cumulative_arrived{ 0 };
};

// ── TrafficSignalEnv ─────────────────────────────────────────────────────────

class TrafficSignalEnv {
public:
    explicit TrafficSignalEnv(EnvConfig config = {});

    /// Reset environment to initial state with optional custom seed.
    /// Returns initial state observation.
    Observation reset(std::optional<uint32_t> seed = std::nullopt);

    /// Execute one discrete RL action:
    ///   0 = Maintain current phase
    ///   1 = Switch phase (if min_green is met)
    /// Advances simulation by step_duration_s and returns StepResult.
    StepResult step(int action);

    [[nodiscard]] const EnvConfig& config() const noexcept { return config_; }
    [[nodiscard]] const Simulation& simulation() const noexcept { return *sim_; }
    [[nodiscard]]       Simulation& simulation()       noexcept { return *sim_; }

    [[nodiscard]] Observation get_observation() const;
    [[nodiscard]] const StepResult& last_step_result() const noexcept { return last_result_; }

    [[nodiscard]] const std::vector<RoadId>& phase_0_roads() const noexcept { return phase_0_roads_; }
    [[nodiscard]] const std::vector<RoadId>& phase_1_roads() const noexcept { return phase_1_roads_; }

private:
    void setup_simulation(uint32_t seed);
    void extract_approach_metrics(float& q_p0, float& q_p1,
                                  float& app_p0, float& app_p1,
                                  float& spd_p0, float& spd_p1,
                                  float& wait_p0, float& wait_p1) const;

    EnvConfig                      config_;
    std::unique_ptr<Simulation>    sim_;
    RLSignalController*            rl_controller_{ nullptr };
    uint32_t                       prev_total_arrived_{ 0 };
    StepResult                     last_result_{};
    std::vector<RoadId>            phase_0_roads_{};
    std::vector<RoadId>            phase_1_roads_{};
};

} // namespace syntraq
