#pragma once

//
// syntraq/signals/rl_controller.hpp
//
// RLController — Native Reinforcement Learning (DQN) traffic signal controller.
// Evaluates 12-dimensional normalized traffic observations via ONNX Runtime to
// execute real-time phase control decisions with sub-millisecond latency.
//
// Four-step controller lifecycle:
//   1. Receive current observation: receive_observation(obs)
//   2. Run inference: run_inference() -> RLActionDecision (Q-values, latency)
//   3. Produce action: produce_action() -> int {0=Maintain, 1=Switch}
//   4. Apply action to traffic signal system: apply_action(action)
//
// Telemetry & measurement:
//   - Inference latency (last and average in microseconds)
//   - Decisions per second (throughput over rolling window)
//   - Decision count
//
// Robustness:
//   - Operates gracefully when no AI model is loaded (safe heuristic fallback)
//   - Strictly enforces min_green_s safety before initiating clearance
//   - Strictly enforces max_green_s cap to prevent starvation deadlock
//

#include "syntraq/ai/rl_policy_inference.hpp"
#include "syntraq/signals/rl_signal_controller.hpp"

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace syntraq {

class RoadNetwork;

// ── RLController ─────────────────────────────────────────────────────────────

class RLController : public RLSignalController {
public:
    static constexpr size_t OBSERVATION_DIM = 12;
    static constexpr size_t ACTION_DIM      = 2;

    RLController();

    RLController(IntersectionId                     intersection_id,
                 std::vector<SignalPhase>           phases,
                 std::vector<RoadId>                all_incoming_roads = {},
                 RLControllerConfig                 config = {},
                 std::shared_ptr<RLPolicyInference> policy = nullptr);

    // ── ITrafficSignalController Overrides ─────────────────────────────────

    void tick(float dt) override;
    void reset() override;

    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "ReinforcementLearning";
    }

    [[nodiscard]] std::unique_ptr<ITrafficSignalController> clone() const override {
        return std::make_unique<RLController>(*this);
    }

    // ── Four-Step RL Controller Lifecycle ──────────────────────────────────

    /// 1. Receive current observation vector
    void receive_observation(const std::vector<float>& observation);
    void receive_observation(const std::array<float, OBSERVATION_DIM>& observation);

    /// 2. Run inference on the received observation via ONNX Runtime
    RLActionDecision run_inference();

    /// 3. Produce action from Q-values or safe fallback
    [[nodiscard]] int produce_action();

    /// 4. Apply action to the traffic signal system (0 = Maintain, 1 = Switch)
    void apply_action(int action);

    /// Unified step: executes steps 1 through 4 sequentially
    RLActionDecision step(const std::vector<float>& observation);

    /// Autonomous mode: when true, Simulation::tick automatically executes decision steps.
    /// Set to false when an external RL environment/agent explicitly steps the controller.
    [[nodiscard]] bool is_autonomous() const noexcept {
        return autonomous_;
    }

    void set_autonomous(bool autonomous) noexcept {
        autonomous_ = autonomous;
    }

    /// True when autonomous and time since last decision >= config.step_interval_s
    [[nodiscard]] bool is_decision_due() const noexcept {
        return autonomous_ && (time_since_decision_s_ >= step_interval_s_);
    }

    // ── AI Model Management ────────────────────────────────────────────────

    /// Load an ONNX policy model from file
    bool load_model(const std::string& model_path);

    /// Query whether an AI model is loaded and available for inference
    [[nodiscard]] bool is_model_loaded() const noexcept;

    /// Bind a preloaded shared policy inference engine
    void set_policy(std::shared_ptr<RLPolicyInference> policy) noexcept;

    [[nodiscard]] std::shared_ptr<RLPolicyInference> policy() const noexcept {
        return policy_;
    }

    [[nodiscard]] const std::string& model_path() const noexcept {
        return model_path_;
    }

    // ── Performance Telemetry & Metrics ────────────────────────────────────

    /// Single-sample latency of the most recent inference run in microseconds
    [[nodiscard]] float last_inference_latency_us() const noexcept {
        return last_decision_.latency_us;
    }

    /// Cumulative average inference latency in microseconds
    [[nodiscard]] float average_inference_latency_us() const noexcept;

    /// Real-world throughput: decisions produced per second
    [[nodiscard]] float decisions_per_second() const noexcept;

    /// Total number of decisions produced since initialization or reset
    [[nodiscard]] uint64_t total_decisions() const noexcept {
        return total_decisions_;
    }

    /// Action produced during the last decision step
    [[nodiscard]] int last_action() const noexcept {
        return last_action_;
    }

    /// Most recent decision snapshot (best_action, Q-values, latency)
    [[nodiscard]] const RLActionDecision& last_decision() const noexcept {
        return last_decision_;
    }

    /// Most recent observation vector ingested
    [[nodiscard]] const std::vector<float>& last_observation() const noexcept {
        return current_observation_;
    }

    void reset_telemetry() noexcept;

    // ── Configuration ──────────────────────────────────────────────────────

    [[nodiscard]] float step_interval_s() const noexcept {
        return step_interval_s_;
    }

    void set_step_interval_s(float interval_s) noexcept {
        step_interval_s_ = (interval_s > 0.0f) ? interval_s : 1.0f;
    }

    // ── Factory Helpers ────────────────────────────────────────────────────

    /// Factory to construct a 2-phase RLController for a junction
    static RLController make_rl(
        const RoadNetwork&                 network,
        IntersectionId                     intersection_id,
        RLControllerConfig                 config = {},
        std::shared_ptr<RLPolicyInference> shared_policy = nullptr);

private:
    [[nodiscard]] int fallback_policy() const noexcept;
    void record_decision_telemetry(float latency_us);

    std::shared_ptr<RLPolicyInference> policy_{ nullptr };
    std::string                        model_path_{ "models/traffic_signal_dqn.onnx" };
    bool                               autonomous_{ true };

    float                              step_interval_s_{ 1.0f };
    float                              time_since_decision_s_{ 0.0f };

    std::vector<float>                 current_observation_{};
    RLActionDecision                   last_decision_{};
    int                                last_action_{ 0 };

    // Telemetry tracking
    uint64_t                           total_decisions_{ 0 };
    double                             total_latency_us_{ 0.0 };
    std::deque<std::chrono::steady_clock::time_point> recent_decision_times_{};
    std::chrono::steady_clock::time_point             first_decision_time_{};
};

} // namespace syntraq

