#pragma once

//
// syntraq/ai/rl_policy_inference.hpp
//
// Native ONNX Runtime inference engine for trained Reinforcement Learning (DQN)
// traffic signal controller policies.
// Evaluates 12-dimensional normalized traffic observations to predict action
// Q-values and determine optimal phase control decisions with sub-millisecond latency.
//

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace syntraq {

struct OrtPolicySessionImpl;

/// Action decision containing greedy optimal action and evaluated Q-values.
struct RLActionDecision {
    int best_action{ 0 };     ///< 0 = MaintainPhase, 1 = SwitchPhase
    float q_maintain{ 0.0f }; ///< Q(s, action=0)
    float q_switch{ 0.0f };   ///< Q(s, action=1)
    float latency_us{ 0.0f }; ///< Single-sample inference latency (microseconds)
};

class RLPolicyInference {
public:
    static constexpr size_t OBSERVATION_DIM = 12;
    static constexpr size_t ACTION_DIM      = 2;

    RLPolicyInference();
    ~RLPolicyInference();

    // Move-only semantics (PIMPL)
    RLPolicyInference(const RLPolicyInference&) = delete;
    RLPolicyInference& operator=(const RLPolicyInference&) = delete;
    RLPolicyInference(RLPolicyInference&&) noexcept;
    RLPolicyInference& operator=(RLPolicyInference&&) noexcept;

    /// Load an exported ONNX DQN policy model from disk.
    /// Returns true on success, false on error without throwing exceptions.
    bool loadModel(const std::string& model_path);

    /// Predict optimal action decision and Q-values from a 12-dimensional observation.
    [[nodiscard]] RLActionDecision predict(const std::vector<float>& observation);

    /// Batch evaluation of multiple observation vectors.
    [[nodiscard]] std::vector<RLActionDecision> predictBatch(
        const std::vector<std::vector<float>>& batch_observations
    );

    /// Query whether model is loaded and ready for execution.
    [[nodiscard]] bool isLoaded() const noexcept;

    /// Retrieve descriptive error message if load or prediction failed.
    [[nodiscard]] std::string getLastError() const;

    /// Telemetry & performance metrics
    [[nodiscard]] float getLastLatencyUs() const noexcept;
    [[nodiscard]] float getAverageLatencyUs() const noexcept;
    [[nodiscard]] uint64_t getInferenceCount() const noexcept;
    void resetMetrics() noexcept;

private:
    std::unique_ptr<OrtPolicySessionImpl> impl_;
    bool is_loaded_{ false };
    std::string last_error_{};

    // Telemetry
    float last_latency_us_{ 0.0f };
    double total_latency_us_{ 0.0 };
    uint64_t inference_count_{ 0 };
};

} // namespace syntraq
