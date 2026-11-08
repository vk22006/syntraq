//
// src/signals/rl_controller.cpp
//
// Implementation of RLController:
// Native ONNX Runtime-powered traffic signal controller.
//

#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace syntraq {

namespace {

std::string resolve_model_path(const std::string& path) {
    if (path.empty()) return path;
    if (std::filesystem::exists(path)) return path;

    const std::string alt1 = "../" + path;
    if (std::filesystem::exists(alt1)) return alt1;

    const std::string alt2 = "../../" + path;
    if (std::filesystem::exists(alt2)) return alt2;

    const std::string bin_path = "bin/" + path;
    if (std::filesystem::exists(bin_path)) return bin_path;

    return path;
}

} // namespace

RLController::RLController()
    : RLSignalController()
    , current_observation_(OBSERVATION_DIM, 0.0f) {
}

RLController::RLController(
    IntersectionId                     intersection_id,
    std::vector<SignalPhase>           phases,
    std::vector<RoadId>                all_incoming_roads,
    RLControllerConfig                 config,
    std::shared_ptr<RLPolicyInference> policy)
    : RLSignalController(intersection_id, std::move(phases), std::move(all_incoming_roads), config)
    , policy_(std::move(policy))
    , model_path_(config.model_path)
    , step_interval_s_(config.step_interval_s)
    , current_observation_(OBSERVATION_DIM, 0.0f) {

    if (!policy_ && config.load_model_on_init) {
        // Attempt loading default policy model
        policy_ = std::make_shared<RLPolicyInference>();
        const std::string resolved = resolve_model_path(model_path_);
        if (std::filesystem::exists(resolved)) {
            if (policy_->loadModel(resolved)) {
                model_path_ = resolved;
            }
        }
    }
}

// ── ITrafficSignalController Overrides ───────────────────────────────────────

void RLController::tick(float dt) {
    if (dt <= 0.0f) return;

    time_since_decision_s_ += dt;

    // Advance underlying signal stage timer and clearance logic
    RLSignalController::tick(dt);
}

void RLController::reset() {
    RLSignalController::reset();
    time_since_decision_s_ = 0.0f;
    std::fill(current_observation_.begin(), current_observation_.end(), 0.0f);
    last_decision_ = RLActionDecision{};
    last_action_   = 0;
    reset_telemetry();
}

// ── Four-Step RL Controller Lifecycle ────────────────────────────────────────

void RLController::receive_observation(const std::vector<float>& observation) {
    if (observation.size() == OBSERVATION_DIM) {
        current_observation_ = observation;
    } else {
        // Dimension mismatch or empty: resize or pad gracefully
        current_observation_.assign(OBSERVATION_DIM, 0.0f);
        const size_t copy_cnt = std::min(observation.size(), OBSERVATION_DIM);
        for (size_t i = 0; i < copy_cnt; ++i) {
            current_observation_[i] = observation[i];
        }
    }
}

void RLController::receive_observation(const std::array<float, OBSERVATION_DIM>& observation) {
    current_observation_.assign(observation.begin(), observation.end());
}

RLActionDecision RLController::run_inference() {
    if (current_observation_.size() != OBSERVATION_DIM) {
        current_observation_.assign(OBSERVATION_DIM, 0.0f);
    }

    if (!is_model_loaded()) {
        // Safe heuristic fallback: no model loaded
        last_decision_ = RLActionDecision{
            fallback_policy(),
            0.0f,
            0.0f,
            0.0f
        };
        record_decision_telemetry(0.0f);
        return last_decision_;
    }

    // Execute native ONNX Runtime inference
    last_decision_ = policy_->predict(current_observation_);
    record_decision_telemetry(last_decision_.latency_us);
    return last_decision_;
}

int RLController::produce_action() {
    if (!is_model_loaded()) {
        last_action_ = fallback_policy();
    } else {
        last_action_ = (last_decision_.best_action == 1) ? 1 : 0;
    }
    return last_action_;
}

void RLController::apply_action(int action) {
    last_action_ = action;
    time_since_decision_s_ = 0.0f;

    if (action == 1) {
        // Action 1: Switch phase
        switch_phase();
    } else {
        // Action 0: Maintain phase
        maintain_phase();
    }
}

RLActionDecision RLController::step(const std::vector<float>& observation) {
    receive_observation(observation);
    auto decision = run_inference();
    int action = produce_action();
    apply_action(action);
    return decision;
}

// ── AI Model Management ──────────────────────────────────────────────────────

bool RLController::load_model(const std::string& model_path) {
    if (!policy_) {
        policy_ = std::make_shared<RLPolicyInference>();
    }
    const std::string resolved = resolve_model_path(model_path);
    const bool success = policy_->loadModel(resolved);
    if (success) {
        model_path_ = resolved;
    }
    return success;
}

bool RLController::is_model_loaded() const noexcept {
    return policy_ && policy_->isLoaded();
}

void RLController::set_policy(std::shared_ptr<RLPolicyInference> policy) noexcept {
    policy_ = std::move(policy);
}

// ── Performance Telemetry & Metrics ──────────────────────────────────────────

float RLController::average_inference_latency_us() const noexcept {
    if (total_decisions_ == 0) return 0.0f;
    return static_cast<float>(total_latency_us_ / static_cast<double>(total_decisions_));
}

float RLController::decisions_per_second() const noexcept {
    if (recent_decision_times_.size() < 2) {
        if (total_decisions_ == 0) return 0.0f;
        const auto now = std::chrono::steady_clock::now();
        const float elapsed_s = std::chrono::duration<float>(now - first_decision_time_).count();
        return (elapsed_s > 0.05f) ? (static_cast<float>(total_decisions_) / elapsed_s) : 0.0f;
    }

    const float window_duration_s = std::chrono::duration<float>(
        recent_decision_times_.back() - recent_decision_times_.front()
    ).count();

    if (window_duration_s > 0.001f) {
        return static_cast<float>(recent_decision_times_.size() - 1) / window_duration_s;
    }
    return 0.0f;
}

void RLController::reset_telemetry() noexcept {
    total_decisions_  = 0;
    total_latency_us_ = 0.0;
    recent_decision_times_.clear();
    first_decision_time_ = std::chrono::steady_clock::time_point{};
}

void RLController::record_decision_telemetry(float latency_us) {
    total_decisions_++;
    total_latency_us_ += static_cast<double>(latency_us);

    const auto now = std::chrono::steady_clock::now();
    if (first_decision_time_ == std::chrono::steady_clock::time_point{}) {
        first_decision_time_ = now;
    }

    recent_decision_times_.push_back(now);

    // Maintain a rolling window of recent decisions (2.0 seconds)
    const auto cutoff = now - std::chrono::milliseconds(2000);
    while (recent_decision_times_.size() > 1 && recent_decision_times_.front() < cutoff) {
        recent_decision_times_.pop_front();
    }
}

int RLController::fallback_policy() const noexcept {
    // When no AI model is loaded, actuate based on green duration and clearance safety
    if (!can_switch()) {
        return 0; // Maintain phase
    }

    // If green stage has elapsed at least 15.0 seconds and has satisfied min_green_s,
    // trigger a switch to service conflicting demand
    constexpr float kNominalFallbackGreenS = 15.0f;
    if (stage_elapsed_s() >= kNominalFallbackGreenS) {
        return 1; // Switch phase
    }

    return 0; // Maintain phase
}

// ── Factory Helpers ──────────────────────────────────────────────────────────

RLController RLController::make_rl(
    const RoadNetwork&                 network,
    IntersectionId                     intersection_id,
    RLControllerConfig                 config,
    std::shared_ptr<RLPolicyInference> shared_policy) {

    auto rl_sig = RLSignalController::make_rl(network, intersection_id, config);

    RLController ctrl(
        intersection_id,
        rl_sig.phases(),
        {},
        config,
        std::move(shared_policy)
    );

    return ctrl;
}

} // namespace syntraq
