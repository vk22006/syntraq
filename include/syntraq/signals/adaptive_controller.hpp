#pragma once

//
// syntraq/signals/adaptive_controller.hpp
//
// AdaptiveController — Non-AI adaptive traffic signal controller.
// Uses real-time approach queue lengths and waiting time conditions to dynamically:
//   - Extend current green (when current phase has queued demand and opposing wait is low)
//   - Shorten phase / gap-out (when current queue dissipates and opposing vehicles are waiting)
//   - Switch phase (when max green is reached or opposing wait exceeds starvation threshold)
//
// The decision policy is strictly deterministic and provides human-explainable
// reasoning for every timing adjustment.
//

#include "syntraq/signals/signal_controller_interface.hpp"

namespace syntraq {

class RoadNetwork;

// ── AdaptiveDecision ─────────────────────────────────────────────────────────

enum class AdaptiveDecision : uint8_t {
    Hold,         ///< Serving mandatory minimum green or clearance phase
    ExtendGreen,  ///< Extending green duration beyond nominal time due to active queue
    ShortenPhase, ///< Terminating green early due to dissipated queue while opposing traffic waits
    SwitchPhase   ///< Transitioning to yellow clearance (max green cap or starvation limit reached)
};

[[nodiscard]] constexpr std::string_view to_string(AdaptiveDecision decision) noexcept {
    switch (decision) {
        case AdaptiveDecision::Hold:         return "Hold";
        case AdaptiveDecision::ExtendGreen:  return "ExtendGreen";
        case AdaptiveDecision::ShortenPhase: return "ShortenPhase";
        case AdaptiveDecision::SwitchPhase:  return "SwitchPhase";
    }
    return "Unknown";
}

// ── AdaptiveControllerConfig ─────────────────────────────────────────────────

struct AdaptiveControllerConfig {
    float min_green_s          {  4.0f }; ///< Minimum green time before green can be shortened (safety)
    float max_green_s          { 30.0f }; ///< Maximum green duration cap
    float nominal_green_s      { 10.0f }; ///< Baseline green duration
    float max_wait_threshold_s { 20.0f }; ///< Opposing waiting time that forces phase switch (starvation prevention)
    float queue_priority_ratio {  1.0f }; ///< Ratio threshold for prioritizing current vs opposing queues
};

// ── AdaptiveController ───────────────────────────────────────────────────────

class AdaptiveController : public ITrafficSignalController {
public:
    AdaptiveController() = default;

    /// Construct an adaptive controller for an intersection
    AdaptiveController(IntersectionId           intersection_id,
                       std::vector<SignalPhase> phases,
                       std::vector<RoadId>      all_incoming_roads = {},
                       AdaptiveControllerConfig config = {});

    // ── ITrafficSignalController Overrides ─────────────────────────────────

    void tick(float dt) override;
    void reset() override;

    /// Ingest current queue counts and vehicle wait times for approach roads
    void observe_traffic(const std::unordered_map<RoadId, uint32_t>& queues,
                         const std::unordered_map<RoadId, float>&    wait_times = {}) override;

    [[nodiscard]] IntersectionId intersection_id() const noexcept override {
        return intersection_id_;
    }

    [[nodiscard]] SignalControllerType type() const noexcept override {
        return SignalControllerType::Adaptive;
    }

    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "Adaptive";
    }

    [[nodiscard]] SignalColor signal_for_road(RoadId road_id) const noexcept override;
    [[nodiscard]] bool can_proceed(RoadId road_id) const noexcept override;
    [[nodiscard]] const TrafficLight* light(RoadId road_id) const noexcept override;

    [[nodiscard]] const std::unordered_map<RoadId, TrafficLight>& lights() const noexcept override {
        return lights_;
    }

    [[nodiscard]] size_t current_phase_index() const noexcept override {
        return current_phase_idx_;
    }

    [[nodiscard]] const SignalPhase* current_phase() const noexcept override;

    [[nodiscard]] PhaseStage current_stage() const noexcept override {
        return current_stage_;
    }

    [[nodiscard]] float stage_elapsed_s() const noexcept override {
        return stage_elapsed_s_;
    }

    [[nodiscard]] float current_stage_duration() const noexcept override;
    [[nodiscard]] float stage_remaining_s() const noexcept override;
    [[nodiscard]] float cycle_duration_s() const noexcept override;

    [[nodiscard]] size_t phase_count() const noexcept override {
        return phases_.size();
    }

    [[nodiscard]] const std::vector<SignalPhase>& phases() const noexcept override {
        return phases_;
    }

    void set_phases(std::vector<SignalPhase> phases,
                    std::vector<RoadId>      all_incoming_roads = {}) override;

    void set_phase_timings(size_t phase_idx,
                           float  green_s,
                           float  yellow_s,
                           float  all_red_s) override;

    [[nodiscard]] bool is_valid() const noexcept override;
    bool validate(std::string* error_msg = nullptr) const override;

    [[nodiscard]] std::unique_ptr<ITrafficSignalController> clone() const override {
        return std::make_unique<AdaptiveController>(*this);
    }

    // ── Adaptive Telemetry & Explainability ─────────────────────────────────

    [[nodiscard]] AdaptiveDecision last_decision() const noexcept {
        return last_decision_;
    }

    [[nodiscard]] std::string_view last_decision_reason() const noexcept {
        return last_decision_reason_;
    }

    [[nodiscard]] uint32_t extension_count() const noexcept {
        return extensions_count_;
    }

    [[nodiscard]] uint32_t shortening_count() const noexcept {
        return shortenings_count_;
    }

    [[nodiscard]] uint32_t switch_count() const noexcept {
        return switches_count_;
    }

    [[nodiscard]] const AdaptiveControllerConfig& config() const noexcept {
        return config_;
    }

    void set_config(AdaptiveControllerConfig cfg) {
        config_ = cfg;
    }

    // ── Factory Helpers ────────────────────────────────────────────────────

    /// Build an adaptive 2-phase controller for an intersection
    static AdaptiveController make_adaptive(
        const RoadNetwork&       network,
        IntersectionId           intersection_id,
        AdaptiveControllerConfig config = {},
        float                    yellow_s  = 3.0f,
        float                    all_red_s = 2.0f);

private:
    void update_lights();
    void evaluate_green_decision();
    void transition_to_yellow();

    IntersectionId           intersection_id_   { kInvalidIntersectionId };
    std::vector<SignalPhase> phases_;
    std::vector<RoadId>      all_incoming_roads_;
    AdaptiveControllerConfig config_;

    size_t     current_phase_idx_{ 0 };
    PhaseStage current_stage_    { PhaseStage::Green };
    float      stage_elapsed_s_  { 0.0f };

    // Dynamic queue observations
    std::unordered_map<RoadId, uint32_t> observed_queues_;
    std::unordered_map<RoadId, float>    observed_wait_times_;

    // Telemetry & Explainability
    AdaptiveDecision last_decision_       { AdaptiveDecision::Hold };
    std::string      last_decision_reason_{ "Initial state" };
    uint32_t         extensions_count_    { 0 };
    uint32_t         shortenings_count_   { 0 };
    uint32_t         switches_count_      { 0 };

    std::unordered_map<RoadId, TrafficLight> lights_;
};

} // namespace syntraq
