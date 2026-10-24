#pragma once

//
// syntraq/signals/rl_signal_controller.hpp
//
// RLSignalController — Reinforcement learning and actuated traffic signal controller.
// Implements the ITrafficSignalController interface and executes discrete RL actions:
//   - Action 0: Maintain current phase (continue current green or ongoing clearance)
//   - Action 1: Switch phase (initiate transition via safe yellow/all-red clearance)
//
// Strictly enforces safety constraints:
//   - Minimum green time (min_green_s) before a phase switch can occur
//   - Maximum green duration cap (max_green_s) to prevent starvation deadlock
//   - Complete yellow and all-red clearance stages before activating conflicting green
//

#include "syntraq/signals/signal_controller_interface.hpp"

namespace syntraq {

class RoadNetwork;

// ── RLControllerConfig ───────────────────────────────────────────────────────

struct RLControllerConfig {
    float min_green_s {  4.0f }; ///< Minimum green time before a switch can be initiated (safety)
    float max_green_s { 45.0f }; ///< Maximum green cap before failsafe switch
    float yellow_s    {  3.0f }; ///< Yellow clearance interval
    float all_red_s   {  2.0f }; ///< All-red clearance interval
};

// ── RLSignalController ───────────────────────────────────────────────────────

class RLSignalController : public ITrafficSignalController {
public:
    RLSignalController() = default;

    /// Construct an RL controller for an intersection
    RLSignalController(IntersectionId           intersection_id,
                       std::vector<SignalPhase> phases,
                       std::vector<RoadId>      all_incoming_roads = {},
                       RLControllerConfig       config = {});

    // ── ITrafficSignalController Overrides ─────────────────────────────────

    void tick(float dt) override;
    void reset() override;

    /// Ingest approach traffic observations (optional telemetry for RL)
    void observe_traffic(const std::unordered_map<RoadId, uint32_t>& queues,
                         const std::unordered_map<RoadId, float>&    wait_times = {}) override;

    [[nodiscard]] IntersectionId intersection_id() const noexcept override {
        return intersection_id_;
    }

    [[nodiscard]] SignalControllerType type() const noexcept override {
        return SignalControllerType::ReinforcementLearning;
    }

    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "ReinforcementLearning";
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

    [[nodiscard]] bool is_valid() const noexcept override;
    bool validate(std::string* error_msg = nullptr) const override;

    [[nodiscard]] std::unique_ptr<ITrafficSignalController> clone() const override {
        return std::make_unique<RLSignalController>(*this);
    }

    void set_phases(std::vector<SignalPhase> phases,
                    std::vector<RoadId>      all_incoming_roads = {}) override;

    void set_phase_timings(size_t phase_idx,
                           float  green_s,
                           float  yellow_s,
                           float  all_red_s) override;

    // ── RL Action Interface ────────────────────────────────────────────────

    /// Action 0: Maintain current phase (continues green or ongoing clearance)
    void maintain_phase() noexcept;

    /// Action 1: Switch phase. If in Green and stage_elapsed >= min_green, initiates Yellow clearance.
    /// Returns true if switch was initiated, false if rejected (min_green not met or already clearing).
    bool switch_phase() noexcept;

    /// True if the controller is in Green and has satisfied min_green_s
    [[nodiscard]] bool can_switch() const noexcept {
        constexpr float kEps = 1e-4f;
        return current_stage_ == PhaseStage::Green && (stage_elapsed_s_ + kEps) >= config_.min_green_s;
    }

    /// True if currently in Yellow or AllRed clearance transition
    [[nodiscard]] bool is_in_transition() const noexcept {
        return current_stage_ != PhaseStage::Green;
    }

    [[nodiscard]] const RLControllerConfig& config() const noexcept { return config_; }
    void set_config(RLControllerConfig cfg) noexcept { config_ = cfg; }

    // ── Factory Helpers ────────────────────────────────────────────────────

    /// Factory to construct a 2-phase (NS vs EW) RL signal controller for a junction
    static RLSignalController make_rl(const RoadNetwork& network,
                                      IntersectionId     intersection_id,
                                      RLControllerConfig config = {});

private:
    void update_lights();

    IntersectionId                              intersection_id_{ kInvalidIntersectionId };
    std::vector<SignalPhase>                    phases_{};
    std::vector<RoadId>                         all_incoming_roads_{};
    std::unordered_map<RoadId, TrafficLight>    lights_{};
    size_t                                      current_phase_idx_{ 0 };
    PhaseStage                                  current_stage_{ PhaseStage::Green };
    float                                       stage_elapsed_s_{ 0.0f };
    RLControllerConfig                          config_{};
    bool                                        switch_requested_{ false };
    std::unordered_map<RoadId, uint32_t>        last_observed_queues_{};
    std::unordered_map<RoadId, float>           last_observed_waits_{};
};

} // namespace syntraq
