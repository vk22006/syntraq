#pragma once

//
// syntraq/signals/fixed_time_controller.hpp
//
// FixedTimeController — Baseline fixed-time intersection signal controller.
// Implements deterministic cyclical finite state machine with configurable
// green, yellow, and all-red clearance timings per phase.
//

#include "syntraq/signals/signal_controller_interface.hpp"

namespace syntraq {

class RoadNetwork;

class FixedTimeController : public ITrafficSignalController {
public:
    FixedTimeController() = default;

    /// Construct a controller for an intersection with a sequence of signal phases
    /// and the full set of incoming road IDs to control.
    FixedTimeController(IntersectionId           intersection_id,
                        std::vector<SignalPhase> phases,
                        std::vector<RoadId>      all_incoming_roads = {});

    // ── ITrafficSignalController Overrides ─────────────────────────────────

    void tick(float dt) override;
    void reset() override;

    /// Fixed-time controller ignores traffic observations and follows fixed schedule
    void observe_traffic(const std::unordered_map<RoadId, uint32_t>& queues,
                         const std::unordered_map<RoadId, float>&    wait_times = {}) override {
        (void)queues;
        (void)wait_times;
    }

    [[nodiscard]] IntersectionId intersection_id() const noexcept override {
        return intersection_id_;
    }

    [[nodiscard]] SignalControllerType type() const noexcept override {
        return SignalControllerType::FixedTime;
    }

    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "FixedTime";
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
        return std::make_unique<FixedTimeController>(*this);
    }

    // ── Configuration ──────────────────────────────────────────────────────

    void set_phases(std::vector<SignalPhase> phases,
                    std::vector<RoadId>      all_incoming_roads = {});

    void set_phase_timings(size_t phase_idx,
                           float  green_s,
                           float  yellow_s,
                           float  all_red_s);

    // ── Factory Helpers ────────────────────────────────────────────────────

    /// Build a standard fixed-time 2-phase controller for an intersection
    /// in the given road network (groups incoming roads into N-S and E-W axes).
    static FixedTimeController make_fixed_time(
        const RoadNetwork& network,
        IntersectionId     intersection_id,
        float              green_s   = 10.0f,
        float              yellow_s  =  3.0f,
        float              all_red_s =  2.0f);

private:
    void update_lights();

    IntersectionId           intersection_id_   { kInvalidIntersectionId };
    std::vector<SignalPhase> phases_;
    std::vector<RoadId>      all_incoming_roads_;

    size_t     current_phase_idx_{ 0 };
    PhaseStage current_stage_    { PhaseStage::Green };
    float      stage_elapsed_s_  { 0.0f };

    std::unordered_map<RoadId, TrafficLight> lights_;
};

} // namespace syntraq
