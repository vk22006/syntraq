#pragma once

//
// syntraq/signals/coordinated_controller.hpp
//
// CoordinatedController — Network-level multi-intersection traffic signal controller.
// Extends local actuated control by exchanging real-time state with neighboring intersections:
//   - Downstream spillback gating: terminates or withholds green when downstream links are blocked,
//     preventing queue propagation across the network.
//   - Upstream green wave progression: detects approaching vehicle platoons from upstream greens
//     and holds/extends green to allow smooth progression without stopping.
//   - Single-intersection backward compatibility: seamlessly operates as a local adaptive
//     controller when no neighbors are present.
//

#include "syntraq/signals/coordination_types.hpp"
#include "syntraq/signals/signal_controller_interface.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace syntraq {

class RoadNetwork;

class CoordinatedController : public ITrafficSignalController {
public:
    CoordinatedController() = default;

    CoordinatedController(
        IntersectionId              intersection_id,
        std::vector<SignalPhase>    phases,
        std::vector<RoadId>         all_incoming_roads = {},
        CoordinatedControllerConfig config = {});

    // ── ITrafficSignalController Overrides ─────────────────────────────────

    void tick(float dt) override;
    void reset() override;

    void observe_traffic(const std::unordered_map<RoadId, uint32_t>& queues,
                         const std::unordered_map<RoadId, float>&    wait_times = {}) override;

    [[nodiscard]] IntersectionId intersection_id() const noexcept override {
        return intersection_id_;
    }

    [[nodiscard]] SignalControllerType type() const noexcept override {
        return SignalControllerType::Coordinated;
    }

    [[nodiscard]] std::string_view type_name() const noexcept override {
        return "Coordinated";
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

    [[nodiscard]] const SignalPhase* current_phase() const noexcept override {
        if (phases_.empty() || current_phase_idx_ >= phases_.size()) return nullptr;
        return &phases_[current_phase_idx_];
    }

    [[nodiscard]] PhaseStage current_stage() const noexcept override {
        return current_stage_;
    }

    [[nodiscard]] float stage_elapsed_s() const noexcept override {
        return stage_elapsed_s_;
    }

    [[nodiscard]] float current_stage_duration() const noexcept override {
        return current_stage_duration_;
    }

    [[nodiscard]] float stage_remaining_s() const noexcept override {
        return std::max(0.0f, current_stage_duration_ - stage_elapsed_s_);
    }

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

    [[nodiscard]] bool is_valid() const noexcept override {
        return !phases_.empty();
    }

    bool validate(std::string* error_msg = nullptr) const override;

    [[nodiscard]] std::unique_ptr<ITrafficSignalController> clone() const override {
        return std::make_unique<CoordinatedController>(*this);
    }

    // ── Neighboring State & Coordination ──────────────────────────────────

    /// Receive traffic state update from a single neighboring intersection
    void receive_neighbor_state(const NeighborIntersectionState& state);

    /// Ingest a collection of neighboring intersection states
    void receive_neighbor_states(const std::vector<NeighborIntersectionState>& states);

    /// Clear stored neighbor states (e.g. at reset)
    void clear_neighbor_states() noexcept;

    // ── Explainable Telemetry & Inspection ─────────────────────────────────

    [[nodiscard]] CoordinatedDecision last_decision() const noexcept {
        return last_decision_;
    }

    [[nodiscard]] const std::string& last_reason() const noexcept {
        return last_reason_;
    }

    [[nodiscard]] std::string_view last_decision_reason() const noexcept {
        return last_reason_;
    }

    [[nodiscard]] const IntersectionLocalState& local_state() const noexcept {
        return local_state_;
    }

    [[nodiscard]] const std::vector<NeighborIntersectionState>& neighbor_states() const noexcept {
        return neighbor_states_;
    }

    [[nodiscard]] const CoordinatedControllerConfig& config() const noexcept {
        return config_;
    }

    void set_config(const CoordinatedControllerConfig& config) noexcept {
        config_ = config;
    }

    [[nodiscard]] uint32_t green_wave_extensions() const noexcept {
        return green_wave_extensions_count_;
    }

    [[nodiscard]] uint32_t green_wave_extensions_count() const noexcept {
        return green_wave_extensions_count_;
    }

    [[nodiscard]] uint32_t spillback_mitigations() const noexcept {
        return spillback_mitigations_count_;
    }

    [[nodiscard]] uint32_t spillback_mitigations_count() const noexcept {
        return spillback_mitigations_count_;
    }

    // ── Helper Factory ─────────────────────────────────────────────────────

    static CoordinatedController make_default_2phase(
        const RoadNetwork&          network,
        IntersectionId              id,
        CoordinatedControllerConfig config = {},
        float                       yellow_duration_s = 3.0f,
        float                       all_red_duration_s = 2.0f);

private:
    IntersectionId                               intersection_id_{ kInvalidIntersectionId };
    std::vector<SignalPhase>                     phases_{};
    std::unordered_map<RoadId, TrafficLight>     lights_{};
    size_t                                       current_phase_idx_{ 0 };
    PhaseStage                                   current_stage_{ PhaseStage::Green };
    float                                        stage_elapsed_s_{ 0.0f };
    float                                        current_stage_duration_{ 10.0f };

    CoordinatedControllerConfig                  config_{};
    IntersectionLocalState                       local_state_{};
    std::vector<NeighborIntersectionState>       neighbor_states_{};
    CoordinatedDecision                          last_decision_{ CoordinatedDecision::Hold };
    std::string                                  last_reason_{ "Initial green stage" };

    uint32_t                                     green_wave_extensions_count_{ 0 };
    uint32_t                                     spillback_mitigations_count_{ 0 };

    void apply_stage_lights();
    void transition_to_stage(PhaseStage next_stage);
    void advance_to_next_phase();
    [[nodiscard]] CoordinatedDecision evaluate_coordination_decision();
    void refresh_local_state(const std::unordered_map<RoadId, uint32_t>& queues,
                             const std::unordered_map<RoadId, float>&    wait_times);
};

} // namespace syntraq
