#pragma once

//
// syntraq/signals/signal_controller_interface.hpp
//
// ITrafficSignalController — Common abstract interface for all traffic signal controllers:
// - FixedTimeController (baseline fixed interval cycling)
// - AdaptiveController (demand/queue-responsive actuated control)
// - Learning / RL controllers (future adaptive policies)
//

#include "syntraq/signals/signal_phase.hpp"
#include "syntraq/signals/traffic_light.hpp"
#include "syntraq/world/ids.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace syntraq {

// ── PhaseStage ───────────────────────────────────────────────────────────────

enum class PhaseStage : uint8_t {
    Green,
    Yellow,
    AllRed
};

[[nodiscard]] constexpr std::string_view to_string(PhaseStage stage) noexcept {
    switch (stage) {
        case PhaseStage::Green:  return "Green";
        case PhaseStage::Yellow: return "Yellow";
        case PhaseStage::AllRed: return "AllRed";
    }
    return "Unknown";
}

// ── SignalControllerType ─────────────────────────────────────────────────────

enum class SignalControllerType : uint8_t {
    FixedTime,
    Adaptive
};

[[nodiscard]] constexpr std::string_view to_string(SignalControllerType type) noexcept {
    switch (type) {
        case SignalControllerType::FixedTime: return "FixedTime";
        case SignalControllerType::Adaptive:  return "Adaptive";
    }
    return "Unknown";
}

// ── ITrafficSignalController ─────────────────────────────────────────────────

class ITrafficSignalController {
public:
    virtual ~ITrafficSignalController() = default;

    // ── Simulation update ──────────────────────────────────────────────────

    /// Advance controller state machine by fixed timestep `dt` (seconds).
    virtual void tick(float dt) = 0;

    /// Reset to Phase 0, Stage Green, elapsed = 0.
    virtual void reset() = 0;

    /// Observe incoming approach traffic conditions (queue lengths and waiting times).
    /// Used by adaptive and learning controllers to make timing adjustments.
    virtual void observe_traffic(const std::unordered_map<RoadId, uint32_t>& queues,
                                 const std::unordered_map<RoadId, float>&    wait_times = {}) = 0;

    // ── Identification & Queries ───────────────────────────────────────────

    [[nodiscard]] virtual IntersectionId intersection_id() const noexcept = 0;
    [[nodiscard]] virtual SignalControllerType type() const noexcept = 0;
    [[nodiscard]] virtual std::string_view type_name() const noexcept = 0;

    /// Current signal color for the specified incoming road.
    [[nodiscard]] virtual SignalColor signal_for_road(RoadId road_id) const noexcept = 0;

    /// Returns true if traffic on the given incoming road is permitted to enter the junction.
    [[nodiscard]] virtual bool can_proceed(RoadId road_id) const noexcept = 0;

    /// Look up the TrafficLight for an incoming road. Returns nullptr if not found.
    [[nodiscard]] virtual const TrafficLight* light(RoadId road_id) const noexcept = 0;

    /// Map of all controlled incoming roads to their current TrafficLight state.
    [[nodiscard]] virtual const std::unordered_map<RoadId, TrafficLight>& lights() const noexcept = 0;

    /// Current phase index in the cycle (0-based).
    [[nodiscard]] virtual size_t current_phase_index() const noexcept = 0;

    /// Current active signal phase. Returns nullptr if no phases configured.
    [[nodiscard]] virtual const SignalPhase* current_phase() const noexcept = 0;

    /// Current stage within the active phase (Green, Yellow, or AllRed).
    [[nodiscard]] virtual PhaseStage current_stage() const noexcept = 0;

    /// Elapsed seconds within the current stage.
    [[nodiscard]] virtual float stage_elapsed_s() const noexcept = 0;

    /// Total configured duration of the current stage in seconds.
    [[nodiscard]] virtual float current_stage_duration() const noexcept = 0;

    /// Remaining seconds before the current stage transitions.
    [[nodiscard]] virtual float stage_remaining_s() const noexcept = 0;

    /// Total cycle duration (sum of all phases' green + yellow + all-red intervals).
    [[nodiscard]] virtual float cycle_duration_s() const noexcept = 0;

    /// Number of configured phases in the cycle.
    [[nodiscard]] virtual size_t phase_count() const noexcept = 0;

    [[nodiscard]] virtual const std::vector<SignalPhase>& phases() const noexcept = 0;

    // ── Configuration ──────────────────────────────────────────────────────

    virtual void set_phases(std::vector<SignalPhase> phases,
                            std::vector<RoadId>      all_incoming_roads = {}) = 0;

    virtual void set_phase_timings(size_t phase_idx,
                                   float  green_s,
                                   float  yellow_s,
                                   float  all_red_s) = 0;

    // ── Validation & Cloning ───────────────────────────────────────────────

    [[nodiscard]] virtual bool is_valid() const noexcept = 0;
    virtual bool validate(std::string* error_msg = nullptr) const = 0;

    [[nodiscard]] virtual std::unique_ptr<ITrafficSignalController> clone() const = 0;
};

} // namespace syntraq
