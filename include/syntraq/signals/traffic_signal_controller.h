#pragma once

//
// syntraq/signals/traffic_signal_controller.h
//
// TrafficSignalController — fixed-time baseline intersection signal controller.
// Implements a deterministic finite state machine with configurable:
//   - green duration
//   - yellow duration
//   - red / interphase clearance timing
//
// Maintains signal states (Red, Yellow, Green) for all incoming approach roads.
//

#include "syntraq/signals/signal_phase.h"
#include "syntraq/signals/traffic_light.h"
#include "syntraq/world/ids.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace syntraq {

class RoadNetwork;

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

// ── TrafficSignalController ──────────────────────────────────────────────────

class TrafficSignalController {
public:
    TrafficSignalController() = default;

    /// Construct a controller for an intersection with a sequence of signal phases
    /// and the full set of incoming road IDs to control.
    TrafficSignalController(IntersectionId           intersection_id,
                            std::vector<SignalPhase> phases,
                            std::vector<RoadId>      all_incoming_roads = {});

    // ── Simulation update ──────────────────────────────────────────────────

    /// Advance controller state machine by fixed timestep `dt` (seconds).
    /// Updates all approach signal lights deterministically.
    void tick(float dt);

    /// Reset to Phase 0, Stage Green, elapsed = 0.
    void reset();

    // ── Queries ────────────────────────────────────────────────────────────

    [[nodiscard]] IntersectionId intersection_id() const noexcept {
        return intersection_id_;
    }

    /// Current signal color for the specified incoming road.
    /// Returns SignalColor::Red if the road is not recognized.
    [[nodiscard]] SignalColor signal_for_road(RoadId road_id) const noexcept;

    /// Returns true if traffic on the given incoming road is permitted to enter the junction.
    [[nodiscard]] bool can_proceed(RoadId road_id) const noexcept;

    /// Look up the TrafficLight for an incoming road. Returns nullptr if not found.
    [[nodiscard]] const TrafficLight* light(RoadId road_id) const noexcept;

    /// Map of all controlled incoming roads to their current TrafficLight state.
    [[nodiscard]] const std::unordered_map<RoadId, TrafficLight>& lights() const noexcept {
        return lights_;
    }

    /// Current phase index in the cycle (0-based).
    [[nodiscard]] size_t current_phase_index() const noexcept {
        return current_phase_idx_;
    }

    /// Current active signal phase. Returns nullptr if no phases configured.
    [[nodiscard]] const SignalPhase* current_phase() const noexcept;

    /// Current stage within the active phase (Green, Yellow, or AllRed).
    [[nodiscard]] PhaseStage current_stage() const noexcept {
        return current_stage_;
    }

    /// Elapsed seconds within the current stage.
    [[nodiscard]] float stage_elapsed_s() const noexcept {
        return stage_elapsed_s_;
    }

    /// Total configured duration of the current stage in seconds.
    [[nodiscard]] float current_stage_duration() const noexcept;

    /// Remaining seconds before the current stage transitions.
    [[nodiscard]] float stage_remaining_s() const noexcept;

    /// Total cycle duration (sum of all phases' green + yellow + all-red intervals).
    [[nodiscard]] float cycle_duration_s() const noexcept;

    /// Number of configured phases in the cycle.
    [[nodiscard]] size_t phase_count() const noexcept {
        return phases_.size();
    }

    [[nodiscard]] const std::vector<SignalPhase>& phases() const noexcept {
        return phases_;
    }

    // ── Configuration & Validation ─────────────────────────────────────────

    /// Reconfigure the phases.
    void set_phases(std::vector<SignalPhase> phases,
                    std::vector<RoadId>      all_incoming_roads = {});

    /// Update timings for a specific phase index.
    void set_phase_timings(size_t phase_idx,
                           float  green_s,
                           float  yellow_s,
                           float  all_red_s);

    /// Check if this controller configuration is valid.
    [[nodiscard]] bool is_valid() const noexcept;

    /// Validates the configuration and optionally fills `error_msg`.
    bool validate(std::string* error_msg = nullptr) const;

    // ── Factory Helpers ────────────────────────────────────────────────────

    /// Build a standard fixed-time 2-phase controller for an intersection
    /// in the given road network (groups incoming roads into N-S and E-W axes).
    static TrafficSignalController make_fixed_time(
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
