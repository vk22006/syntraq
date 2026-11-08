#pragma once

//
// syntraq/signals/coordination_types.hpp
//
// Data structures and state definitions for network-level coordinated traffic control:
//   - IntersectionLocalState: local approach queues, active phase, stage, and delays
//   - NeighborIntersectionState: traffic information exchanged between adjacent intersections
//   - CoordinatedDecision: explainable coordination actions (green wave, spillback mitigation, etc.)
//   - CoordinatedControllerConfig: tuning parameters for progression and queue gating
//

#include "syntraq/signals/signal_controller_interface.hpp"
#include "syntraq/world/ids.hpp"

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace syntraq {

// ── CoordinatedDecision ───────────────────────────────────────────────────────

enum class CoordinatedDecision : uint8_t {
    Hold,               ///< Serving mandatory minimum green or clearance phase
    ExtendGreenLocal,   ///< Extending green duration due to local queue demand
    ExtendGreenWave,    ///< Extending green to serve incoming upstream platoon (green wave)
    MitigateSpillback,  ///< Early green termination to prevent downstream queue spillback
    ShortenPhase,       ///< Early gap-out due to dissipated queue with waiting opposing traffic
    SwitchPhase         ///< Normal transition to yellow (max green cap or starvation limit reached)
};

[[nodiscard]] constexpr std::string_view to_string(CoordinatedDecision decision) noexcept {
    switch (decision) {
        case CoordinatedDecision::Hold:              return "Hold";
        case CoordinatedDecision::ExtendGreenLocal:   return "ExtendGreenLocal";
        case CoordinatedDecision::ExtendGreenWave:    return "ExtendGreenWave";
        case CoordinatedDecision::MitigateSpillback:  return "MitigateSpillback";
        case CoordinatedDecision::ShortenPhase:       return "ShortenPhase";
        case CoordinatedDecision::SwitchPhase:        return "SwitchPhase";
    }
    return "Unknown";
}

// ── CoordinatedControllerConfig ───────────────────────────────────────────────

struct CoordinatedControllerConfig {
    float    min_green_s                {  4.0f };   ///< Minimum green time before switching (safety)
    float    max_green_s                { 35.0f };   ///< Maximum green duration cap
    float    nominal_green_s            { 10.0f };   ///< Baseline green duration
    float    max_wait_threshold_s       { 20.0f };   ///< Starvation threshold for cross traffic

    // Network coordination parameters
    uint32_t spillback_queue_threshold  { 5 };       ///< Downstream queue depth triggering spillback mitigation
    float    spillback_occupancy_ratio  { 0.60f };   ///< Downstream occupancy ratio triggering spillback mitigation
    float    progression_window_s       { 6.0f };    ///< ETA window (seconds) to hold/extend green for arriving upstream platoon
    float    platoon_speed_mps          { 13.89f };  ///< Assumed platoon speed along corridor (50 km/h)
    bool     enable_green_wave          { true };    ///< Enable upstream platoon progression
    bool     enable_spillback_prevention{ true };    ///< Enable downstream queue spillback mitigation
};

// ── IntersectionLocalState ────────────────────────────────────────────────────

struct ApproachLocalState {
    RoadId   road_id        { kInvalidRoadId };
    uint32_t queue_length   { 0 };     ///< Queued vehicles on approach (speed < 0.5 m/s)
    float    max_wait_time_s{ 0.0f };  ///< Maximum waiting time of queued vehicle
    float    avg_speed_mps  { 0.0f };  ///< Average speed on approach
};

struct IntersectionLocalState {
    IntersectionId                                intersection_id   { kInvalidIntersectionId };
    size_t                                        active_phase_index{ 0 };
    PhaseStage                                    stage             { PhaseStage::Green };
    float                                         stage_elapsed_s   { 0.0f };
    float                                         stage_remaining_s { 0.0f };
    uint32_t                                      active_phase_queue{ 0 };
    uint32_t                                      cross_phase_queue { 0 };
    float                                         cross_phase_max_wait_s{ 0.0f };
    std::unordered_map<RoadId, ApproachLocalState> approaches        {};
};

// ── NeighborIntersectionState ─────────────────────────────────────────────────

struct NeighborIntersectionState {
    IntersectionId neighbor_id               { kInvalidIntersectionId };
    RoadId         connecting_road           { kInvalidRoadId }; ///< Road linking the two intersections
    bool           is_upstream               { true };           ///< True if neighbor feeds into local intersection
    float          link_length_m             { 0.0f };           ///< Distance between the two intersections

    // Downstream state (when neighbor is downstream of this intersection)
    uint32_t       downstream_queue          { 0 };              ///< Queue on connecting link entering neighbor
    float          downstream_occupancy_ratio{ 0.0f };           ///< Fraction of link occupied by queue [0.0 .. 1.0]
    bool           spillback_imminent        { false };          ///< True if downstream queue is backing up to boundary

    // Upstream state (when neighbor is upstream of this intersection)
    size_t         upstream_phase            { 0 };
    PhaseStage     upstream_stage            { PhaseStage::Green };
    bool           platoon_dispatched        { false };          ///< True if upstream is green and releasing platoon
    float          estimated_platoon_eta_s   { 0.0f };           ///< Estimated arrival time of released platoon (s)
    uint32_t       estimated_platoon_size    { 0 };              ///< Vehicles in approaching platoon
};

} // namespace syntraq

