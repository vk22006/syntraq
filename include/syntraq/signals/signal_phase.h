#pragma once

//
// syntraq/signals/signal_phase.h
//
// Represents one phase in an intersection signal cycle.
// Holds configurable stage durations (green, yellow, interphase all-red)
// and the set of incoming roads that receive right-of-way during this phase.
//

#include "syntraq/world/ids.h"

#include <algorithm>
#include <string>
#include <vector>

namespace syntraq {

// ── SignalPhase ──────────────────────────────────────────────────────────────

struct SignalPhase {
    std::string         name;                ///< Human-readable label (e.g. "North-South")
    std::vector<RoadId> green_roads;         ///< Incoming roads that receive Green during this phase
    float               green_duration_s  { 10.0f }; ///< Green interval in seconds
    float               yellow_duration_s {  3.0f }; ///< Yellow interval in seconds
    float               all_red_duration_s{  2.0f }; ///< Interphase all-red clearance in seconds

    /// Total duration of this phase (green + yellow + clearance).
    [[nodiscard]] float total_duration_s() const noexcept {
        return green_duration_s + yellow_duration_s + all_red_duration_s;
    }

    /// True if the given road has right-of-way in this phase.
    [[nodiscard]] bool contains_road(RoadId id) const noexcept {
        return std::find(green_roads.begin(), green_roads.end(), id) != green_roads.end();
    }

    /// Basic sanity validation.
    [[nodiscard]] bool is_valid() const noexcept {
        return !green_roads.empty()
            && green_duration_s > 0.0f
            && yellow_duration_s >= 0.0f
            && all_red_duration_s >= 0.0f;
    }
};

} // namespace syntraq
