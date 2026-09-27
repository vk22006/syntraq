#pragma once

//
// syntraq/world/road.h
//
// Lane and Road — the directed edge model of the road network.
//
// Design choices:
//   - A Road is DIRECTED: it connects Intersection `from` → `to`.
//     Bi-directional streets are represented as two Road objects (one per
//     direction). This keeps pathfinding algorithms simple and explicit.
//   - Lanes are value types stored inside Road — no separate heap allocation.
//   - Speed limits and lengths are stored in SI units (m/s, metres) so
//     simulation arithmetic stays clean.
//

#include "syntraq/world/ids.h"
#include <string>
#include <vector>

namespace syntraq {

// ── Lane ─────────────────────────────────────────────────────────────────────

/// A single lane within a road.
/// Lanes share the road's length; they differ only in index (for rendering).
struct Lane {
    uint32_t index{ 0 };        ///< 0-based lane index within the road
    float    width_m{ 3.5f };   ///< Lane width in metres (standard ~3.5 m)
};

// ── Road ─────────────────────────────────────────────────────────────────────

/// A directed road segment connecting two intersections.
/// One Road = one direction of travel.
struct Road {
    RoadId         id{ kInvalidRoadId };
    IntersectionId from{ kInvalidIntersectionId };
    IntersectionId to  { kInvalidIntersectionId };

    std::string    name;            ///< Optional label (e.g. "Main St N→S")
    float          length_m{ 0.f }; ///< Road length in metres
    float          speed_limit_mps{ 13.89f }; ///< Default 50 km/h in m/s

    std::vector<Lane> lanes;        ///< One or more lanes; index 0 = leftmost

    // ── Derived helpers ───────────────────────────────────────────────────

    /// Number of lanes in this road.
    [[nodiscard]] uint32_t lane_count() const noexcept {
        return static_cast<uint32_t>(lanes.size());
    }

    /// Total road width (sum of all lane widths).
    [[nodiscard]] float total_width_m() const noexcept {
        float w = 0.f;
        for (const auto& l : lanes) w += l.width_m;
        return w;
    }

    /// True if this road connects valid intersections and has at least one lane.
    [[nodiscard]] bool is_valid() const noexcept {
        return id        != kInvalidRoadId
            && from      != kInvalidIntersectionId
            && to        != kInvalidIntersectionId
            && from      != to
            && length_m  >  0.f
            && !lanes.empty();
    }
};

} // namespace syntraq
