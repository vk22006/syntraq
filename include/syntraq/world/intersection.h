#pragma once

//
// syntraq/world/intersection.h
//
// Intersection — a graph node in the road network.
//
// Holds:
//   - A screen-space position (pixels) for rendering.
//   - An optional name for debugging / UI.
//   - Lists of outgoing and incoming road IDs (maintained by RoadNetwork).
//

#include "syntraq/world/ids.h"
#include <string>
#include <vector>

namespace syntraq {

/// 2D position in screen / world coordinates (pixels).
struct Vec2 {
    float x{ 0.f };
    float y{ 0.f };
};

/// A node in the directed road graph.
struct Intersection {
    IntersectionId id{ kInvalidIntersectionId };
    Vec2           position;          ///< Pixel coordinates (top-left origin)
    std::string    name;              ///< Optional label

    /// IDs of roads that leave this intersection (outgoing edges).
    std::vector<RoadId> outgoing_roads;

    /// IDs of roads that arrive at this intersection (incoming edges).
    /// Kept for reverse-traversal and rendering; not needed by basic Dijkstra.
    std::vector<RoadId> incoming_roads;

    /// True if this intersection has been assigned a valid ID.
    [[nodiscard]] bool is_valid() const noexcept {
        return id != kInvalidIntersectionId;
    }
};

} // namespace syntraq
