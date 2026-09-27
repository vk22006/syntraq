#pragma once

//
// syntraq/world/road_network.h
//
// RoadNetwork — the directed graph that models the urban road layout.
//
// Ownership model:
//   - RoadNetwork owns all Intersection and Road objects.
//   - Stored in std::unordered_map by ID — stable references, no raw owning
//     pointers exposed to callers.
//   - ID generation is internal: callers receive the assigned ID on add.
//
// Graph model:
//   - Nodes  = Intersections
//   - Edges  = Roads (directed)
//   - Bi-directional street = two Road objects, one per direction
//   - Adjacency is stored in Intersection::outgoing_roads for fast iteration
//
// This structure is directly suitable for Dijkstra / A* in Milestone 3.
//

#include "syntraq/world/intersection.h"
#include "syntraq/world/road.h"

#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace syntraq {

class RoadNetwork {
public:
    RoadNetwork() = default;

    // Non-copyable (owns the graph data).
    // Move is allowed for factory functions.
    RoadNetwork(const RoadNetwork&)            = delete;
    RoadNetwork& operator=(const RoadNetwork&) = delete;
    RoadNetwork(RoadNetwork&&)                 = default;
    RoadNetwork& operator=(RoadNetwork&&)      = default;

    // ── Mutation API ──────────────────────────────────────────────────────

    /// Add an intersection at the given position with an optional name.
    /// Returns the assigned IntersectionId.
    IntersectionId add_intersection(Vec2 position, std::string name = {});

    /// Add a directed road from `from` to `to`.
    /// `lanes` specifies how many lanes (default 1).
    /// Returns kInvalidRoadId if either intersection does not exist,
    /// or if a duplicate road (same from→to) already exists.
    RoadId add_road(IntersectionId from,
                    IntersectionId to,
                    float          length_m,
                    float          speed_limit_mps = 13.89f,  // 50 km/h
                    uint32_t       num_lanes       = 1,
                    std::string    name            = {});

    // ── Query API ─────────────────────────────────────────────────────────

    /// Look up an intersection by ID. Returns nullptr if not found.
    [[nodiscard]] const Intersection* intersection(IntersectionId id) const;
    [[nodiscard]]       Intersection* intersection(IntersectionId id);

    /// Look up a road by ID. Returns nullptr if not found.
    [[nodiscard]] const Road* road(RoadId id) const;
    [[nodiscard]]       Road* road(RoadId id);

    /// All roads leaving a given intersection (outgoing edges).
    /// Returns an empty span if the intersection doesn't exist.
    [[nodiscard]] std::vector<RoadId> outgoing_roads(IntersectionId id) const;

    /// All roads arriving at a given intersection (incoming edges).
    [[nodiscard]] std::vector<RoadId> incoming_roads(IntersectionId id) const;

    /// True if a road with the given from→to already exists.
    [[nodiscard]] bool has_road(IntersectionId from, IntersectionId to) const;

    /// True if the intersection exists.
    [[nodiscard]] bool has_intersection(IntersectionId id) const;

    // ── Collection accessors ──────────────────────────────────────────────

    [[nodiscard]] const std::unordered_map<IntersectionId, Intersection>&
        intersections() const noexcept { return intersections_; }

    [[nodiscard]] const std::unordered_map<RoadId, Road>&
        roads() const noexcept { return roads_; }

    [[nodiscard]] uint32_t intersection_count() const noexcept {
        return static_cast<uint32_t>(intersections_.size());
    }

    [[nodiscard]] uint32_t road_count() const noexcept {
        return static_cast<uint32_t>(roads_.size());
    }

    /// Remove all intersections and roads.
    void clear();

private:
    std::unordered_map<IntersectionId, Intersection> intersections_;
    std::unordered_map<RoadId,         Road>         roads_;

    uint32_t next_intersection_id_{ 0 };
    uint32_t next_road_id_        { 0 };
};

// ── Test-map factory ──────────────────────────────────────────────────────────

/// Build a small 4×3 grid road network for visualisation and tests.
/// Intersections are laid out with `spacing_px` pixels between them,
/// with `origin` as the top-left corner.
/// Every adjacent pair of intersections is connected by two directed roads
/// (one per direction), each with two lanes and a 50 km/h speed limit.
RoadNetwork make_test_map(Vec2     origin     = { 160.f, 120.f },
                          float    spacing_px = 180.f);

} // namespace syntraq
