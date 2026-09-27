//
// src/world/road_network.cpp
//

#include "syntraq/world/road_network.h"

#include <format>
#include <stdexcept>

namespace syntraq {

// ── Mutation ──────────────────────────────────────────────────────────────────

IntersectionId RoadNetwork::add_intersection(Vec2 position, std::string name) {
    const auto id = static_cast<IntersectionId>(next_intersection_id_++);

    Intersection node;
    node.id       = id;
    node.position = position;
    node.name     = std::move(name);

    intersections_.emplace(id, std::move(node));
    return id;
}

RoadId RoadNetwork::add_road(IntersectionId from,
                             IntersectionId to,
                             float          length_m,
                             float          speed_limit_mps,
                             uint32_t       num_lanes,
                             std::string    name) {
    // Validate both intersections exist
    if (!has_intersection(from) || !has_intersection(to)) {
        return kInvalidRoadId;
    }
    // Reject self-loops
    if (from == to) {
        return kInvalidRoadId;
    }
    // Reject duplicate directed road
    if (has_road(from, to)) {
        return kInvalidRoadId;
    }
    // Reject degenerate length
    if (length_m <= 0.f) {
        return kInvalidRoadId;
    }
    // Need at least one lane
    if (num_lanes == 0) {
        return kInvalidRoadId;
    }

    const auto id = static_cast<RoadId>(next_road_id_++);

    Road r;
    r.id              = id;
    r.from            = from;
    r.to              = to;
    r.length_m        = length_m;
    r.speed_limit_mps = speed_limit_mps;
    r.name            = std::move(name);

    r.lanes.resize(num_lanes);
    for (uint32_t i = 0; i < num_lanes; ++i) {
        r.lanes[i].index = i;
        r.lanes[i].width_m = 3.5f;
    }

    // Wire adjacency lists in the endpoint intersections
    intersections_.at(from).outgoing_roads.push_back(id);
    intersections_.at(to  ).incoming_roads.push_back(id);

    roads_.emplace(id, std::move(r));
    return id;
}

// ── Query ─────────────────────────────────────────────────────────────────────

const Intersection* RoadNetwork::intersection(IntersectionId id) const {
    auto it = intersections_.find(id);
    return (it != intersections_.end()) ? &it->second : nullptr;
}

Intersection* RoadNetwork::intersection(IntersectionId id) {
    auto it = intersections_.find(id);
    return (it != intersections_.end()) ? &it->second : nullptr;
}

const Road* RoadNetwork::road(RoadId id) const {
    auto it = roads_.find(id);
    return (it != roads_.end()) ? &it->second : nullptr;
}

Road* RoadNetwork::road(RoadId id) {
    auto it = roads_.find(id);
    return (it != roads_.end()) ? &it->second : nullptr;
}

std::vector<RoadId> RoadNetwork::outgoing_roads(IntersectionId id) const {
    const Intersection* node = intersection(id);
    if (!node) return {};
    return node->outgoing_roads;
}

std::vector<RoadId> RoadNetwork::incoming_roads(IntersectionId id) const {
    const Intersection* node = intersection(id);
    if (!node) return {};
    return node->incoming_roads;
}

bool RoadNetwork::has_road(IntersectionId from, IntersectionId to) const {
    const Intersection* src = intersection(from);
    if (!src) return false;
    for (RoadId rid : src->outgoing_roads) {
        const Road* r = road(rid);
        if (r && r->to == to) return true;
    }
    return false;
}

bool RoadNetwork::has_intersection(IntersectionId id) const {
    return intersections_.count(id) > 0;
}

void RoadNetwork::clear() {
    intersections_.clear();
    roads_.clear();
    next_intersection_id_ = 0;
    next_road_id_         = 0;
}

// ── Test-map factory ──────────────────────────────────────────────────────────

RoadNetwork make_test_map(Vec2 origin, float spacing_px) {
    // 4 columns × 3 rows = 12 intersections
    // Layout (col, row):
    //   (0,0) (1,0) (2,0) (3,0)
    //   (0,1) (1,1) (2,1) (3,1)
    //   (0,2) (1,2) (2,2) (3,2)

    constexpr int COLS = 4;
    constexpr int ROWS = 3;

    RoadNetwork net;

    // ── Add intersections ─────────────────────────────────────────────────
    // ids_[row][col] — assigned sequentially, so id = row*COLS + col
    IntersectionId ids[ROWS][COLS];
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            Vec2 pos{
                origin.x + static_cast<float>(col) * spacing_px,
                origin.y + static_cast<float>(row) * spacing_px
            };
            const std::string label =
                "I(" + std::to_string(col) + "," + std::to_string(row) + ")";
            ids[row][col] = net.add_intersection(pos, label);
        }
    }

    // ── Helper to add a bi-directional road pair ──────────────────────────
    // Horizontal roads: 50 km/h, 2 lanes, length = spacing_px (1 px ≈ 1 m)
    // Vertical   roads: 50 km/h, 2 lanes, length = spacing_px
    auto add_bidirectional = [&](IntersectionId a, IntersectionId b,
                                 float len_m, const std::string& label) {
        net.add_road(a, b, len_m, 13.89f, 2, label + " →");
        net.add_road(b, a, len_m, 13.89f, 2, label + " ←");
    };

    // ── Horizontal edges ──────────────────────────────────────────────────
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS - 1; ++col) {
            const std::string lbl =
                "H(" + std::to_string(col) + "-" + std::to_string(col + 1) +
                "," + std::to_string(row) + ")";
            add_bidirectional(ids[row][col], ids[row][col + 1],
                              spacing_px, lbl);
        }
    }

    // ── Vertical edges ────────────────────────────────────────────────────
    for (int row = 0; row < ROWS - 1; ++row) {
        for (int col = 0; col < COLS; ++col) {
            const std::string lbl =
                "V(" + std::to_string(col) + "," +
                std::to_string(row) + "-" + std::to_string(row + 1) + ")";
            add_bidirectional(ids[row][col], ids[row + 1][col],
                              spacing_px, lbl);
        }
    }

    return net;
}

} // namespace syntraq
