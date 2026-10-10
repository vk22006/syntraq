#pragma once

//
// syntraq/routing/heuristic.h
//
// Heuristic function abstractions for A* search.
// Provides Euclidean distance (default), Manhattan distance, and zero heuristic.
//

#include "syntraq/world/road_network.h"
#include <cmath>
#include <functional>

namespace syntraq {

/// Callable type for A* heuristic estimation: h(network, current, target) -> float.
using HeuristicFunction = std::function<float(const RoadNetwork& network,
                                              IntersectionId     current,
                                              IntersectionId     target)>;

/// Euclidean straight-line distance heuristic.
/// Admissible and consistent for distance-based edge weights when 1 px = 1 m.
struct EuclideanDistanceHeuristic {
    [[nodiscard]] float operator()(const RoadNetwork& network,
                                   IntersectionId     current,
                                   IntersectionId     target) const noexcept {
        const auto* u = network.intersection(current);
        const auto* v = network.intersection(target);
        if (!u || !v) return 0.0f;
        return std::hypot(v->position.x - u->position.x,
                          v->position.y - u->position.y);
    }
};

/// Zero heuristic: h(n) = 0.
/// Degenerates A* to Dijkstra; strictly admissible and consistent on any graph.
struct ZeroHeuristic {
    [[nodiscard]] float operator()(const RoadNetwork& /*network*/,
                                   IntersectionId     /*current*/,
                                   IntersectionId     /*target*/) const noexcept {
        return 0.0f;
    }
};

/// Manhattan distance heuristic.
struct ManhattanDistanceHeuristic {
    [[nodiscard]] float operator()(const RoadNetwork& network,
                                   IntersectionId     current,
                                   IntersectionId     target) const noexcept {
        const auto* u = network.intersection(current);
        const auto* v = network.intersection(target);
        if (!u || !v) return 0.0f;
        return std::abs(v->position.x - u->position.x) +
               std::abs(v->position.y - u->position.y);
    }
};

} // namespace syntraq
