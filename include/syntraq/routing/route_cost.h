#pragma once

//
// syntraq/routing/route_cost.h
//
// Cost function abstractions for edge traversal weighting.
// Allows routing algorithms to support distance, travel time, or custom weights.
//

#include "syntraq/world/road.h"
#include <functional>

namespace syntraq {

/// Callable type for calculating the traversal cost of a road edge.
using RoadCostFunction = std::function<float(const Road& road)>;

/// Distance-based cost (metres). Default edge weighting strategy.
struct DistanceCost {
    [[nodiscard]] float operator()(const Road& road) const noexcept {
        return road.length_m;
    }
};

/// Free-flow travel time cost (seconds).
/// Uses length / speed limit, falling back to length if speed limit is invalid.
struct FreeFlowTravelTimeCost {
    [[nodiscard]] float operator()(const Road& road) const noexcept {
        if (road.speed_limit_mps <= 0.0f) {
            return road.length_m;
        }
        return road.length_m / road.speed_limit_mps;
    }
};

} // namespace syntraq
