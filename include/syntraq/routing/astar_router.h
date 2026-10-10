#pragma once

//
// syntraq/routing/astar_router.h
//
// AStarRouter — Goal-directed shortest path planning using the A* algorithm.
// Supports arbitrary non-negative road cost functions and admissible heuristics
// (default: Euclidean distance heuristic).
//

#include "syntraq/routing/heuristic.h"
#include "syntraq/routing/router.h"

namespace syntraq {

class AStarRouter : public IRouter {
public:
    explicit AStarRouter(RoadCostFunction  cost_fn      = DistanceCost{},
                         HeuristicFunction heuristic_fn = EuclideanDistanceHeuristic{});

    [[nodiscard]] RouteResult find_route_with_info(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "A*";
    }

    void set_cost_function(RoadCostFunction cost_fn) {
        cost_fn_ = std::move(cost_fn);
    }

    void set_heuristic_function(HeuristicFunction heuristic_fn) {
        heuristic_fn_ = std::move(heuristic_fn);
    }

    [[nodiscard]] const RoadCostFunction& cost_function() const noexcept {
        return cost_fn_;
    }

    [[nodiscard]] const HeuristicFunction& heuristic_function() const noexcept {
        return heuristic_fn_;
    }

private:
    RoadCostFunction  cost_fn_;
    HeuristicFunction heuristic_fn_;
};

} // namespace syntraq
