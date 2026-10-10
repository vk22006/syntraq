#pragma once

//
// syntraq/routing/dijkstra_router.h
//
// DijkstraRouter — Shortest path planning using Dijkstra's algorithm.
// Supports arbitrary non-negative road cost functions (default: distance in metres).
//

#include "syntraq/routing/router.h"

namespace syntraq {

class DijkstraRouter : public IRouter {
public:
    explicit DijkstraRouter(RoadCostFunction cost_fn = DistanceCost{});

    [[nodiscard]] RouteResult find_route_with_info(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "Dijkstra";
    }

    void set_cost_function(RoadCostFunction cost_fn) {
        cost_fn_ = std::move(cost_fn);
    }

    [[nodiscard]] const RoadCostFunction& cost_function() const noexcept {
        return cost_fn_;
    }

private:
    RoadCostFunction cost_fn_;
};

} // namespace syntraq
