//
// src/routing/static_routing.cpp
//

#include "syntraq/routing/static_routing.hpp"

namespace syntraq {

StaticRouting::StaticRouting(Algorithm algo, RoadCostFunction cost_fn)
    : algorithm_(algo)
    , cost_fn_(std::move(cost_fn))
{
    rebuild_backends();
}

void StaticRouting::set_algorithm(Algorithm algo) {
    algorithm_ = algo;
    rebuild_backends();
}

void StaticRouting::set_cost_function(RoadCostFunction cost_fn) {
    cost_fn_ = std::move(cost_fn);
    rebuild_backends();
}

void StaticRouting::rebuild_backends() {
    if (algorithm_ == Algorithm::AStar) {
        astar_ = std::make_unique<AStarRouter>(cost_fn_);
        dijkstra_.reset();
    } else {
        dijkstra_ = std::make_unique<DijkstraRouter>(cost_fn_);
        astar_.reset();
    }
}

RouteResult StaticRouting::find_route_with_info(
    const RoadNetwork& network,
    IntersectionId     source,
    IntersectionId     destination) const
{
    if (algorithm_ == Algorithm::AStar && astar_) {
        return astar_->find_route_with_info(network, source, destination);
    }
    if (dijkstra_) {
        return dijkstra_->find_route_with_info(network, source, destination);
    }
    return { {}, 0.0f, 0, false };
}

} // namespace syntraq
