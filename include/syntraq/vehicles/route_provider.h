#pragma once

//
// syntraq/vehicles/route_provider.h
//
// Route provider abstraction for vehicle navigation.
// Defines IRouteProvider interface and GreedyRouteProvider (M3/M3.5).
// Milestone 4 will add Dijkstra / A* implementations without modifying
// the vehicle or spawner core logic.
//

#include "syntraq/world/ids.h"
#include "syntraq/world/road_network.h"

#include <memory>
#include <vector>

namespace syntraq {

/// Interface for route calculation on a RoadNetwork.
class IRouteProvider {
public:
    virtual ~IRouteProvider() = default;

    /// Build a route from source to destination intersection.
    /// Returns an ordered list of RoadIds, or an empty vector if no route exists.
    [[nodiscard]] virtual std::vector<RoadId> find_route(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const = 0;
};

/// Greedy BFS-like walk route provider (temporary mechanism from M3).
class GreedyRouteProvider : public IRouteProvider {
public:
    [[nodiscard]] std::vector<RoadId> find_route(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const override;
};

} // namespace syntraq
