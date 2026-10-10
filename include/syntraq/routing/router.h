#pragma once

//
// syntraq/routing/router.h
//
// Core routing result structure and IRouter interface.
// Extends IRouteProvider so all router implementations can be directly used
// by VehicleSpawner, Vehicle, and Simulation without modification.
//

#include "syntraq/routing/route_cost.h"
#include "syntraq/vehicles/route_provider.h"
#include "syntraq/world/ids.h"
#include "syntraq/world/road_network.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace syntraq {

/// Detailed result of a route search.
struct RouteResult {
    std::vector<RoadId> roads;              ///< Ordered list of road IDs from source to destination
    float               total_cost{ 0.0f }; ///< Sum of edge weights along the route
    uint32_t            nodes_visited{ 0 }; ///< Total nodes popped/expanded during search
    bool                success{ false };   ///< True if a valid path was found (or trivial source==dest)
};

/// Base interface for advanced pathfinding engines (Dijkstra, A*, etc.).
/// Inherits from IRouteProvider for seamless integration with the vehicle subsystem.
class IRouter : public IRouteProvider {
public:
    ~IRouter() override = default;

    /// Find route with full metrics (cost, nodes visited, success flag).
    [[nodiscard]] virtual RouteResult find_route_with_info(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const = 0;

    /// Implementation of IRouteProvider::find_route.
    /// Returns the ordered road IDs, or an empty vector if no route exists.
    [[nodiscard]] std::vector<RoadId> find_route(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const override {
        return find_route_with_info(network, source, destination).roads;
    }

    /// Descriptive name of the routing algorithm (e.g. "Dijkstra", "A*").
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

} // namespace syntraq
