//
// src/vehicles/route_provider.cpp
//

#include "syntraq/vehicles/route_provider.h"

#include <unordered_set>

namespace syntraq {

std::vector<RoadId>
GreedyRouteProvider::find_route(const RoadNetwork &network,
                                IntersectionId source,
                                IntersectionId destination) const {
  if (source == destination)
    return {};

  std::vector<RoadId> route;
  std::unordered_set<IntersectionId> visited;

  IntersectionId current = source;
  visited.insert(current);

  const uint32_t max_steps = network.intersection_count();

  for (uint32_t step = 0; step < max_steps; ++step) {
    if (current == destination)
      break;

    const Intersection *node = network.intersection(current);
    if (!node || node->outgoing_roads.empty())
      break;

    RoadId best_road = kInvalidRoadId;
    IntersectionId next = kInvalidIntersectionId;
    bool found_dest = false;

    for (RoadId rid : node->outgoing_roads) {
      const Road *r = network.road(rid);
      if (!r)
        continue;
      if (r->to == destination) {
        best_road = rid;
        next = destination;
        found_dest = true;
        break;
      }
      if (!found_dest && visited.find(r->to) == visited.end()) {
        best_road = rid;
        next = r->to;
      }
    }

    if (best_road == kInvalidRoadId)
      break;

    route.push_back(best_road);
    visited.insert(next);
    current = next;
  }

  if (current != destination)
    return {};
  return route;
}

} // namespace syntraq
