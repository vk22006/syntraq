//
// src/routing/dijkstra_router.cpp
//

#include "syntraq/routing/dijkstra_router.h"

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace syntraq {

namespace {

struct PQNode {
    float          cost;
    IntersectionId id;

    bool operator>(const PQNode& other) const noexcept {
        if (cost != other.cost) return cost > other.cost;
        return id > other.id;
    }
};

struct Predecessor {
    IntersectionId from_node{ kInvalidIntersectionId };
    RoadId         road_id  { kInvalidRoadId };
};

} // namespace

DijkstraRouter::DijkstraRouter(RoadCostFunction cost_fn)
    : cost_fn_(std::move(cost_fn)) {}

RouteResult DijkstraRouter::find_route_with_info(
    const RoadNetwork& network,
    IntersectionId     source,
    IntersectionId     destination) const {
    // Basic existence checks
    if (!network.has_intersection(source) || !network.has_intersection(destination)) {
        return { {}, 0.0f, 0, false };
    }

    // Trivial query
    if (source == destination) {
        return { {}, 0.0f, 0, true };
    }

    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> open_set;
    std::unordered_map<IntersectionId, float>       dist;
    std::unordered_map<IntersectionId, Predecessor> came_from;

    dist[source] = 0.0f;
    open_set.push({ 0.0f, source });

    uint32_t nodes_visited = 0;
    bool reached = false;

    while (!open_set.empty()) {
        auto [current_cost, u] = open_set.top();
        open_set.pop();

        auto dist_it = dist.find(u);
        if (dist_it != dist.end() && current_cost > dist_it->second) {
            continue;
        }

        nodes_visited++;

        if (u == destination) {
            reached = true;
            break;
        }

        const Intersection* u_node = network.intersection(u);
        if (!u_node) continue;

        for (RoadId rid : u_node->outgoing_roads) {
            const Road* r = network.road(rid);
            if (!r) continue;

            IntersectionId v = r->to;
            float edge_cost = cost_fn_ ? cost_fn_(*r) : r->length_m;
            if (edge_cost < 0.0f) edge_cost = 0.0f;

            float new_cost = current_cost + edge_cost;
            auto it = dist.find(v);
            if (it == dist.end() || new_cost < it->second) {
                dist[v] = new_cost;
                came_from[v] = Predecessor{ u, rid };
                open_set.push({ new_cost, v });
            }
        }
    }

    if (!reached) {
        return { {}, 0.0f, nodes_visited, false };
    }

    std::vector<RoadId> path;
    IntersectionId curr = destination;
    while (curr != source) {
        auto it = came_from.find(curr);
        if (it == came_from.end()) {
            return { {}, 0.0f, nodes_visited, false };
        }
        path.push_back(it->second.road_id);
        curr = it->second.from_node;
    }
    std::reverse(path.begin(), path.end());

    return { std::move(path), dist[destination], nodes_visited, true };
}

} // namespace syntraq
