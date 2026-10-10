//
// src/routing/astar_router.cpp
//

#include "syntraq/routing/astar_router.h"

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace syntraq {

namespace {

struct PQNode {
    float          f_score;
    float          g_score;
    IntersectionId id;

    bool operator>(const PQNode& other) const noexcept {
        if (f_score != other.f_score) return f_score > other.f_score;
        return id > other.id;
    }
};

struct Predecessor {
    IntersectionId from_node{ kInvalidIntersectionId };
    RoadId         road_id  { kInvalidRoadId };
};

} // namespace

AStarRouter::AStarRouter(RoadCostFunction cost_fn, HeuristicFunction heuristic_fn)
    : cost_fn_     (std::move(cost_fn))
    , heuristic_fn_(std::move(heuristic_fn)) {}

RouteResult AStarRouter::find_route_with_info(
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
    std::unordered_map<IntersectionId, float>       g_score;
    std::unordered_map<IntersectionId, Predecessor> came_from;

    g_score[source] = 0.0f;
    float h0 = heuristic_fn_ ? heuristic_fn_(network, source, destination) : 0.0f;
    if (h0 < 0.0f) h0 = 0.0f;
    open_set.push({ h0, 0.0f, source });

    uint32_t nodes_visited = 0;
    bool reached = false;

    while (!open_set.empty()) {
        auto [f, g, u] = open_set.top();
        open_set.pop();

        auto g_it = g_score.find(u);
        if (g_it != g_score.end() && g > g_it->second) {
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

            float tentative_g = g + edge_cost;
            auto v_it = g_score.find(v);
            if (v_it == g_score.end() || tentative_g < v_it->second) {
                g_score[v] = tentative_g;
                came_from[v] = Predecessor{ u, rid };
                float h = heuristic_fn_ ? heuristic_fn_(network, v, destination) : 0.0f;
                if (h < 0.0f) h = 0.0f;
                open_set.push({ tentative_g + h, tentative_g, v });
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

    return { std::move(path), g_score[destination], nodes_visited, true };
}

} // namespace syntraq
