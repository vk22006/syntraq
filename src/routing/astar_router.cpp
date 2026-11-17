//
// src/routing/astar_router.cpp
//

#include "syntraq/routing/astar_router.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

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

/// Thread-local reusable A* search workspace (O(1) epoch reset, zero per-query allocations).
struct AStarWorkspace {
    std::vector<float>       g_score;
    std::vector<Predecessor> came_from;
    std::vector<uint32_t>    epoch;
    std::vector<PQNode>      open_heap;
    uint32_t                 current_epoch{ 0 };

    void prepare(size_t max_nodes) {
        if (g_score.size() < max_nodes) {
            g_score.resize(max_nodes, 0.0f);
            came_from.resize(max_nodes);
            epoch.resize(max_nodes, 0u);
        }
        ++current_epoch;
        if (current_epoch == 0) {
            std::fill(epoch.begin(), epoch.end(), 0u);
            current_epoch = 1;
        }
        open_heap.clear();
    }
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

    const size_t max_nodes = static_cast<size_t>(network.max_intersection_id());
    const auto src_idx = static_cast<size_t>(source);
    const auto dst_idx = static_cast<size_t>(destination);
    if (src_idx >= max_nodes || dst_idx >= max_nodes) {
        return { {}, 0.0f, 0, false };
    }

    thread_local AStarWorkspace ws;
    ws.prepare(max_nodes);
    const uint32_t cur_epoch = ws.current_epoch;
    auto& g_score   = ws.g_score;
    auto& came_from = ws.came_from;
    auto& epoch     = ws.epoch;
    auto& open_heap = ws.open_heap;
    constexpr std::greater<PQNode> heap_cmp{};

    epoch[src_idx]   = cur_epoch;
    g_score[src_idx] = 0.0f;

    float h0 = heuristic_fn_ ? heuristic_fn_(network, source, destination) : 0.0f;
    if (h0 < 0.0f) h0 = 0.0f;
    open_heap.push_back({ h0, 0.0f, source });
    std::push_heap(open_heap.begin(), open_heap.end(), heap_cmp);

    uint32_t nodes_visited = 0;
    bool reached = false;

    while (!open_heap.empty()) {
        std::pop_heap(open_heap.begin(), open_heap.end(), heap_cmp);
        const PQNode top = open_heap.back();
        open_heap.pop_back();

        const float g = top.g_score;
        const IntersectionId u = top.id;
        const auto u_idx = static_cast<size_t>(u);

        if (epoch[u_idx] == cur_epoch && g > g_score[u_idx]) {
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

            const IntersectionId v = r->to;
            const auto v_idx = static_cast<size_t>(v);
            if (v_idx >= max_nodes) continue;

            float edge_cost = cost_fn_ ? cost_fn_(*r) : r->length_m;
            if (edge_cost < 0.0f) edge_cost = 0.0f;

            const float tentative_g = g + edge_cost;
            if (epoch[v_idx] != cur_epoch || tentative_g < g_score[v_idx]) {
                epoch[v_idx]     = cur_epoch;
                g_score[v_idx]   = tentative_g;
                came_from[v_idx] = Predecessor{ u, rid };
                float h = heuristic_fn_ ? heuristic_fn_(network, v, destination) : 0.0f;
                if (h < 0.0f) h = 0.0f;
                open_heap.push_back({ tentative_g + h, tentative_g, v });
                std::push_heap(open_heap.begin(), open_heap.end(), heap_cmp);
            }
        }
    }

    if (!reached) {
        return { {}, 0.0f, nodes_visited, false };
    }

    std::vector<RoadId> path;
    IntersectionId curr = destination;
    while (curr != source) {
        const auto curr_idx = static_cast<size_t>(curr);
        if (curr_idx >= max_nodes || epoch[curr_idx] != cur_epoch) {
            return { {}, 0.0f, nodes_visited, false };
        }
        path.push_back(came_from[curr_idx].road_id);
        curr = came_from[curr_idx].from_node;
    }
    std::reverse(path.begin(), path.end());

    return { std::move(path), g_score[dst_idx], nodes_visited, true };
}

} // namespace syntraq
