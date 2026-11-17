//
// src/routing/dijkstra_router.cpp
//

#include "syntraq/routing/dijkstra_router.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

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

/// Thread-local reusable Dijkstra workspace (O(1) epoch reset, zero per-query allocations).
struct DijkstraWorkspace {
    std::vector<float>       dist;
    std::vector<Predecessor> came_from;
    std::vector<uint32_t>    epoch;
    std::vector<PQNode>      open_heap;
    uint32_t                 current_epoch{ 0 };

    void prepare(size_t max_nodes) {
        if (dist.size() < max_nodes) {
            dist.resize(max_nodes, 0.0f);
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

    const size_t max_nodes = static_cast<size_t>(network.max_intersection_id());
    const auto src_idx = static_cast<size_t>(source);
    const auto dst_idx = static_cast<size_t>(destination);
    if (src_idx >= max_nodes || dst_idx >= max_nodes) {
        return { {}, 0.0f, 0, false };
    }

    thread_local DijkstraWorkspace ws;
    ws.prepare(max_nodes);
    const uint32_t cur_epoch = ws.current_epoch;
    auto& dist      = ws.dist;
    auto& came_from = ws.came_from;
    auto& epoch     = ws.epoch;
    auto& open_heap = ws.open_heap;
    constexpr std::greater<PQNode> heap_cmp{};

    epoch[src_idx] = cur_epoch;
    dist[src_idx]  = 0.0f;
    open_heap.push_back({ 0.0f, source });
    std::push_heap(open_heap.begin(), open_heap.end(), heap_cmp);

    uint32_t nodes_visited = 0;
    bool reached = false;

    while (!open_heap.empty()) {
        std::pop_heap(open_heap.begin(), open_heap.end(), heap_cmp);
        const PQNode top = open_heap.back();
        open_heap.pop_back();

        const float current_cost = top.cost;
        const IntersectionId u   = top.id;
        const auto u_idx = static_cast<size_t>(u);

        if (epoch[u_idx] == cur_epoch && current_cost > dist[u_idx]) {
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

            const float new_cost = current_cost + edge_cost;
            if (epoch[v_idx] != cur_epoch || new_cost < dist[v_idx]) {
                epoch[v_idx]     = cur_epoch;
                dist[v_idx]      = new_cost;
                came_from[v_idx] = Predecessor{ u, rid };
                open_heap.push_back({ new_cost, v });
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

    return { std::move(path), dist[dst_idx], nodes_visited, true };
}

} // namespace syntraq
