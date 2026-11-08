//
// src/signals/network_coordination_manager.cpp
//

#include "syntraq/signals/network_coordination_manager.hpp"
#include "syntraq/signals/coordinated_controller.hpp"

#include <algorithm>

namespace syntraq {

void NetworkCoordinationManager::build_topology(
    const RoadNetwork& network,
    const std::unordered_map<IntersectionId, TrafficSignalController>& controllers)
{
    links_.clear();

    for (const auto& [to_id, _] : controllers) {
        const auto incoming = network.incoming_roads(to_id);
        for (RoadId r_id : incoming) {
            const Road* r = network.road(r_id);
            if (!r) continue;

            const IntersectionId from_id = r->from;
            if (controllers.find(from_id) != controllers.end()) {
                CoordinatedLink link;
                link.from_intersection = from_id;
                link.to_intersection   = to_id;
                link.road_id           = r_id;
                link.length_m          = r->length_m;
                link.speed_limit_mps   = r->speed_limit_mps;
                link.lane_count        = r->lane_count();
                links_.push_back(link);
            }
        }
    }
}

void NetworkCoordinationManager::exchange_coordination_states(
    const RoadNetwork& network,
    std::unordered_map<IntersectionId, TrafficSignalController>& controllers,
    const std::unordered_map<RoadId, uint32_t>& road_queues,
    const std::unordered_map<RoadId, float>&    road_max_waits)
{
    (void)network;
    (void)road_max_waits;

    for (const auto& link : links_) {
        auto up_it = controllers.find(link.from_intersection);
        auto down_it = controllers.find(link.to_intersection);
        if (up_it == controllers.end() || down_it == controllers.end()) continue;

        // Current queue and capacity of the connecting road link
        uint32_t queue_depth = 0;
        auto q_it = road_queues.find(link.road_id);
        if (q_it != road_queues.end()) queue_depth = q_it->second;

        // Effective storage capacity (~7.5m per passenger car)
        const float est_capacity = std::max(1.0f,
            (link.length_m * static_cast<float>(std::max(1u, link.lane_count))) / 7.5f);
        const float occupancy_ratio = std::clamp(static_cast<float>(queue_depth) / est_capacity, 0.0f, 1.0f);

        // ── 1. Downstream feedback sent to Upstream Controller (Spillback Gating)
        if (auto* coord_up = up_it->second.as<CoordinatedController>()) {
            NeighborIntersectionState down_state;
            down_state.neighbor_id                = link.to_intersection;
            down_state.connecting_road            = link.road_id;
            down_state.is_upstream                = false;
            down_state.link_length_m              = link.length_m;
            down_state.downstream_queue           = queue_depth;
            down_state.downstream_occupancy_ratio = occupancy_ratio;
            down_state.spillback_imminent         = (occupancy_ratio >= 0.60f || queue_depth >= 5);

            coord_up->receive_neighbor_state(down_state);
        }

        // ── 2. Upstream feedforward sent to Downstream Controller (Green-Wave Progression)
        if (auto* coord_down = down_it->second.as<CoordinatedController>()) {
            NeighborIntersectionState up_state;
            up_state.neighbor_id             = link.from_intersection;
            up_state.connecting_road         = link.road_id;
            up_state.is_upstream             = true;
            up_state.link_length_m           = link.length_m;
            up_state.upstream_phase          = up_it->second.current_phase_index();
            up_state.upstream_stage          = up_it->second.current_stage();
            up_state.platoon_dispatched      = (up_it->second.current_stage() == PhaseStage::Green);
            up_state.estimated_platoon_eta_s = link.speed_limit_mps > 0.1f
                                            ? (link.length_m / link.speed_limit_mps)
                                            : 10.0f;
            up_state.estimated_platoon_size  = std::max(1u, queue_depth);

            coord_down->receive_neighbor_state(up_state);
        }
    }
}

} // namespace syntraq

