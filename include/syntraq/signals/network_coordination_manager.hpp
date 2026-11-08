#pragma once

//
// syntraq/signals/network_coordination_manager.hpp
//
// NetworkCoordinationManager — Coordinates information exchange across neighboring intersections.
// Discovers network adjacency between adjacent traffic signals and periodically dispatches:
//   - Downstream queue & occupancy feedback to upstream controllers (spillback mitigation)
//   - Upstream phase & platoon release timings to downstream controllers (green-wave progression)
//

#include "syntraq/signals/coordination_types.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/world/ids.hpp"
#include "syntraq/world/road_network.hpp"

#include <unordered_map>
#include <vector>

namespace syntraq {

struct CoordinatedLink {
    IntersectionId from_intersection{ kInvalidIntersectionId }; ///< Upstream junction
    IntersectionId to_intersection  { kInvalidIntersectionId }; ///< Downstream junction
    RoadId         road_id          { kInvalidRoadId };         ///< Connecting road edge
    float          length_m         { 0.0f };
    float          speed_limit_mps  { 13.89f };
    uint32_t       lane_count       { 1 };
};

class NetworkCoordinationManager {
public:
    NetworkCoordinationManager() = default;

    /// Discover and establish neighbor linkages between controlled intersections
    void build_topology(
        const RoadNetwork& network,
        const std::unordered_map<IntersectionId, TrafficSignalController>& controllers);

    /// Exchange real-time local states and neighbor status among adjacent controllers
    void exchange_coordination_states(
        const RoadNetwork& network,
        std::unordered_map<IntersectionId, TrafficSignalController>& controllers,
        const std::unordered_map<RoadId, uint32_t>& road_queues,
        const std::unordered_map<RoadId, float>&    road_max_waits);

    [[nodiscard]] const std::vector<CoordinatedLink>& links() const noexcept {
        return links_;
    }

    [[nodiscard]] size_t link_count() const noexcept {
        return links_.size();
    }

    void clear() noexcept {
        links_.clear();
    }

private:
    std::vector<CoordinatedLink> links_{};
};

} // namespace syntraq

