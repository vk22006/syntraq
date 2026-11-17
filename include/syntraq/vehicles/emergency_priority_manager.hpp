#pragma once

//
// syntraq/vehicles/emergency_priority_manager.hpp
//
// EmergencyPriorityManager — Subsystem responsible for managing emergency priority
// corridors, traffic signal preemption, normal vehicle lane yielding, and dynamic
// detour rerouting across the road network without deadlocking background traffic.
//

#include "syntraq/world/ids.hpp"
#include "syntraq/world/road_network.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace syntraq {

class TrafficAwareRouting;

/// Live status and aggregated metrics for the emergency corridor system
struct EmergencyCorridorStatus {
    bool                               active{ false };
    uint32_t                           active_emergency_count{ 0 };
    std::unordered_set<RoadId>         corridor_roads{};
    std::unordered_set<IntersectionId> preempted_intersections{};
    uint32_t                           completed_trips{ 0 };
    float                              total_travel_time_s{ 0.0f };
    float                              total_wait_time_s{ 0.0f };
    float                              avg_travel_time_s{ 0.0f };
    float                              avg_wait_time_s{ 0.0f };
    float                              avg_speed_mps{ 0.0f };
};

class EmergencyPriorityManager {
public:
    EmergencyPriorityManager() = default;

    /// Update priority corridor state, signal preemption, lane yielding, and rerouting
    void update(
        const RoadNetwork& network,
        std::vector<Vehicle>& vehicles,
        std::unordered_map<IntersectionId, TrafficSignalController>& controllers,
        TrafficAwareRouting* router,
        float dt);

    /// Record completed emergency vehicle arrival
    void record_vehicle_arrival(const Vehicle& vehicle);

    /// Reset all active corridor states and preemption requests
    void reset();

    // ── Configuration ──────────────────────────────────────────────────────
    bool  enable_preemption{ true };
    bool  enable_rerouting { true };
    bool  enable_yielding  { true };
    float preemption_distance_m{ 120.0f };

    // ── Inspection / Telemetry ─────────────────────────────────────────────
    [[nodiscard]] const EmergencyCorridorStatus& status() const noexcept {
        return status_;
    }

    [[nodiscard]] const std::unordered_set<RoadId>& active_corridor_roads() const noexcept {
        return status_.corridor_roads;
    }

    [[nodiscard]] const std::unordered_set<IntersectionId>& preempted_intersections() const noexcept {
        return status_.preempted_intersections;
    }

    [[nodiscard]] bool is_corridor_road(RoadId road_id) const noexcept {
        return status_.corridor_roads.find(road_id) != status_.corridor_roads.end();
    }

    [[nodiscard]] bool is_intersection_preempted(IntersectionId intersection_id) const noexcept {
        return status_.preempted_intersections.find(intersection_id) != status_.preempted_intersections.end();
    }

private:
    EmergencyCorridorStatus                    status_{};
    std::unordered_set<IntersectionId>         previously_preempted_{};

    // Reusable per-tick scratch buffers (zero heap allocations on hot path)
    std::vector<size_t>                        emergency_indices_{};
    std::unordered_set<RoadId>                 current_corridors_{};
    std::unordered_map<IntersectionId, RoadId> preemption_requests_{};
};

} // namespace syntraq

