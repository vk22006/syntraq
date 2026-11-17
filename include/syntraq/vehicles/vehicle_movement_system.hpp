#pragma once

//
// syntraq/vehicles/vehicle_movement_system.hpp
//
// VehicleMovementSystem — advances every vehicle by dt seconds using a
// topology-aligned spatial lane-bucket index for O(1) car-following and
// intersection entrance clearance checks with zero per-tick heap allocations.
//

#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/world/road_network.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace syntraq {

class VehicleMovementSystem {
public:
    VehicleMovementSystem() = default;

    /// Update all vehicles by one timestep `dt` seconds (unsignalized baseline).
    void update(std::vector<Vehicle>& vehicles,
                const RoadNetwork&    network,
                float                 dt);

    /// Update all vehicles by one timestep `dt` seconds, respecting intersection traffic signals.
    void update(std::vector<Vehicle>&                                              vehicles,
                const RoadNetwork&                                                 network,
                const std::unordered_map<IntersectionId, TrafficSignalController>& signal_controllers,
                float                                                              dt);

private:
    /// Calculate safe turn speed limit based on geometric angle between incoming and outgoing roads.
    [[nodiscard]] float calculate_turn_speed(const Vehicle&     v,
                                             const Road&        cur_road,
                                             const RoadNetwork& network) const;

    /// O(1) spatial lane-bucket check for whether target lane entrance on next road is occupied.
    [[nodiscard]] bool is_lane_entrance_occupied(RoadId                      target_road_id,
                                                 uint32_t                    target_lane,
                                                 float                       clearance_needed_m,
                                                 const std::vector<Vehicle>& vehicles,
                                                 size_t                      self_idx) const;

    /// Attempt to transition vehicle onto the next road in its route.
    /// Returns true if successful, false if route is finished.
    bool advance_road(Vehicle& v, const RoadNetwork& network);

    /// Ensure flat spatial lane-bucket buffers cover [0, max_road_id) x [0, lane_stride_).
    void ensure_bucket_capacity(uint32_t max_road_id, uint32_t min_lane_stride);

    // Reusable topology-aligned spatial lane-bucket buffers (zero per-tick allocations)
    uint32_t                         lane_stride_{ 4 };
    uint32_t                         bucket_epoch_{ 0 };
    std::vector<std::vector<size_t>> lane_buckets_;
    std::vector<size_t>              active_buckets_;
    std::vector<uint32_t>            newly_entered_epoch_;
    std::vector<float>               newly_entered_min_progress_;
    std::vector<size_t>              newly_entered_owner_;
};

} // namespace syntraq
