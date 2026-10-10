#pragma once

//
// syntraq/vehicles/vehicle_movement_system.h
//
// VehicleMovementSystem — advances every vehicle by dt seconds.
//
// Movement model (intentionally simple for M3):
//   1. Accelerate toward the road's speed limit (capped at vehicle max_speed).
//   2. Advance progress_m by speed * dt.
//   3. When progress_m >= road.length_m: transition to the next road in route.
//   4. When the route is exhausted: mark vehicle as Arrived.
//
// No collision detection, no car-following, no signal awareness yet.
// Those add in future milestones.
//

#include "syntraq/signals/traffic_signal_controller.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/world/road_network.h"
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

    /// Check if target lane entrance on next road is occupied by another vehicle.
    [[nodiscard]] bool is_lane_entrance_occupied(RoadId                      target_road_id,
                                                 uint32_t                    target_lane,
                                                 float                       clearance_needed_m,
                                                 const std::vector<Vehicle>& vehicles,
                                                 size_t                      self_idx) const;

    /// Attempt to transition vehicle onto the next road in its route.
    /// Returns true if successful, false if route is finished.
    bool advance_road(Vehicle& v, const RoadNetwork& network);
};

} // namespace syntraq
