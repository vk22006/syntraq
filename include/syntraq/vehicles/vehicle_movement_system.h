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

#include "syntraq/vehicles/vehicle.h"
#include "syntraq/world/road_network.h"
#include <vector>

namespace syntraq {

class VehicleMovementSystem {
public:
    VehicleMovementSystem() = default;

    /// Update all vehicles by one timestep `dt` seconds.
    /// Vehicles marked Arrived are NOT removed here — the caller
    /// (Simulation) is responsible for despawning them.
    void update(std::vector<Vehicle>& vehicles,
                const RoadNetwork&    network,
                float                 dt);

private:
    /// Advance a single vehicle. Returns true if the vehicle arrived.
    bool update_vehicle(Vehicle&           v,
                        const RoadNetwork& network,
                        float              dt);

    /// Attempt to transition vehicle onto the next road in its route.
    /// Returns true if successful, false if route is finished.
    bool advance_road(Vehicle& v, const RoadNetwork& network);
};

} // namespace syntraq
