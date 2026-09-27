//
// src/vehicles/vehicle_movement_system.cpp
//

#include "syntraq/vehicles/vehicle_movement_system.h"

#include <algorithm>
#include <cmath>

namespace syntraq {

// ── Public ────────────────────────────────────────────────────────────────────

void VehicleMovementSystem::update(std::vector<Vehicle>& vehicles,
                                   const RoadNetwork&    network,
                                   float                 dt) {
    for (auto& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;
        update_vehicle(v, network, dt);
    }
}

// ── Private ───────────────────────────────────────────────────────────────────

bool VehicleMovementSystem::update_vehicle(Vehicle&           v,
                                           const RoadNetwork& network,
                                           float              dt) {
    if (v.state != VehicleState::Moving) return false;

    const Road* road = network.road(v.current_road);
    if (!road) {
        v.state = VehicleState::Arrived; // orphaned vehicle
        return true;
    }

    // ── 1. Accelerate toward road speed limit ─────────────────────────────
    const float target_speed = std::min(v.max_speed_mps, road->speed_limit_mps);

    if (v.speed_mps < target_speed) {
        v.speed_mps = std::min(target_speed,
                               v.speed_mps + v.accel_mps2 * dt);
    } else if (v.speed_mps > target_speed) {
        v.speed_mps = std::max(target_speed,
                               v.speed_mps - v.decel_mps2 * dt);
    }

    // ── 2. Advance position ───────────────────────────────────────────────
    const float step = v.speed_mps * dt;
    v.progress_m    += step;
    v.distance_m    += step;
    v.travel_time_s += dt;

    // ── 3. Consume road-end transitions in a loop ─────────────────────────
    // A single large dt can overshoot multiple short roads in one tick;
    // loop until progress is within the current road or route is exhausted.
    while (v.state == VehicleState::Moving) {
        const Road* cur = network.road(v.current_road);
        if (!cur || v.progress_m < cur->length_m) break;

        const float overshoot = v.progress_m - cur->length_m;

        if (!advance_road(v, network)) {
            // Route exhausted → arrived
            v.state      = VehicleState::Arrived;
            v.speed_mps  = 0.f;
            v.progress_m = 0.f;
            return true;
        }

        // Carry overshoot onto the new road
        v.progress_m = overshoot;
    }

    return v.state == VehicleState::Arrived;
}

bool VehicleMovementSystem::advance_road(Vehicle&           v,
                                         const RoadNetwork& network) {
    v.route_index += 1;

    if (v.route_index >= static_cast<uint32_t>(v.route.size())) {
        return false; // No more roads in route
    }

    v.current_road = v.route[v.route_index];
    v.progress_m   = 0.f;

    // Keep same lane index, clamped to new road's lane count
    const Road* next = network.road(v.current_road);
    if (next && v.lane_index >= next->lane_count()) {
        v.lane_index = 0;
    }

    return true;
}

} // namespace syntraq
