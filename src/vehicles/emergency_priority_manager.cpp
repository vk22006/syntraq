//
// src/vehicles/emergency_priority_manager.cpp
//
// EmergencyPriorityManager implementation.
//

#include "syntraq/vehicles/emergency_priority_manager.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"

#include <algorithm>

namespace syntraq {

void EmergencyPriorityManager::update(
    const RoadNetwork& network,
    std::vector<Vehicle>& vehicles,
    std::unordered_map<IntersectionId, TrafficSignalController>& controllers,
    TrafficAwareRouting* router,
    float dt)
{
    (void)dt;
    emergency_indices_.clear();

    // 1. Identify active emergency vehicles in a single linear pass
    for (size_t i = 0; i < vehicles.size(); ++i) {
        const auto& v = vehicles[i];
        if (v.is_emergency() && v.state != VehicleState::Arrived && v.current_road != kInvalidRoadId) {
            emergency_indices_.push_back(i);
        }
    }

    // Fast path: no active emergency vehicles and nothing to release
    if (emergency_indices_.empty() && previously_preempted_.empty() && !status_.active) {
        status_.active_emergency_count = 0;
        return;
    }

    current_corridors_.clear();
    preemption_requests_.clear();
    const uint32_t active_emergencies = static_cast<uint32_t>(emergency_indices_.size());

    for (size_t idx : emergency_indices_) {
        const auto& v = vehicles[idx];
        current_corridors_.insert(v.current_road);

        const Road* current_road = network.road(v.current_road);
        if (!current_road) continue;

        const float dist_to_inter = std::max(0.0f, current_road->length_m - v.progress_m);

        // Preempt approaching intersection if within detection range
        if (dist_to_inter <= preemption_distance_m) {
            if (enable_preemption) {
                preemption_requests_[current_road->to] = v.current_road;
            }

            // Extend priority corridor to subsequent roads in route
            if (v.route_index + 1 < v.route.size()) {
                const RoadId next_road_id = v.route[v.route_index + 1];
                current_corridors_.insert(next_road_id);

                // If approaching close, prepare the subsequent intersection along green corridor
                const Road* next_road = network.road(next_road_id);
                if (next_road && (dist_to_inter + next_road->length_m <= preemption_distance_m)) {
                    if (enable_preemption) {
                        preemption_requests_[next_road->to] = next_road_id;
                    }
                    if (v.route_index + 2 < v.route.size()) {
                        current_corridors_.insert(v.route[v.route_index + 2]);
                    }
                }
            }
        }
    }

    // 2. Normal Vehicle Lane Yielding (single pass over vehicles when emergencies exist)
    if (enable_yielding && !emergency_indices_.empty()) {
        for (auto& v_norm : vehicles) {
            if (v_norm.is_emergency() || v_norm.state == VehicleState::Arrived) continue;
            for (size_t e_idx : emergency_indices_) {
                const auto& v_emerg = vehicles[e_idx];
                if (v_norm.current_road != v_emerg.current_road) continue;
                if (v_norm.lane_index != v_emerg.lane_index) continue;

                const float gap = v_norm.progress_m - v_emerg.progress_m;
                if (gap > 0.0f && gap < 60.0f) {
                    const Road* road = network.road(v_emerg.current_road);
                    if (road && road->lane_count() > 1) {
                        v_norm.lane_index = (v_norm.lane_index + 1) % road->lane_count();
                    }
                    break;
                }
            }
        }
    }

    // 3. Dispatch Signal Preemption Requests
    for (const auto& [inter_id, road_id] : preemption_requests_) {
        auto it = controllers.find(inter_id);
        if (it != controllers.end()) {
            it->second.request_emergency_preemption(road_id);
        }
    }

    // 4. Release Preemptions for Intersections No Longer on Active Path
    for (IntersectionId inter_id : previously_preempted_) {
        if (preemption_requests_.find(inter_id) == preemption_requests_.end()) {
            auto it = controllers.find(inter_id);
            if (it != controllers.end()) {
                it->second.clear_emergency_preemption();
            }
        }
    }

    // Update set of currently preempted intersections
    previously_preempted_.clear();
    for (const auto& [inter_id, _] : preemption_requests_) {
        previously_preempted_.insert(inter_id);
    }

    // 5. Update Dynamic Routing Engine with Active Corridor Roads
    if (router) {
        if (enable_rerouting && !current_corridors_.empty()) {
            router->set_emergency_corridor_roads(current_corridors_);
        } else {
            router->clear_emergency_corridor_roads();
        }
    }

    // 6. Refresh Status Telemetry
    status_.active                  = !current_corridors_.empty();
    status_.active_emergency_count  = active_emergencies;
    status_.corridor_roads          = current_corridors_;
    status_.preempted_intersections = previously_preempted_;
}

void EmergencyPriorityManager::record_vehicle_arrival(const Vehicle& vehicle) {
    if (!vehicle.is_emergency()) return;

    status_.completed_trips++;
    status_.total_travel_time_s += vehicle.travel_time_s;
    status_.total_wait_time_s   += vehicle.wait_time_s;

    status_.avg_travel_time_s = status_.total_travel_time_s / static_cast<float>(status_.completed_trips);
    status_.avg_wait_time_s   = status_.total_wait_time_s / static_cast<float>(status_.completed_trips);

    if (vehicle.travel_time_s > 0.001f && vehicle.distance_m > 0.0f) {
        const float trip_speed = vehicle.distance_m / vehicle.travel_time_s;
        if (status_.completed_trips == 1) {
            status_.avg_speed_mps = trip_speed;
        } else {
            status_.avg_speed_mps = (status_.avg_speed_mps * static_cast<float>(status_.completed_trips - 1) + trip_speed)
                                  / static_cast<float>(status_.completed_trips);
        }
    }
}

void EmergencyPriorityManager::reset() {
    previously_preempted_.clear();
    status_ = EmergencyCorridorStatus{};
}

} // namespace syntraq

