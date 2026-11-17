//
// src/vehicles/vehicle_movement_system.cpp
//

#include "syntraq/vehicles/vehicle_movement_system.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace syntraq {

// ── Public ────────────────────────────────────────────────────────────────────

void VehicleMovementSystem::ensure_bucket_capacity(uint32_t max_road_id, uint32_t min_lane_stride) {
    bool stride_changed = false;
    if (min_lane_stride > lane_stride_) {
        lane_stride_ = min_lane_stride;
        stride_changed = true;
    }

    const size_t required = static_cast<size_t>(max_road_id) * static_cast<size_t>(lane_stride_);
    if (stride_changed) {
        // Clear any residual buckets before resizing with new stride
        for (size_t b : active_buckets_) {
            if (b < lane_buckets_.size()) {
                lane_buckets_[b].clear();
            }
        }
        active_buckets_.clear();
        std::fill(newly_entered_epoch_.begin(), newly_entered_epoch_.end(), 0u);
    }

    if (lane_buckets_.size() < required) {
        lane_buckets_.resize(required);
        newly_entered_epoch_.resize(required, 0u);
        newly_entered_min_progress_.resize(required, std::numeric_limits<float>::max());
        newly_entered_owner_.resize(required, std::numeric_limits<size_t>::max());
    }
}

void VehicleMovementSystem::update(std::vector<Vehicle>& vehicles,
                                   const RoadNetwork&    network,
                                   float                 dt) {
    update(vehicles, network, {}, dt);
}

void VehicleMovementSystem::update(
    std::vector<Vehicle>&                                              vehicles,
    const RoadNetwork&                                                 network,
    const std::unordered_map<IntersectionId, TrafficSignalController>& signal_controllers,
    float                                                              dt) {
    if (vehicles.empty() || dt <= 0.0f) return;

    const uint32_t max_road_id = network.max_road_id();
    if (max_road_id == 0) {
        for (auto& v : vehicles) {
            if (v.state != VehicleState::Arrived) {
                v.state = VehicleState::Arrived;
            }
        }
        return;
    }

    // Clear buckets used in the previous tick (preserves inner vector capacities — zero heap allocations)
    for (size_t b : active_buckets_) {
        if (b < lane_buckets_.size()) {
            lane_buckets_[b].clear();
        }
    }
    active_buckets_.clear();

    // Advance epoch stamp for O(1) reset of newly-entered lane tracking
    ++bucket_epoch_;
    if (bucket_epoch_ == 0) {
        std::fill(newly_entered_epoch_.begin(), newly_entered_epoch_.end(), 0u);
        bucket_epoch_ = 1;
    }

    // Determine required lane stride across active vehicles
    uint32_t max_lane_needed = lane_stride_;
    for (const auto& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;
        if (v.lane_index + 1 > max_lane_needed) {
            max_lane_needed = v.lane_index + 1;
        }
    }
    ensure_bucket_capacity(max_road_id, max_lane_needed);

    const auto* controllers_ptr = signal_controllers.empty() ? nullptr : &signal_controllers;

    // 1. Spatial lane-bucketing: group active vehicle indices by (RoadId, lane_index)
    for (size_t i = 0; i < vehicles.size(); ++i) {
        auto& v = vehicles[i];
        if (v.state == VehicleState::Arrived) continue;

        const auto rid_u32 = static_cast<uint32_t>(v.current_road);
        if (rid_u32 >= max_road_id || !network.road(v.current_road)) {
            v.state = VehicleState::Arrived;
            continue;
        }

        const size_t bucket_idx = static_cast<size_t>(rid_u32) * lane_stride_ + v.lane_index;
        auto& bucket = lane_buckets_[bucket_idx];
        if (bucket.empty()) {
            active_buckets_.push_back(bucket_idx);
        }
        bucket.push_back(i);
    }

    // Sort active bucket keys ascending so processing order matches std::map<(RoadId, lane)>
    std::sort(active_buckets_.begin(), active_buckets_.end());

    // 2. Process each active spatial lane bucket (leader to follower)
    for (size_t bucket_idx : active_buckets_) {
        auto& indices = lane_buckets_[bucket_idx];
        if (indices.size() > 1) {
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                if (std::abs(vehicles[a].progress_m - vehicles[b].progress_m) > 1e-4f) {
                    return vehicles[a].progress_m > vehicles[b].progress_m;
                }
                return vehicles[a].speed_mps > vehicles[b].speed_mps;
            });
        }

        // Hoist road & traffic signal lookup ONCE per lane bucket (all vehicles share current_road)
        const RoadId bucket_road_id = static_cast<RoadId>(bucket_idx / lane_stride_);
        const Road* road = network.road(bucket_road_id);
        if (!road) {
            for (size_t v_idx : indices) {
                vehicles[v_idx].state = VehicleState::Arrived;
            }
            continue;
        }

        constexpr float kStopLineOffset = 2.0f; // 2m before intersection boundary
        const float stop_line_m = std::max(0.0f, road->length_m - kStopLineOffset);

        bool signal_can_proceed = true;
        if (controllers_ptr) {
            const auto it = controllers_ptr->find(road->to);
            if (it != controllers_ptr->end()) {
                signal_can_proceed = it->second.can_proceed(bucket_road_id);
            }
        }

        // 3. Process vehicles from front to back on this lane
        for (size_t rank = 0; rank < indices.size(); ++rank) {
            const size_t v_idx = indices[rank];
            auto& v = vehicles[v_idx];

            const RoadId   orig_road = v.current_road;
            const uint32_t orig_lane = v.lane_index;

            const float dist_to_stop_line = stop_line_m - v.progress_m;
            bool can_proceed = signal_can_proceed;

            // ── Lane occupancy check for intersection entry (O(1) spatial bucket lookup) ──
            if (can_proceed && (v.route_index + 1 < v.route.size())) {
                const RoadId next_road_id = v.route[v.route_index + 1];
                const Road* next_road = network.road(next_road_id);
                uint32_t target_lane = v.lane_index;
                if (next_road && target_lane >= next_road->lane_count()) {
                    target_lane = 0;
                }

                // If approaching the intersection, ensure entrance to next road is clear
                if (dist_to_stop_line <= 15.0f) {
                    const float clearance_needed = v.length_m + v.min_gap_m;
                    if (v.is_emergency() && next_road && next_road->lane_count() > 1) {
                        for (uint32_t l = 0; l < next_road->lane_count(); ++l) {
                            if (!is_lane_entrance_occupied(next_road_id, l, clearance_needed, vehicles, v_idx)) {
                                target_lane = l;
                                v.lane_index = l;
                                break;
                            }
                        }
                    }
                    if (is_lane_entrance_occupied(next_road_id, target_lane, clearance_needed, vehicles, v_idx)) {
                        can_proceed = false; // Hold at stop line until entry clears
                    }
                }
            }

            // ── Calculate desired cruising / turning speed ────────────────
            float target_speed = v.is_emergency() ? v.max_speed_mps : std::min(v.max_speed_mps, road->speed_limit_mps);

            if (can_proceed && (v.route_index + 1 < v.route.size()) && dist_to_stop_line <= 25.0f) {
                const float turn_speed = calculate_turn_speed(v, *road, network);
                if (turn_speed < target_speed) {
                    // Deceleration profile for turn entry
                    const float allowed_turn_entry = std::sqrt(
                        turn_speed * turn_speed + 2.0f * v.decel_mps2 * std::max(0.1f, dist_to_stop_line));
                    target_speed = std::min(target_speed, allowed_turn_entry);
                }
            }

            // ── Identify governing obstacle (lead vehicle vs stop line) ───
            bool  has_obstacle       = false;
            float obstacle_gap       = 1e6f;
            float obstacle_speed     = 0.0f;
            float max_allowed_pos    = road->length_m + 100.0f;
            float standstill_buffer  = 0.0f;

            if (!can_proceed) {
                // Red light or blocked junction: stop line is an obstacle
                has_obstacle      = true;
                obstacle_gap      = std::max(0.0f, dist_to_stop_line);
                obstacle_speed    = 0.0f;
                max_allowed_pos   = stop_line_m;
                standstill_buffer = 0.0f; // Front bumper reaches stop line
            }

            if (rank > 0 && vehicles[indices[rank - 1]].current_road == v.current_road) {
                // There is a leading vehicle on the same road and lane
                const auto& lead_v = vehicles[indices[rank - 1]];
                const float lead_rear = lead_v.progress_m - lead_v.length_m;
                const float gap_to_lead = std::max(0.0f, lead_rear - v.progress_m);
                const float lead_max_pos = lead_rear - v.min_gap_m;

                if (!has_obstacle || lead_max_pos < max_allowed_pos) {
                    has_obstacle      = true;
                    obstacle_gap      = gap_to_lead;
                    obstacle_speed    = lead_v.speed_mps;
                    max_allowed_pos   = lead_max_pos;
                    standstill_buffer = v.min_gap_m; // Buffer behind lead vehicle
                }
            }

            // ── Car-Following & Braking Kinematics ─────────────────────────
            if (has_obstacle) {
                const float tau = v.time_headway_s;
                const float b   = std::max(0.1f, v.decel_mps2);
                const float net_gap = obstacle_gap - standstill_buffer;

                float v_safe = 0.0f;
                if (net_gap > 0.0f) {
                    const float b_tau = b * tau;
                    const float rad   = (b_tau * b_tau) + (obstacle_speed * obstacle_speed) + 2.0f * b * net_gap;
                    v_safe = -b_tau + std::sqrt(std::max(0.0f, rad));
                }
                target_speed = std::min(target_speed, v_safe);
            }

            // If stopped and acceleration is zero, vehicle must not move
            if (v.state == VehicleState::Stopped && v.accel_mps2 <= 0.0f) {
                v.speed_mps = 0.0f;
                v.travel_time_s += dt;
                v.wait_time_s   += dt;
                continue;
            }

            // Standstill snap when arriving at obstacle target
            if (has_obstacle && (max_allowed_pos - v.progress_m <= 0.2f) && v.speed_mps <= 0.5f) {
                v.progress_m = std::min(v.progress_m, max_allowed_pos);
                v.speed_mps  = 0.0f;
                v.state      = VehicleState::Stopped;
                v.travel_time_s += dt;
                v.wait_time_s   += dt;
                continue;
            }

            // Accelerate or decelerate toward target_speed
            if (v.speed_mps < target_speed) {
                v.speed_mps = std::min(target_speed, v.speed_mps + v.accel_mps2 * dt);
            } else if (v.speed_mps > target_speed) {
                v.speed_mps = std::max(target_speed, v.speed_mps - v.decel_mps2 * dt);
            }

            v.speed_mps = std::max(0.0f, v.speed_mps);

            // Step forward bounded by collision avoidance clamp
            float step = v.speed_mps * dt;
            if (has_obstacle) {
                const float allowed_step = std::max(0.0f, max_allowed_pos - v.progress_m);
                step = std::min(step, allowed_step);
            }

            v.progress_m    += step;
            v.distance_m    += step;
            v.travel_time_s += dt;
            if (v.speed_mps < 0.5f || v.state == VehicleState::Stopped) {
                v.wait_time_s += dt;
            }

            // Final standstill check
            if (has_obstacle && ((max_allowed_pos - v.progress_m <= 0.05f) || (step == 0.0f && v.speed_mps <= 0.1f))) {
                v.progress_m = std::min(v.progress_m, max_allowed_pos);
                v.speed_mps  = 0.0f;
                v.state      = VehicleState::Stopped;
            } else if (v.speed_mps > 0.05f) {
                v.state = VehicleState::Moving;
            }

            // ── Advance road transitions if vehicle reaches road end ──────
            const Road* cur = road;
            while (v.state == VehicleState::Moving) {
                if (!cur || v.progress_m < cur->length_m) break;

                const float overshoot = v.progress_m - cur->length_m;

                if (!advance_road(v, network)) {
                    v.state      = VehicleState::Arrived;
                    v.speed_mps  = 0.0f;
                    v.progress_m = 0.0f;
                    break;
                }

                v.progress_m = overshoot;
                cur = network.road(v.current_road);
                if (cur) {
                    v.max_speed_mps = cur->speed_limit_mps;
                    v.speed_mps     = std::min(v.speed_mps, cur->speed_limit_mps);
                }
            }

            // Record intra-tick road/lane transition for downstream O(1) entrance checks
            if (v.state != VehicleState::Arrived &&
                (v.current_road != orig_road || v.lane_index != orig_lane)) {
                const auto new_rid = static_cast<uint32_t>(v.current_road);
                if (new_rid < max_road_id && v.lane_index < lane_stride_) {
                    const size_t new_b = static_cast<size_t>(new_rid) * lane_stride_ + v.lane_index;
                    if (newly_entered_epoch_[new_b] != bucket_epoch_ ||
                        v.progress_m < newly_entered_min_progress_[new_b]) {
                        newly_entered_epoch_[new_b]        = bucket_epoch_;
                        newly_entered_min_progress_[new_b] = v.progress_m;
                        newly_entered_owner_[new_b]        = v_idx;
                    }
                }
            }
        }
    }
}

// ── Private Helpers ────────────────────────────────────────────────────────────

float VehicleMovementSystem::calculate_turn_speed(
    const Vehicle&     v,
    const Road&        cur_road,
    const RoadNetwork& network) const {
    if (v.route_index + 1 >= v.route.size()) {
        return cur_road.speed_limit_mps;
    }
    const RoadId next_road_id = v.route[v.route_index + 1];
    const Road* next_road     = network.road(next_road_id);
    if (!next_road) return cur_road.speed_limit_mps;

    const auto* from_node  = network.intersection(cur_road.from);
    const auto* inter_node = network.intersection(cur_road.to);
    const auto* to_node    = network.intersection(next_road->to);
    if (!from_node || !inter_node || !to_node) return cur_road.speed_limit_mps;

    const float dx_in  = inter_node->position.x - from_node->position.x;
    const float dy_in  = inter_node->position.y - from_node->position.y;
    const float len_in = std::sqrt(dx_in * dx_in + dy_in * dy_in);

    const float dx_out  = to_node->position.x - inter_node->position.x;
    const float dy_out  = to_node->position.y - inter_node->position.y;
    const float len_out = std::sqrt(dx_out * dx_out + dy_out * dy_out);

    if (len_in < 0.001f || len_out < 0.001f) {
        return next_road->speed_limit_mps;
    }

    // Dot product of normalized direction vectors: cos(theta)
    const float cos_angle = (dx_in * dx_out + dy_in * dy_out) / (len_in * len_out);

    if (cos_angle > 0.85f) {
        // Straight-through or gentle curvature (< 30 deg)
        return next_road->speed_limit_mps;
    } else if (cos_angle > 0.15f) {
        // Moderate turn (30 to ~80 deg)
        return std::min(next_road->speed_limit_mps, 7.0f); // ~25 km/h
    } else {
        // Sharp turn (approx 90 deg or sharper)
        return std::min(next_road->speed_limit_mps, 4.5f); // ~16 km/h
    }
}

bool VehicleMovementSystem::is_lane_entrance_occupied(
    RoadId                      target_road_id,
    uint32_t                    target_lane,
    float                       clearance_needed_m,
    const std::vector<Vehicle>& vehicles,
    size_t                      self_idx) const {
    if (target_lane >= lane_stride_) return false;
    const size_t bucket_idx = static_cast<size_t>(target_road_id) * lane_stride_ + target_lane;
    if (bucket_idx >= lane_buckets_.size()) return false;

    // 1. Check vehicles that transitioned into this lane earlier in the same tick (O(1))
    if (newly_entered_epoch_[bucket_idx] == bucket_epoch_ &&
        newly_entered_owner_[bucket_idx] != self_idx &&
        newly_entered_min_progress_[bucket_idx] < clearance_needed_m) {
        return true;
    }

    // 2. Check vehicles present in this spatial lane bucket at start of tick (O(k), k = lane density)
    const auto& bucket = lane_buckets_[bucket_idx];
    for (size_t i : bucket) {
        if (i == self_idx) continue;
        const auto& other = vehicles[i];
        if (other.state == VehicleState::Arrived) continue;
        if (other.current_road == target_road_id && other.lane_index == target_lane) {
            if (other.progress_m < clearance_needed_m) {
                return true;
            }
        }
    }
    return false;
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
