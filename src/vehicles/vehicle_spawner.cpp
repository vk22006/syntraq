//
// src/vehicles/vehicle_spawner.cpp
//

#include "syntraq/vehicles/vehicle_spawner.h"
#include "syntraq/routing/astar_router.h"

#include <algorithm>

namespace syntraq {

VehicleSpawner::VehicleSpawner(uint32_t                              rng_seed,
                               float                                 spawn_interval_s_,
                               uint32_t                              max_vehicles_,
                               std::shared_ptr<const IRouteProvider> route_provider)
    : spawn_interval_s(spawn_interval_s_)
    , max_vehicles    (max_vehicles_)
    , rng_            (rng_seed)
    , route_provider_ (route_provider ? std::move(route_provider) : std::make_shared<AStarRouter>())
{
}

void VehicleSpawner::set_seed(uint32_t seed) {
    rng_.seed(seed);
    last_spawn_s_ = -999.f;
}

// ── try_spawn ─────────────────────────────────────────────────────────────────

std::optional<Vehicle> VehicleSpawner::try_spawn(const RoadNetwork&          network,
                                                 const std::vector<Vehicle>& existing_vehicles,
                                                 float                       sim_elapsed_s) {
    // Rate limit
    if (sim_elapsed_s - last_spawn_s_ < spawn_interval_s) return std::nullopt;

    // Count active vehicles
    uint32_t active_count = 0;
    for (const auto& v : existing_vehicles) {
        if (v.state != VehicleState::Arrived) {
            ++active_count;
        }
    }
    if (active_count >= max_vehicles) return std::nullopt;
    if (network.intersection_count() < 2) return std::nullopt;

    // Pick source and destination — retry a few times to avoid src==dst
    IntersectionId src = kInvalidIntersectionId;
    IntersectionId dst = kInvalidIntersectionId;
    for (int attempt = 0; attempt < 10; ++attempt) {
        src = random_intersection(network);
        dst = random_intersection(network);
        if (src != dst) break;
    }
    if (src == dst) return std::nullopt;

    auto route = build_route(network, src, dst);
    if (route.empty()) return std::nullopt;

    const Road* first_road = network.road(route[0]);
    if (!first_road || first_road->lane_count() == 0) return std::nullopt;

    // ── Lane Occupancy Check at Spawn Point ──────────────────────────────────
    // Determine which lane on first_road is free within the spawn buffer
    constexpr float kSpawnClearanceBufferM = 10.0f; // 4.5m car + 2.5m gap + margin
    std::vector<uint32_t> available_lanes;
    for (uint32_t l = 0; l < first_road->lane_count(); ++l) {
        bool lane_occupied = false;
        for (const auto& v : existing_vehicles) {
            if (v.state == VehicleState::Arrived) continue;
            if (v.current_road == route[0] && v.lane_index == l && v.progress_m < kSpawnClearanceBufferM) {
                lane_occupied = true;
                break;
            }
        }
        if (!lane_occupied) {
            available_lanes.push_back(l);
        }
    }

    if (available_lanes.empty()) {
        // All entry lanes currently occupied; defer spawn to avoid overlap
        return std::nullopt;
    }

    // Pick random available lane
    std::uniform_int_distribution<size_t> lane_pick(0, available_lanes.size() - 1);
    const uint32_t chosen_lane = available_lanes[lane_pick(rng_)];

    // Build vehicle
    Vehicle v;
    v.id            = static_cast<VehicleId>(next_vehicle_id_++);
    v.route         = std::move(route);
    v.route_index   = 0;
    v.destination   = dst;
    v.current_road  = v.route[0];
    v.lane_index    = chosen_lane;
    v.progress_m    = 0.f;
    v.speed_mps     = 0.f;
    v.max_speed_mps = first_road->speed_limit_mps;
    v.state         = VehicleState::Moving;

    last_spawn_s_ = sim_elapsed_s;
    return v;
}

std::optional<Vehicle> VehicleSpawner::try_spawn(const RoadNetwork& network,
                                                 uint32_t           active_count,
                                                 float              sim_elapsed_s) {
    // Construct dummy vehicle list of size active_count with position far away
    // to preserve exact signature and behavior for standalone tests
    std::vector<Vehicle> dummy_vehicles(active_count);
    for (auto& dv : dummy_vehicles) {
        dv.state      = VehicleState::Moving;
        dv.progress_m = 999.0f; // Won't trigger spawn clearance conflict
    }
    return try_spawn(network, dummy_vehicles, sim_elapsed_s);
}

// ── build_route ───────────────────────────────────────────────────────────────

std::vector<RoadId> VehicleSpawner::build_route(const RoadNetwork& network,
                                                IntersectionId     source,
                                                IntersectionId     destination) const {
    return route_provider_->find_route(network, source, destination);
}

// ── private helpers ───────────────────────────────────────────────────────────

IntersectionId VehicleSpawner::random_intersection(const RoadNetwork& network) {
    const auto& intersections = network.intersections();
    if (intersections.empty()) return kInvalidIntersectionId;

    std::uniform_int_distribution<size_t> dist(0, intersections.size() - 1);
    auto it = intersections.begin();
    std::advance(it, dist(rng_));
    return it->first;
}

} // namespace syntraq
