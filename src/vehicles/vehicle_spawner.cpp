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

// ── try_spawn ─────────────────────────────────────────────────────────────────

std::optional<Vehicle> VehicleSpawner::try_spawn(const RoadNetwork& network,
                                                  uint32_t           active_count,
                                                  float              sim_elapsed_s) {
    // Rate limit
    if (sim_elapsed_s - last_spawn_s_ < spawn_interval_s) return std::nullopt;
    // Cap
    if (active_count >= max_vehicles) return std::nullopt;
    // Need at least 2 intersections for a meaningful route
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

    // Build vehicle
    Vehicle v;
    v.id          = static_cast<VehicleId>(next_vehicle_id_++);
    v.route       = std::move(route);
    v.route_index = 0;
    v.destination = dst;
    v.current_road = v.route[0];
    v.progress_m  = 0.f;
    v.speed_mps   = 0.f;
    v.state       = VehicleState::Moving;

    // Set max_speed from the first road's speed limit
    const Road* first_road = network.road(v.current_road);
    if (first_road) {
        v.max_speed_mps = first_road->speed_limit_mps;
    }

    // Choose a random lane
    if (first_road && first_road->lane_count() > 0) {
        std::uniform_int_distribution<uint32_t> lane_dist(
            0, first_road->lane_count() - 1);
        v.lane_index = lane_dist(rng_);
    }

    last_spawn_s_ = sim_elapsed_s;
    return v;
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
