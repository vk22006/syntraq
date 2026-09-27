#pragma once

//
// syntraq/vehicles/vehicle_spawner.h
//
// VehicleSpawner — creates new vehicles on the road network.
//
// Route building strategy (M3):
//   A simple greedy walk: from the spawn intersection, repeatedly pick
//   the first outgoing road whose destination has not been visited,
//   until the destination intersection is reached or no progress is possible.
//
//   This is NOT pathfinding — it produces a valid (if suboptimal) route for
//   testing the movement system without implementing Dijkstra/A* yet.
//   Milestone 4 will replace this with proper shortest-path routing.
//
// Spawn policy:
//   - A vehicle is spawned at most once per `spawn_interval_s` seconds.
//   - Source and destination intersections are chosen by the caller,
//     or randomly using the provided RNG seed.
//

#include "syntraq/vehicles/vehicle.h"
#include "syntraq/world/road_network.h"

#include <optional>
#include <random>
#include <vector>

namespace syntraq {

class VehicleSpawner {
public:
    explicit VehicleSpawner(uint32_t rng_seed        = 42,
                            float    spawn_interval_s = 2.0f,
                            uint32_t max_vehicles     = 50);

    /// Attempt to spawn a vehicle. Returns the new Vehicle if spawning
    /// succeeded (interval elapsed and max not reached), or nullopt otherwise.
    std::optional<Vehicle> try_spawn(const RoadNetwork&   network,
                                     uint32_t             active_count,
                                     float                sim_elapsed_s);

    /// Build a route from `source` to `destination` using a greedy walk.
    /// Returns an empty vector if no path can be found.
    std::vector<RoadId> build_route(const RoadNetwork& network,
                                    IntersectionId     source,
                                    IntersectionId     destination) const;

    // ── Configuration ─────────────────────────────────────────────────────
    float    spawn_interval_s{ 2.0f };
    uint32_t max_vehicles    { 50 };

private:
    std::mt19937 rng_;
    uint32_t     next_vehicle_id_{ 0 };
    float        last_spawn_s_   { -999.f }; // Forces first spawn immediately

    /// Pick a random intersection ID from the network.
    IntersectionId random_intersection(const RoadNetwork& network);
};

} // namespace syntraq
