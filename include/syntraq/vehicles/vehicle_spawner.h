#pragma once

//
// syntraq/vehicles/vehicle_spawner.h
//
// VehicleSpawner — creates new vehicles on the road network.
// Uses an IRouteProvider (default: GreedyRouteProvider) to generate routes.
//

#include "syntraq/vehicles/route_provider.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/world/road_network.h"

#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace syntraq {

class VehicleSpawner {
public:
    explicit VehicleSpawner(uint32_t                              rng_seed         = 42,
                            float                                 spawn_interval_s = 2.0f,
                            uint32_t                              max_vehicles     = 50,
                            std::shared_ptr<const IRouteProvider> route_provider   = nullptr);

    /// Attempt to spawn a vehicle with lane occupancy check against existing vehicles.
    std::optional<Vehicle> try_spawn(const RoadNetwork&          network,
                                     const std::vector<Vehicle>& existing_vehicles,
                                     float                       sim_elapsed_s);

    /// Overload for backwards compatibility (tests without vehicle list).
    std::optional<Vehicle> try_spawn(const RoadNetwork&   network,
                                     uint32_t             active_count,
                                     float                sim_elapsed_s);

    /// Re-seed the random number generator (for repeatable scenarios).
    void set_seed(uint32_t seed);

    /// Build a route from `source` to `destination` using the route provider.
    /// Returns an empty vector if no path can be found.
    std::vector<RoadId> build_route(const RoadNetwork& network,
                                    IntersectionId     source,
                                    IntersectionId     destination) const;

    /// Replace the route provider (e.g., for Milestone 4 shortest path).
    void set_route_provider(std::shared_ptr<const IRouteProvider> provider) {
        if (provider) route_provider_ = std::move(provider);
    }

    [[nodiscard]] const IRouteProvider& route_provider() const noexcept {
        return *route_provider_;
    }

    // ── Configuration ─────────────────────────────────────────────────────
    float    spawn_interval_s{ 2.0f };
    uint32_t max_vehicles    { 50 };

private:
    std::mt19937                          rng_;
    uint32_t                              next_vehicle_id_{ 0 };
    float                                 last_spawn_s_   { -999.f }; // Forces first spawn immediately
    std::shared_ptr<const IRouteProvider> route_provider_;

    /// Pick a random intersection ID from the network.
    IntersectionId random_intersection(const RoadNetwork& network);
};

} // namespace syntraq
