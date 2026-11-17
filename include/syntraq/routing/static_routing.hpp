#pragma once

//
// syntraq/routing/static_routing.h
//
// StaticRouting — Baseline distance-only (or static cost) pathfinding strategy.
// Evaluates route selection without incorporating dynamic traffic conditions.
// Implements IRouter and IRouteProvider for direct integration with simulation.
//

#include "syntraq/routing/astar_router.hpp"
#include "syntraq/routing/dijkstra_router.hpp"
#include "syntraq/routing/route_cost.hpp"
#include "syntraq/routing/router.hpp"

#include <cstdint>
#include <memory>

namespace syntraq {

class StaticRouting : public IRouter {
public:
    enum class Algorithm : uint8_t {
        AStar,
        Dijkstra
    };

    explicit StaticRouting(Algorithm algo = Algorithm::AStar,
                           RoadCostFunction cost_fn = DistanceCost{});

    ~StaticRouting() override;

    // Movable, non-copyable due to router ownership
    StaticRouting(StaticRouting&&) noexcept;
    StaticRouting& operator=(StaticRouting&&) noexcept;
    StaticRouting(const StaticRouting&) = delete;
    StaticRouting& operator=(const StaticRouting&) = delete;

    [[nodiscard]] RouteResult find_route_with_info(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "StaticRouting";
    }

    void set_algorithm(Algorithm algo);
    [[nodiscard]] Algorithm algorithm() const noexcept {
        return algorithm_;
    }

    void set_cost_function(RoadCostFunction cost_fn);
    [[nodiscard]] const RoadCostFunction& cost_function() const noexcept {
        return cost_fn_;
    }

    /// Enable or disable internal topology-validated route cache (enabled by default).
    void set_cache_enabled(bool enabled) noexcept;
    [[nodiscard]] bool is_cache_enabled() const noexcept { return cache_enabled_; }
    void clear_cache() const noexcept;
    [[nodiscard]] uint64_t cache_hits() const noexcept;
    [[nodiscard]] uint64_t cache_misses() const noexcept;

private:
    struct RouteCache;

    Algorithm                       algorithm_{ Algorithm::AStar };
    RoadCostFunction                cost_fn_{ DistanceCost{} };
    std::unique_ptr<AStarRouter>    astar_{ nullptr };
    std::unique_ptr<DijkstraRouter> dijkstra_{ nullptr };
    bool                            cache_enabled_{ true };
    std::unique_ptr<RouteCache>     cache_;

    void rebuild_backends();
};

} // namespace syntraq
