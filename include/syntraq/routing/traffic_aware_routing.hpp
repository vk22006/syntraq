#pragma once

//
// syntraq/routing/traffic_aware_routing.h
//
// TrafficAwareRouting — Dynamic traffic-responsive pathfinding strategy.
// Incorporates real-time traffic conditions into route costs:
//   road cost = distance + congestion penalty + predicted delay
//
// Decoupled from any specific AI model through pluggable DelayProvider callbacks.
// Implements IRouter and IRouteProvider for seamless simulation integration.
//

#include "syntraq/routing/astar_router.hpp"
#include "syntraq/routing/route_cost.hpp"
#include "syntraq/routing/router.hpp"
#include "syntraq/vehicles/vehicle.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace syntraq {

class TrafficAwareRouting : public IRouter {
public:
    explicit TrafficAwareRouting(
        TrafficAwareCostConfig config = {},
        DelayProvider delay_provider = nullptr);

    ~TrafficAwareRouting() override = default;

    TrafficAwareRouting(TrafficAwareRouting&&) = delete;
    TrafficAwareRouting& operator=(TrafficAwareRouting&&) = delete;
    TrafficAwareRouting(const TrafficAwareRouting&) = delete;
    TrafficAwareRouting& operator=(const TrafficAwareRouting&) = delete;

    [[nodiscard]] RouteResult find_route_with_info(
        const RoadNetwork& network,
        IntersectionId     source,
        IntersectionId     destination) const override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "TrafficAwareRouting";
    }

    // ── Traffic State Updates ─────────────────────────────────────────────

    /// Explicitly set the condition for a single road
    void set_road_condition(RoadId road_id, const RoadTrafficCondition& condition);

    /// Explicitly set conditions for multiple roads
    void set_road_conditions(std::unordered_map<RoadId, RoadTrafficCondition> conditions);

    /// Clear all stored road traffic conditions (reverts to free-flow distance costs)
    void clear_road_conditions() noexcept;

    /// Ingest active vehicle fleet positions and speeds to compute road traffic conditions
    void update_from_simulation(const RoadNetwork& network,
                                const std::vector<Vehicle>& vehicles);

    /// Ingest road queue counts and optional average approach speeds
    void update_from_queues(const std::unordered_map<RoadId, uint32_t>& queues,
                            const std::unordered_map<RoadId, float>& avg_speeds = {});

    // ── Emergency Priority Corridor Rerouting ────────────────────────────

    /// Set the road IDs currently part of an active emergency vehicle priority corridor
    void set_emergency_corridor_roads(std::unordered_set<RoadId> roads);

    /// Clear all active emergency corridor road designations
    void clear_emergency_corridor_roads() noexcept;

    [[nodiscard]] const std::unordered_set<RoadId>& emergency_corridor_roads() const noexcept {
        return emergency_corridor_roads_;
    }

    // ── Pluggable Delay Estimation (Decoupled from AI models) ─────────────

    /// Attach a custom delay prediction function (e.g. ML inference wrapper, heuristic, or analytic model)
    void set_delay_provider(DelayProvider provider);

    // ── Cost Tuning & Inspection ──────────────────────────────────────────

    void set_cost_config(const TrafficAwareCostConfig& config) noexcept;
    [[nodiscard]] const TrafficAwareCostConfig& cost_config() const noexcept;

    [[nodiscard]] const std::unordered_map<RoadId, RoadTrafficCondition>&
        road_conditions() const noexcept;

    [[nodiscard]] RoadTrafficCondition condition_for(RoadId id) const noexcept;

    /// Compute effective total edge cost for a specific road with current traffic state
    [[nodiscard]] float compute_road_cost(const Road& road) const noexcept;

    /// Enable or disable internal versioned route cache (enabled by default).
    void set_cache_enabled(bool enabled) noexcept;
    [[nodiscard]] bool is_cache_enabled() const noexcept { return cache_enabled_; }
    void clear_cache() const noexcept;
    [[nodiscard]] uint64_t cache_hits() const noexcept { return cache_hits_; }
    [[nodiscard]] uint64_t cache_misses() const noexcept { return cache_misses_; }

private:
    TrafficAwareRoadCost cost_model_;

    // Dense epoch-stamped traffic condition storage (zero per-tick hash map allocations)
    std::vector<RoadTrafficCondition> dense_conditions_{};
    std::vector<uint32_t>             dense_epochs_{};
    std::vector<float>                scratch_speed_sums_{};
    std::vector<uint32_t>             active_road_ids_{};
    uint32_t                          current_epoch_{ 1 };

    // Lazy compatibility view for road_conditions()
    mutable std::unordered_map<RoadId, RoadTrafficCondition> conditions_view_{};
    mutable bool                                             conditions_view_dirty_{ false };

    // Emergency corridor state with O(1) dense mask
    std::unordered_set<RoadId> emergency_corridor_roads_{};
    std::vector<uint8_t>       corridor_mask_{};
    bool                       has_emergency_corridors_{ false };

    std::unique_ptr<AStarRouter> router_{ nullptr };

    // Generation-versioned route cache
    bool     cache_enabled_{ true };
    uint64_t state_version_{ 1 };
    mutable std::mutex                              cache_mutex_{};
    mutable std::unordered_map<uint64_t, RouteResult> route_cache_{};
    mutable uint64_t                                cached_state_version_{ 0 };
    mutable const RoadNetwork*                      cached_network_{ nullptr };
    mutable uint64_t                                cached_topology_version_{ 0 };
    mutable uint64_t                                cache_hits_{ 0 };
    mutable uint64_t                                cache_misses_{ 0 };

    void setup_router();
    void ensure_dense_capacity(uint32_t road_idx);
    void advance_epoch() noexcept;
};

} // namespace syntraq
