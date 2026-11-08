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

#include <memory>
#include <unordered_map>
#include <vector>

namespace syntraq {

class TrafficAwareRouting : public IRouter {
public:
    explicit TrafficAwareRouting(
        TrafficAwareCostConfig config = {},
        DelayProvider delay_provider = nullptr);

    ~TrafficAwareRouting() override = default;

    TrafficAwareRouting(TrafficAwareRouting&&) noexcept = default;
    TrafficAwareRouting& operator=(TrafficAwareRouting&&) noexcept = default;
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

    // ── Pluggable Delay Estimation (Decoupled from AI models) ─────────────

    /// Attach a custom delay prediction function (e.g. ML inference wrapper, heuristic, or analytic model)
    void set_delay_provider(DelayProvider provider);

    // ── Cost Tuning & Inspection ──────────────────────────────────────────

    void set_cost_config(const TrafficAwareCostConfig& config) noexcept;
    [[nodiscard]] const TrafficAwareCostConfig& cost_config() const noexcept;

    [[nodiscard]] const std::unordered_map<RoadId, RoadTrafficCondition>&
        road_conditions() const noexcept { return conditions_; }

    [[nodiscard]] RoadTrafficCondition condition_for(RoadId id) const noexcept;

    /// Compute effective total edge cost for a specific road with current traffic state
    [[nodiscard]] float compute_road_cost(const Road& road) const noexcept;

private:
    TrafficAwareRoadCost                             cost_model_;
    std::unordered_map<RoadId, RoadTrafficCondition> conditions_{};
    std::unique_ptr<AStarRouter>                     router_{ nullptr };

    void setup_router();
};

} // namespace syntraq
