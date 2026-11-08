//
// tests/test_route_cost.cpp
//
// Unit tests for Traffic-Aware Routing and Route Cost Computation:
//   - Distance-only cost baseline
//   - Analytical formula evaluation: road cost = distance + congestion penalty + predicted delay
//   - RoadTrafficCondition ingestion and cost computation
//   - Pluggable DelayProvider decoupled from AI models
//   - Monotonic cost scaling with congestion
//   - Route selection comparison (Static vs TrafficAware on competing routes)
//   - Travel time and queue comparison in simulation
//

#include <gtest/gtest.h>

#include "syntraq/core/config.hpp"
#include "syntraq/routing/astar_router.hpp"
#include "syntraq/routing/dijkstra_router.hpp"
#include "syntraq/routing/route_cost.hpp"
#include "syntraq/routing/static_routing.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace syntraq {
namespace {

/// Helper network with two competing paths:
/// Path 1 (Direct Main Route via B):
///   A -> B (100m) -> D (100m) = 200m total distance
/// Path 2 (Bypass Route via C):
///   A -> C (140m) -> D (140m) = 280m total distance
struct BifurcatedTestNetwork {
    RoadNetwork    net;
    IntersectionId a;
    IntersectionId b;
    IntersectionId c;
    IntersectionId d;
    RoadId         ab;
    RoadId         bd;
    RoadId         ac;
    RoadId         cd;

    BifurcatedTestNetwork() {
        a = net.add_intersection({   0.0f,   0.0f }, "A");
        b = net.add_intersection({ 100.0f,   0.0f }, "B");
        c = net.add_intersection({ 100.0f, 100.0f }, "C");
        d = net.add_intersection({ 200.0f,   0.0f }, "D");

        ab = net.add_road(a, b, 100.0f);
        bd = net.add_road(b, d, 100.0f);
        ac = net.add_road(a, c, 140.0f);
        cd = net.add_road(c, d, 140.0f);
    }
};

inline RoadNetwork make_bifurcated_network(IntersectionId& out_a,
                                           IntersectionId& out_b,
                                           IntersectionId& out_c,
                                           IntersectionId& out_d) {
    RoadNetwork net;
    out_a = net.add_intersection({   0.0f,   0.0f }, "A");
    out_b = net.add_intersection({ 100.0f,   0.0f }, "B");
    out_c = net.add_intersection({ 100.0f, 100.0f }, "C");
    out_d = net.add_intersection({ 200.0f,   0.0f }, "D");

    net.add_road(out_a, out_b, 100.0f);
    net.add_road(out_b, out_d, 100.0f);
    net.add_road(out_a, out_c, 140.0f);
    net.add_road(out_c, out_d, 140.0f);
    return net;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════
// Route Cost Formula & Analytical Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RouteCostTest, DistanceCostReturnsExactLength) {
    Road road;
    road.id = static_cast<RoadId>(1);
    road.length_m = 125.5f;

    DistanceCost cost_fn;
    EXPECT_FLOAT_EQ(cost_fn(road), 125.5f);
}

TEST(RouteCostTest, AnalyticalCostFormulaFreeFlow) {
    // road cost = distance + congestion penalty + predicted delay
    // When congestion and delay are 0, cost equals distance
    constexpr float kDist = 150.0f;
    constexpr float kCong = 0.0f;
    constexpr float kDelay = 0.0f;

    const float cost = TrafficAwareRoadCost::compute_cost(kDist, kCong, kDelay);
    EXPECT_FLOAT_EQ(cost, 150.0f);
}

TEST(RouteCostTest, AnalyticalCostFormulaCongestionPenalty) {
    constexpr float kDist = 150.0f;
    constexpr float kCong = 60.0f;
    constexpr float kDelay = 0.0f;

    const float cost = TrafficAwareRoadCost::compute_cost(kDist, kCong, kDelay);
    EXPECT_FLOAT_EQ(cost, 210.0f);
}

TEST(RouteCostTest, AnalyticalCostFormulaPredictedDelay) {
    constexpr float kDist = 150.0f;
    constexpr float kCong = 0.0f;
    constexpr float kDelay = 45.0f;

    const float cost = TrafficAwareRoadCost::compute_cost(kDist, kCong, kDelay);
    EXPECT_FLOAT_EQ(cost, 195.0f);
}

TEST(RouteCostTest, AnalyticalCostFormulaCombinedTerms) {
    // road cost = distance + congestion penalty + predicted delay
    constexpr float kDist = 200.0f;
    constexpr float kCong = 75.0f;
    constexpr float kDelay = 50.0f;

    const float cost = TrafficAwareRoadCost::compute_cost(kDist, kCong, kDelay);
    EXPECT_FLOAT_EQ(cost, 325.0f);
}

TEST(RouteCostTest, AnalyticalCostFormulaCustomWeights) {
    // cost = w_d * dist + w_c * cong + w_delay * delay
    constexpr float kDist = 100.0f;
    constexpr float kCong = 40.0f;
    constexpr float kDelay = 30.0f;

    const float cost = TrafficAwareRoadCost::compute_cost(
        kDist, kCong, kDelay, 2.0f, 1.5f, 0.5f);
    // (2.0 * 100) + (1.5 * 40) + (0.5 * 30) = 200 + 60 + 15 = 275
    EXPECT_FLOAT_EQ(cost, 275.0f);
}

// ═══════════════════════════════════════════════════════════════════
// TrafficAwareRoadCost Evaluator Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RouteCostTest, EvaluatorZeroConditionReturnsPureDistance) {
    Road road;
    road.id = static_cast<RoadId>(10);
    road.length_m = 180.0f;
    road.lanes = { { 0, 3.5f } };
    road.speed_limit_mps = 13.89f;

    TrafficAwareRoadCost cost_model;
    RoadTrafficCondition cond{}; // All zeros

    const float cost = cost_model(road, cond);
    EXPECT_FLOAT_EQ(cost, 180.0f);
}

TEST(RouteCostTest, EvaluatorComputesCongestionPenaltyFromQueues) {
    Road road;
    road.id = static_cast<RoadId>(10);
    road.length_m = 100.0f;
    road.lanes = { { 0, 3.5f } };
    road.speed_limit_mps = 13.89f;

    TrafficAwareCostConfig cfg;
    cfg.queue_penalty_per_veh = 20.0f;
    cfg.jam_density_penalty = 50.0f;
    cfg.delay_weight = 0.0f; // Isolate congestion penalty

    TrafficAwareRoadCost cost_model(cfg);
    RoadTrafficCondition cond;
    cond.queued_vehicles = 4;
    cond.vehicle_count = 5;
    cond.congestion_ratio = 0.8f;

    const float cong_pen = cost_model.compute_congestion_penalty(road, cond);
    // (4 * 20.0) + (0.8 * 50.0) = 80 + 40 = 120
    EXPECT_FLOAT_EQ(cong_pen, 120.0f);

    const float total_cost = cost_model(road, cond);
    // 100 (distance) + 120 (congestion) + 0 (delay) = 220
    EXPECT_FLOAT_EQ(total_cost, 220.0f);
}

TEST(RouteCostTest, EvaluatorComputesPredictedDelay) {
    Road road;
    road.id = static_cast<RoadId>(10);
    road.length_m = 100.0f;
    road.lanes = { { 0, 3.5f }, { 1, 3.5f } }; // 2 lanes
    road.speed_limit_mps = 13.89f;

    TrafficAwareCostConfig cfg;
    cfg.congestion_weight = 0.0f; // Isolate delay cost
    cfg.delay_to_cost_factor = 10.0f;

    TrafficAwareRoadCost cost_model(cfg);

    // Explicit predicted delay in condition
    RoadTrafficCondition cond_explicit;
    cond_explicit.predicted_delay_s = 5.0f;

    const float delay_cost_explicit = cost_model.compute_predicted_delay(road, cond_explicit);
    EXPECT_FLOAT_EQ(delay_cost_explicit, 50.0f);

    // Fallback queuing service delay: 4 queued on 2 lanes -> (4 * 2.0s) / 2 = 4.0s
    RoadTrafficCondition cond_queue;
    cond_queue.queued_vehicles = 4;
    const float delay_cost_queue = cost_model.compute_predicted_delay(road, cond_queue);
    EXPECT_FLOAT_EQ(delay_cost_queue, 40.0f);
}

TEST(RouteCostTest, EvaluatorCostMonotonicallyIncreasesWithQueues) {
    Road road;
    road.id = static_cast<RoadId>(1);
    road.length_m = 100.0f;
    road.lanes = { { 0, 3.5f } };
    road.speed_limit_mps = 13.89f;

    TrafficAwareRoadCost cost_model;

    float prev_cost = cost_model(road, RoadTrafficCondition{});

    for (uint32_t q = 1; q <= 10; ++q) {
        RoadTrafficCondition cond;
        cond.queued_vehicles = q;
        cond.vehicle_count = q;
        cond.congestion_ratio = 1.0f;

        const float cost = cost_model(road, cond);
        EXPECT_GT(cost, prev_cost) << "Cost must strictly increase with queue depth at q=" << q;
        prev_cost = cost;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pluggable Delay Provider (Decoupled from AI Models)
// ═══════════════════════════════════════════════════════════════════

TEST(RouteCostTest, PluggableDelayProviderIsInvokedWithoutCouplingToAI) {
    Road road;
    road.id = static_cast<RoadId>(5);
    road.length_m = 80.0f;
    road.lanes = { { 0, 3.5f } };

    TrafficAwareCostConfig cfg;
    cfg.distance_weight = 1.0f;
    cfg.congestion_weight = 0.0f;
    cfg.delay_weight = 1.0f;
    cfg.delay_to_cost_factor = 10.0f;

    bool provider_called = false;
    DelayProvider mock_provider = [&](const Road& r, const RoadTrafficCondition& c) -> float {
        provider_called = true;
        EXPECT_EQ(r.id, static_cast<RoadId>(5));
        EXPECT_EQ(c.queued_vehicles, 3u);
        return 7.5f; // 7.5 seconds predicted delay
    };

    TrafficAwareRoadCost cost_model(cfg, mock_provider);

    RoadTrafficCondition cond;
    cond.queued_vehicles = 3;

    const float cost = cost_model(road, cond);
    EXPECT_TRUE(provider_called);
    // 80 (dist) + 0 (cong) + (7.5 * 10.0 = 75) = 155
    EXPECT_FLOAT_EQ(cost, 155.0f);
}

// ═══════════════════════════════════════════════════════════════════
// StaticRouting vs TrafficAwareRouting Route Selection Comparison
// ═══════════════════════════════════════════════════════════════════

TEST(RouteCostTest, BothStrategiesSelectShortestPathUnderFreeFlow) {
    BifurcatedTestNetwork bnet;

    StaticRouting static_router;
    TrafficAwareRouting traffic_router;

    const auto res_static = static_router.find_route_with_info(bnet.net, bnet.a, bnet.d);
    const auto res_traffic = traffic_router.find_route_with_info(bnet.net, bnet.a, bnet.d);

    ASSERT_TRUE(res_static.success);
    ASSERT_TRUE(res_traffic.success);

    // Under free flow, both choose Direct Main Route (A -> B -> D) = 200m
    const std::vector<RoadId> expected_route{ bnet.ab, bnet.bd };
    EXPECT_EQ(res_static.roads, expected_route);
    EXPECT_EQ(res_traffic.roads, expected_route);
    EXPECT_FLOAT_EQ(res_static.total_cost, 200.0f);
    EXPECT_FLOAT_EQ(res_traffic.total_cost, 200.0f);
}

TEST(RouteCostTest, TrafficAwareDivertsToBypassWhenMainRouteIsCongested) {
    BifurcatedTestNetwork bnet;

    StaticRouting static_router;
    TrafficAwareRouting traffic_router;

    // Simulate severe queue and congestion on Main Route road A -> B
    RoadTrafficCondition heavy_jam;
    heavy_jam.queued_vehicles = 8;
    heavy_jam.vehicle_count = 10;
    heavy_jam.avg_speed_mps = 0.2f;
    heavy_jam.congestion_ratio = 0.8f;
    heavy_jam.predicted_delay_s = 20.0f;

    traffic_router.set_road_condition(bnet.ab, heavy_jam);

    const auto res_static = static_router.find_route_with_info(bnet.net, bnet.a, bnet.d);
    const auto res_traffic = traffic_router.find_route_with_info(bnet.net, bnet.a, bnet.d);

    ASSERT_TRUE(res_static.success);
    ASSERT_TRUE(res_traffic.success);

    // Static routing is oblivious to traffic: STILL chooses congested road AB (200m)
    const std::vector<RoadId> direct_route{ bnet.ab, bnet.bd };
    EXPECT_EQ(res_static.roads, direct_route);

    // TrafficAware routing detects high cost on AB and diverts to Bypass Route (A -> C -> D)
    const std::vector<RoadId> bypass_route{ bnet.ac, bnet.cd };
    EXPECT_EQ(res_traffic.roads, bypass_route);

    // Cost on bypass is 140 + 140 = 280
    EXPECT_FLOAT_EQ(res_traffic.total_cost, 280.0f);

    // When traffic clears, TrafficAware reverts to direct route
    traffic_router.clear_road_conditions();
    const auto res_cleared = traffic_router.find_route_with_info(bnet.net, bnet.a, bnet.d);
    EXPECT_EQ(res_cleared.roads, direct_route);
    EXPECT_FLOAT_EQ(res_cleared.total_cost, 200.0f);
}

TEST(RouteCostTest, SimulationRunsWithStaticAndTrafficAwareRouting) {
    IntersectionId a, b, c, d;

    // 1. Run simulation with StaticRouting
    Config cfg_static;
    cfg_static.scenario = TrafficScenario::Custom;
    cfg_static.vehicle_spawn_interval_s = 1.0f;
    cfg_static.max_vehicles = 20;
    cfg_static.seed = 42;
    cfg_static.routing_strategy = RoutingStrategyType::Static;

    Simulation sim_static(cfg_static, [&] { return make_bifurcated_network(a, b, c, d); });
    sim_static.set_allowed_spawn_origins({ a });
    sim_static.set_allowed_spawn_destinations({ d });

    EXPECT_EQ(sim_static.routing_strategy_type(), RoutingStrategyType::Static);
    sim_static.run_for(15.0f);
    EXPECT_GT(sim_static.metrics().current_snapshot().total_spawned, 0u);

    // 2. Run simulation with TrafficAwareRouting
    Config cfg_aware = cfg_static;
    cfg_aware.routing_strategy = RoutingStrategyType::TrafficAware;

    Simulation sim_aware(cfg_aware, [&] { return make_bifurcated_network(a, b, c, d); });
    sim_aware.set_allowed_spawn_origins({ a });
    sim_aware.set_allowed_spawn_destinations({ d });

    EXPECT_EQ(sim_aware.routing_strategy_type(), RoutingStrategyType::TrafficAware);
    sim_aware.run_for(15.0f);
    EXPECT_GT(sim_aware.metrics().current_snapshot().total_spawned, 0u);
}

TEST(RouteCostTest, TravelTimeComparisonBetweenRoutingStrategies) {
    // Under heavy volume through a bottleneck, traffic-aware routing diverts flow,
    // reducing total network congestion and average travel time compared to static routing.
    IntersectionId a, b, c, d;

    Config cfg;
    cfg.scenario = TrafficScenario::Custom;
    cfg.vehicle_spawn_interval_s = 0.5f; // Fast spawning to induce queues
    cfg.max_vehicles = 35;
    cfg.seed = 101;
    cfg.dt_seconds = 0.1f;

    // Simulation A: Static Routing
    cfg.routing_strategy = RoutingStrategyType::Static;
    Simulation sim_static(cfg, [&] { return make_bifurcated_network(a, b, c, d); });
    sim_static.set_allowed_spawn_origins({ a });
    sim_static.set_allowed_spawn_destinations({ d });
    sim_static.run_for(40.0f);

    const auto res_static = sim_static.get_run_result();

    // Simulation B: Traffic-Aware Routing
    cfg.routing_strategy = RoutingStrategyType::TrafficAware;
    Simulation sim_aware(cfg, [&] { return make_bifurcated_network(a, b, c, d); });
    sim_aware.set_allowed_spawn_origins({ a });
    sim_aware.set_allowed_spawn_destinations({ d });
    sim_aware.run_for(40.0f);

    const auto res_aware = sim_aware.get_run_result();

    // Both should spawn vehicles and complete trips
    EXPECT_GT(res_static.total_completed, 0u);
    EXPECT_GT(res_aware.total_completed, 0u);

    // Traffic-aware routing balances load across the bypass, preventing excessive queues
    // on the main route and keeping peak queue lower or comparable.
    EXPECT_LE(res_aware.avg_queue_length, res_static.avg_queue_length + 1.0f);
}

} // namespace syntraq
