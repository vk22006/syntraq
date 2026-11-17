//
// tests/test_performance_optimizations.cpp
//
// Unit tests verifying correctness, determinism, cache invalidation, thread-safety,
// and numerical parity of all performance optimizations.
//

#include "syntraq/ai/rl_policy_inference.hpp"
#include "syntraq/routing/astar_router.hpp"
#include "syntraq/routing/dijkstra_router.hpp"
#include "syntraq/routing/graph_generator.hpp"
#include "syntraq/routing/parallel_routing.hpp"
#include "syntraq/routing/static_routing.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/vehicles/vehicle_movement_system.hpp"
#include "syntraq/world/road_network.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>
#include <utility>
#include <vector>

namespace syntraq {
namespace {

std::string find_rl_model_path() {
    const std::string rel = "models/traffic_signal_dqn.onnx";
    if (std::filesystem::exists(rel)) return rel;
    if (std::filesystem::exists("../" + rel)) return "../" + rel;
    if (std::filesystem::exists("../../" + rel)) return "../../" + rel;
    return "";
}

// 1. RoadNetwork O(1) dense lookup index, topology versioning, and move safety
TEST(PerformanceOptimizationTest, RoadNetworkDenseLookupAndTopologyVersion) {
    RoadNetwork net;
    const uint64_t v0 = net.topology_version();

    const auto n0 = net.add_intersection(Vec2{ 0.0f, 0.0f }, "A");
    const auto n1 = net.add_intersection(Vec2{ 100.0f, 0.0f }, "B");
    const auto r0 = net.add_road(n0, n1, 100.0f, 13.89f, 2, "AB");

    EXPECT_GT(net.topology_version(), v0);
    EXPECT_EQ(net.intersection_ids().size(), 2u);
    EXPECT_EQ(net.max_intersection_id(), 2u);
    EXPECT_EQ(net.max_road_id(), 1u);

    ASSERT_NE(net.intersection(n0), nullptr);
    EXPECT_EQ(net.intersection(n0)->name, "A");
    ASSERT_NE(net.road(r0), nullptr);
    EXPECT_EQ(net.road(r0)->name, "AB");
    EXPECT_EQ(net.intersection(kInvalidIntersectionId), nullptr);
    EXPECT_EQ(net.road(kInvalidRoadId), nullptr);

    // Move construction must preserve O(1) pointer lookups
    RoadNetwork moved = std::move(net);
    ASSERT_NE(moved.intersection(n1), nullptr);
    EXPECT_EQ(moved.intersection(n1)->name, "B");
    ASSERT_NE(moved.road(r0), nullptr);
    EXPECT_EQ(moved.road(r0)->to, n1);

    const uint64_t v_before_clear = moved.topology_version();
    moved.clear();
    EXPECT_GT(moved.topology_version(), v_before_clear);
    EXPECT_EQ(moved.intersection(n0), nullptr);
    EXPECT_EQ(moved.road(r0), nullptr);
    EXPECT_TRUE(moved.intersection_ids().empty());
}

// 2. StaticRouting route cache hit/miss parity and automatic topology invalidation
TEST(PerformanceOptimizationTest, StaticRoutingCacheAndTopologyInvalidation) {
    RoadNetwork net = make_grid_network(6, 6, 100.0f);
    StaticRouting router(StaticRouting::Algorithm::AStar);

    const auto src = static_cast<IntersectionId>(0);
    const auto dst = static_cast<IntersectionId>(35);

    const auto r1 = router.find_route_with_info(net, src, dst);
    EXPECT_TRUE(r1.success);
    EXPECT_EQ(router.cache_misses(), 1u);
    EXPECT_EQ(router.cache_hits(), 0u);

    // Repeated query should hit cache and match identically
    const auto r2 = router.find_route_with_info(net, src, dst);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(router.cache_hits(), 1u);
    EXPECT_EQ(r1.roads, r2.roads);
    EXPECT_FLOAT_EQ(r1.total_cost, r2.total_cost);

    // Mutate network topology by adding a direct shortcut road -> cache must auto-invalidate
    const auto shortcut = net.add_road(src, dst, 50.0f, 20.0f, 1, "Shortcut");
    ASSERT_NE(shortcut, kInvalidRoadId);

    const auto r3 = router.find_route_with_info(net, src, dst);
    EXPECT_TRUE(r3.success);
    EXPECT_EQ(r3.roads.size(), 1u);
    EXPECT_EQ(r3.roads[0], shortcut);
    EXPECT_FLOAT_EQ(r3.total_cost, 50.0f);
    EXPECT_EQ(router.cache_misses(), 2u);
}

// 3. TrafficAwareRouting dense condition storage & versioned cache invalidation
TEST(PerformanceOptimizationTest, TrafficAwareDenseStorageAndCacheInvalidation) {
    RoadNetwork net = make_grid_network(5, 5, 100.0f);
    TrafficAwareRouting router;

    const auto src = static_cast<IntersectionId>(0);
    const auto dst = static_cast<IntersectionId>(24);

    const auto free_flow = router.find_route_with_info(net, src, dst);
    ASSERT_TRUE(free_flow.success);
    ASSERT_FALSE(free_flow.roads.empty());

    // Second query hits cache
    const auto cached = router.find_route_with_info(net, src, dst);
    EXPECT_EQ(cached.roads, free_flow.roads);
    EXPECT_EQ(router.cache_hits(), 1u);

    // Calling clear_emergency_corridor_roads when already empty must NOT flush cache
    router.clear_emergency_corridor_roads();
    (void)router.find_route_with_info(net, src, dst);
    EXPECT_EQ(router.cache_hits(), 2u);

    // Heavily congest first road of free_flow -> cache must invalidate and divert
    const RoadId blocked_road = free_flow.roads[0];
    router.set_road_condition(blocked_road, RoadTrafficCondition{ 25, 25, 0.2f, 1.0f, 60.0f });

    const auto diverted = router.find_route_with_info(net, src, dst);
    EXPECT_TRUE(diverted.success);
    EXPECT_NE(diverted.roads[0], blocked_road);
    EXPECT_EQ(router.cache_misses(), 2u);
}

// 4. Parallel batch pathfinding matches sequential results bit-for-bit across threads
TEST(PerformanceOptimizationTest, ParallelBatchPathfindingDeterminismAndThreadSafety) {
    RoadNetwork net = make_grid_network(15, 15, 100.0f);
    AStarRouter astar;
    DijkstraRouter dijkstra;

    std::mt19937 rng(777);
    std::uniform_int_distribution<uint32_t> dist(0, net.intersection_count() - 1);
    std::vector<std::pair<IntersectionId, IntersectionId>> queries;
    queries.reserve(64);
    for (int i = 0; i < 64; ++i) {
        queries.emplace_back(static_cast<IntersectionId>(dist(rng)),
                             static_cast<IntersectionId>(dist(rng)));
    }

    const auto par_astar    = find_routes_parallel(astar, net, queries, 4);
    const auto par_dijkstra = find_routes_parallel(dijkstra, net, queries, 4);

    ASSERT_EQ(par_astar.size(), queries.size());
    ASSERT_EQ(par_dijkstra.size(), queries.size());

    for (size_t i = 0; i < queries.size(); ++i) {
        const auto seq_a = astar.find_route_with_info(net, queries[i].first, queries[i].second);
        EXPECT_EQ(par_astar[i].success, seq_a.success);
        EXPECT_EQ(par_astar[i].roads, seq_a.roads);
        EXPECT_FLOAT_EQ(par_astar[i].total_cost, seq_a.total_cost);
        EXPECT_NEAR(par_dijkstra[i].total_cost, seq_a.total_cost, 1e-3f);
    }
}

// 5. VehicleMovementSystem spatial lane-bucketing handles intra-tick transitions & >4 lanes
TEST(PerformanceOptimizationTest, SpatialLaneBucketingMultiLaneAndIntraTickTransition) {
    RoadNetwork net;
    const auto n0 = net.add_intersection(Vec2{ 0.0f, 0.0f }, "N0");
    const auto n1 = net.add_intersection(Vec2{ 100.0f, 0.0f }, "N1");
    const auto n2 = net.add_intersection(Vec2{ 200.0f, 0.0f }, "N2");
    const auto r0 = net.add_road(n0, n1, 100.0f, 15.0f, 6, "R0_6Lanes");
    const auto r1 = net.add_road(n1, n2, 100.0f, 15.0f, 6, "R1_6Lanes");

    VehicleMovementSystem movement;
    std::vector<Vehicle> vehicles;

    // Two vehicles on lane 5 (exceeding initial lane_stride_=4) transitioning onto r1
    Vehicle v1;
    v1.id           = static_cast<VehicleId>(1);
    v1.route        = { r0, r1 };
    v1.current_road = r0;
    v1.lane_index   = 5;
    v1.progress_m   = 99.5f;
    v1.speed_mps    = 12.0f;
    v1.max_speed_mps = 15.0f;

    Vehicle v2;
    v2.id           = static_cast<VehicleId>(2);
    v2.route        = { r0, r1 };
    v2.current_road = r0;
    v2.lane_index   = 5;
    v2.progress_m   = 90.0f;
    v2.speed_mps    = 12.0f;
    v2.max_speed_mps = 15.0f;

    vehicles.push_back(v1);
    vehicles.push_back(v2);

    // Tick forward: v1 enters r1 near progress 0.7m; v2 must see r1's entrance occupied and hold safely
    for (int t = 0; t < 20; ++t) {
        movement.update(vehicles, net, 0.1f);
        if (vehicles[0].current_road == vehicles[1].current_road &&
            vehicles[0].state != VehicleState::Arrived &&
            vehicles[1].state != VehicleState::Arrived) {
            EXPECT_GT(vehicles[0].progress_m, vehicles[1].progress_m);
        }
    }
    EXPECT_EQ(vehicles[0].current_road, r1);
    EXPECT_EQ(vehicles[0].lane_index, 5u);
}

// 6. RLPolicyInference batched tensor inference matches single-sample predict() numerically
TEST(PerformanceOptimizationTest, RLPolicyBatchedInferenceMatchesSingleSample) {
    const std::string model_path = find_rl_model_path();
    if (model_path.empty()) {
        GTEST_SKIP() << "ONNX RL model not present in working directory.";
    }

    RLPolicyInference policy;
    ASSERT_TRUE(policy.loadModel(model_path));

    constexpr size_t kBatch = 16;
    std::vector<std::vector<float>> batch_obs(kBatch, std::vector<float>(RLPolicyInference::OBSERVATION_DIM, 0.0f));
    for (size_t i = 0; i < kBatch; ++i) {
        for (size_t j = 0; j < RLPolicyInference::OBSERVATION_DIM; ++j) {
            batch_obs[i][j] = static_cast<float>((i * 7 + j * 3) % 10) / 10.0f;
        }
    }

    const auto batch_decisions = policy.predictBatch(batch_obs);
    ASSERT_EQ(batch_decisions.size(), kBatch);

    for (size_t i = 0; i < kBatch; ++i) {
        const auto single = policy.predict(batch_obs[i]);
        EXPECT_EQ(batch_decisions[i].best_action, single.best_action);
        EXPECT_NEAR(batch_decisions[i].q_maintain, single.q_maintain, 1e-4f);
        EXPECT_NEAR(batch_decisions[i].q_switch, single.q_switch, 1e-4f);
    }
}

// 7. AdaptiveController local observation filtering ignores unrelated network roads
TEST(PerformanceOptimizationTest, AdaptiveControllerLocalObservationParity) {
    RoadNetwork net = make_single_intersection_map();
    const auto center = static_cast<IntersectionId>(0);
    auto ctrl = AdaptiveController::make_adaptive(net, center);

    const auto& p0_roads = ctrl.phases()[0].green_roads;
    const auto& p1_roads = ctrl.phases()[1].green_roads;
    ASSERT_FALSE(p0_roads.empty());
    ASSERT_FALSE(p1_roads.empty());

    // Populate map with unrelated road IDs + opposing queue
    std::unordered_map<RoadId, uint32_t> queues;
    std::unordered_map<RoadId, float> waits;
    for (uint32_t r = 100; r < 300; ++r) {
        queues[static_cast<RoadId>(r)] = 50;
        waits[static_cast<RoadId>(r)]  = 120.0f;
    }
    queues[p1_roads[0]] = 6;
    waits[p1_roads[0]]  = 15.0f;

    ctrl.observe_traffic(queues, waits);
    // Advance past min_green_s -> current_queue == 0 while opposing_queue == 6 -> should shorten early
    for (int t = 0; t < 60; ++t) {
        ctrl.tick(0.1f);
        if (ctrl.last_decision() == AdaptiveDecision::ShortenPhase) break;
    }
    EXPECT_EQ(ctrl.last_decision(), AdaptiveDecision::ShortenPhase);
    EXPECT_FALSE(ctrl.last_decision_reason().empty());
}

} // namespace
} // namespace syntraq

