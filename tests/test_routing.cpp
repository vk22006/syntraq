//
// tests/test_routing.cpp
//
// Unit and integration tests for Milestone 5 Routing Engine:
//   - Dijkstra & A* shortest paths
//   - Unreachable destinations
//   - Trivial source == destination
//   - Multiple competing paths
//   - Weighted road costs (distance vs travel time)
//   - Heuristic admissibility and consistency
//   - Vehicle integration and route traversal
//

#include <gtest/gtest.h>

#include "syntraq/routing/astar_router.h"
#include "syntraq/routing/dijkstra_router.h"
#include "syntraq/routing/graph_generator.h"
#include "syntraq/routing/heuristic.h"
#include "syntraq/routing/route_cost.h"
#include "syntraq/vehicles/vehicle_movement_system.h"
#include "syntraq/vehicles/vehicle_spawner.h"
#include "syntraq/world/road_network.h"

#include <cmath>
#include <random>

namespace syntraq {
namespace {

// ── Test fixture / helper networks ───────────────────────────────────────────

struct SimpleLinearNetwork {
    RoadNetwork    net;
    IntersectionId a;
    IntersectionId b;
    IntersectionId c;
    IntersectionId d;
    RoadId         ab;
    RoadId         bc;
    RoadId         cd;

    SimpleLinearNetwork() {
        a = net.add_intersection({   0.f, 0.f }, "A");
        b = net.add_intersection({  50.f, 0.f }, "B");
        c = net.add_intersection({ 125.f, 0.f }, "C");
        d = net.add_intersection({ 225.f, 0.f }, "D");

        ab = net.add_road(a, b,  50.f);
        bc = net.add_road(b, c,  75.f);
        cd = net.add_road(c, d, 100.f);
    }
};

} // namespace

// ═══════════════════════════════════════════════════════════════════
// Trivial Source == Destination Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, TrivialSourceEqualsDestinationDijkstra) {
    SimpleLinearNetwork ln;
    DijkstraRouter router;

    auto result = router.find_route_with_info(ln.net, ln.a, ln.a);
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.roads.empty());
    EXPECT_FLOAT_EQ(result.total_cost, 0.0f);
    EXPECT_EQ(result.nodes_visited, 0u);

    auto roads = router.find_route(ln.net, ln.a, ln.a);
    EXPECT_TRUE(roads.empty());
}

TEST(RoutingTest, TrivialSourceEqualsDestinationAStar) {
    SimpleLinearNetwork ln;
    AStarRouter router;

    auto result = router.find_route_with_info(ln.net, ln.a, ln.a);
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.roads.empty());
    EXPECT_FLOAT_EQ(result.total_cost, 0.0f);
    EXPECT_EQ(result.nodes_visited, 0u);

    auto roads = router.find_route(ln.net, ln.a, ln.a);
    EXPECT_TRUE(roads.empty());
}

// ═══════════════════════════════════════════════════════════════════
// Unreachable Destinations Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, UnreachableDestinationsDisconnectedComponents) {
    RoadNetwork net;
    auto a = net.add_intersection({   0.f, 0.f }, "A");
    auto b = net.add_intersection({ 100.f, 0.f }, "B");
    auto c = net.add_intersection({ 200.f, 0.f }, "C");
    auto d = net.add_intersection({ 300.f, 0.f }, "D");

    net.add_road(a, b, 100.f);
    net.add_road(c, d, 100.f);

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_dijkstra = dijkstra.find_route_with_info(net, a, d);
    EXPECT_FALSE(res_dijkstra.success);
    EXPECT_TRUE(res_dijkstra.roads.empty());

    auto res_astar = astar.find_route_with_info(net, a, d);
    EXPECT_FALSE(res_astar.success);
    EXPECT_TRUE(res_astar.roads.empty());
}

TEST(RoutingTest, InvalidIntersectionIdsReturnEmpty) {
    SimpleLinearNetwork ln;
    DijkstraRouter dijkstra;
    AStarRouter    astar;

    constexpr IntersectionId kBogusId = static_cast<IntersectionId>(9999);

    EXPECT_FALSE(dijkstra.find_route_with_info(ln.net, ln.a, kBogusId).success);
    EXPECT_FALSE(dijkstra.find_route_with_info(ln.net, kBogusId, ln.b).success);
    EXPECT_FALSE(astar.find_route_with_info(ln.net, ln.a, kBogusId).success);
    EXPECT_FALSE(astar.find_route_with_info(ln.net, kBogusId, ln.b).success);
}

// ═══════════════════════════════════════════════════════════════════
// Shortest Path Simple Linear Chain Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, ShortestPathLinearChain) {
    SimpleLinearNetwork ln;
    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_d = dijkstra.find_route_with_info(ln.net, ln.a, ln.d);
    ASSERT_TRUE(res_d.success);
    ASSERT_EQ(res_d.roads.size(), 3u);
    EXPECT_EQ(res_d.roads[0], ln.ab);
    EXPECT_EQ(res_d.roads[1], ln.bc);
    EXPECT_EQ(res_d.roads[2], ln.cd);
    EXPECT_FLOAT_EQ(res_d.total_cost, 225.0f);

    auto res_a = astar.find_route_with_info(ln.net, ln.a, ln.d);
    ASSERT_TRUE(res_a.success);
    ASSERT_EQ(res_a.roads.size(), 3u);
    EXPECT_EQ(res_a.roads[0], ln.ab);
    EXPECT_EQ(res_a.roads[1], ln.bc);
    EXPECT_EQ(res_a.roads[2], ln.cd);
    EXPECT_FLOAT_EQ(res_a.total_cost, 225.0f);
}

// ═══════════════════════════════════════════════════════════════════
// Multiple Competing Paths Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, MultipleCompetingPathsDiamondGraph) {
    //
    //        B (top path: 50 + 50 = 100m)
    //      ↗   ↘
    //    A       D
    //      ↘   ↗
    //        C (bottom path: 20 + 20 = 40m) -- SHORTER
    //
    RoadNetwork net;
    auto a = net.add_intersection({   0.f,  50.f }, "A");
    auto b = net.add_intersection({  50.f,   0.f }, "B");
    auto c = net.add_intersection({  50.f, 100.f }, "C");
    auto d = net.add_intersection({ 100.f,  50.f }, "D");

    net.add_road(a, b, 50.f);
    net.add_road(b, d, 50.f);
    auto r_ac = net.add_road(a, c, 20.f);
    auto r_cd = net.add_road(c, d, 20.f);

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_d = dijkstra.find_route_with_info(net, a, d);
    ASSERT_TRUE(res_d.success);
    ASSERT_EQ(res_d.roads.size(), 2u);
    EXPECT_EQ(res_d.roads[0], r_ac);
    EXPECT_EQ(res_d.roads[1], r_cd);
    EXPECT_FLOAT_EQ(res_d.total_cost, 40.0f);

    auto res_a = astar.find_route_with_info(net, a, d);
    ASSERT_TRUE(res_a.success);
    ASSERT_EQ(res_a.roads.size(), 2u);
    EXPECT_EQ(res_a.roads[0], r_ac);
    EXPECT_EQ(res_a.roads[1], r_cd);
    EXPECT_FLOAT_EQ(res_a.total_cost, 40.0f);
}

TEST(RoutingTest, MultipleCompetingPathsDirectLongVsMultiHopShort) {
    // Direct road A -> D is 500m (1 hop).
    // Multi-hop path A -> B -> C -> D is 50 + 50 + 50 = 150m (3 hops).
    // Algorithms must pick the multi-hop path because total distance is shorter.
    RoadNetwork net;
    auto a = net.add_intersection({   0.f, 0.f }, "A");
    auto b = net.add_intersection({  50.f, 0.f }, "B");
    auto c = net.add_intersection({ 100.f, 0.f }, "C");
    auto d = net.add_intersection({ 150.f, 0.f }, "D");

    net.add_road(a, d, 500.f); // Direct long detour
    auto r_ab = net.add_road(a, b, 50.f);
    auto r_bc = net.add_road(b, c, 50.f);
    auto r_cd = net.add_road(c, d, 50.f);

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_d = dijkstra.find_route_with_info(net, a, d);
    ASSERT_TRUE(res_d.success);
    ASSERT_EQ(res_d.roads.size(), 3u);
    EXPECT_EQ(res_d.roads[0], r_ab);
    EXPECT_EQ(res_d.roads[1], r_bc);
    EXPECT_EQ(res_d.roads[2], r_cd);
    EXPECT_FLOAT_EQ(res_d.total_cost, 150.0f);

    auto res_a = astar.find_route_with_info(net, a, d);
    ASSERT_TRUE(res_a.success);
    ASSERT_EQ(res_a.roads.size(), 3u);
    EXPECT_EQ(res_a.roads[0], r_ab);
    EXPECT_EQ(res_a.roads[1], r_bc);
    EXPECT_EQ(res_a.roads[2], r_cd);
    EXPECT_FLOAT_EQ(res_a.total_cost, 150.0f);
}

// ═══════════════════════════════════════════════════════════════════
// Weighted Road Costs Tests (Distance vs Travel Time)
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, WeightedRoadCostsTravelTimeVsDistance) {
    // Road 1 (Highway): 300m, speed limit 30 m/s -> time = 10s
    // Road 2 (City street): 150m, speed limit 5 m/s -> time = 30s
    // By distance: Road 2 is shorter (150m vs 300m).
    // Alternative network with parallel paths:
    RoadNetwork parallel_net;
    auto src = parallel_net.add_intersection({   0.f, 0.f }, "Src");
    auto dst = parallel_net.add_intersection({ 100.f, 0.f }, "Dst");
    auto mid1 = parallel_net.add_intersection({ 50.f, -50.f }, "HighwayNode");
    auto mid2 = parallel_net.add_intersection({ 50.f,  50.f }, "StreetNode");

    // Highway route: 300m total, 30 m/s -> 10 seconds
    auto h1 = parallel_net.add_road(src, mid1, 150.f, 30.0f);
    auto h2 = parallel_net.add_road(mid1, dst, 150.f, 30.0f);

    // City route: 150m total, 5 m/s -> 30 seconds
    auto c1 = parallel_net.add_road(src, mid2,  75.f,  5.0f);
    auto c2 = parallel_net.add_road(mid2, dst,  75.f,  5.0f);

    // Test with DistanceCost:
    DijkstraRouter dist_dijkstra{ DistanceCost{} };
    auto dist_res = dist_dijkstra.find_route_with_info(parallel_net, src, dst);
    ASSERT_TRUE(dist_res.success);
    ASSERT_EQ(dist_res.roads.size(), 2u);
    EXPECT_EQ(dist_res.roads[0], c1);
    EXPECT_EQ(dist_res.roads[1], c2);
    EXPECT_FLOAT_EQ(dist_res.total_cost, 150.0f);

    // Test with FreeFlowTravelTimeCost:
    DijkstraRouter time_dijkstra{ FreeFlowTravelTimeCost{} };
    auto time_res = time_dijkstra.find_route_with_info(parallel_net, src, dst);
    ASSERT_TRUE(time_res.success);
    ASSERT_EQ(time_res.roads.size(), 2u);
    EXPECT_EQ(time_res.roads[0], h1);
    EXPECT_EQ(time_res.roads[1], h2);
    EXPECT_FLOAT_EQ(time_res.total_cost, 10.0f);

    // A* with FreeFlowTravelTimeCost:
    AStarRouter time_astar{ FreeFlowTravelTimeCost{}, ZeroHeuristic{} };
    auto time_astar_res = time_astar.find_route_with_info(parallel_net, src, dst);
    ASSERT_TRUE(time_astar_res.success);
    ASSERT_EQ(time_astar_res.roads.size(), 2u);
    EXPECT_EQ(time_astar_res.roads[0], h1);
    EXPECT_EQ(time_astar_res.roads[1], h2);
    EXPECT_FLOAT_EQ(time_astar_res.total_cost, 10.0f);
}

// ═══════════════════════════════════════════════════════════════════
// A* Heuristic Correctness Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, HeuristicAdmissibilityOnGrid) {
    // Admissibility: h(u, target) <= true_shortest_distance(u, target)
    // for all nodes u and all targets.
    auto net = make_grid_network(5, 5, 100.0f);
    DijkstraRouter dijkstra;
    EuclideanDistanceHeuristic heuristic;

    for (const auto& [u_id, u_node] : net.intersections()) {
        for (const auto& [v_id, v_node] : net.intersections()) {
            float h = heuristic(net, u_id, v_id);
            auto res = dijkstra.find_route_with_info(net, u_id, v_id);
            ASSERT_TRUE(res.success);
            float true_dist = res.total_cost;

            EXPECT_LE(h, true_dist + 1e-3f)
                << "Heuristic violated admissibility between nodes "
                << u_id << " and " << v_id;
        }
    }
}

TEST(RoutingTest, HeuristicConsistencyOnGrid) {
    // Consistency (monotonicity): h(u, t) <= cost(u, v) + h(v, t)
    // for every directed edge u -> v and every target t.
    auto net = make_grid_network(4, 4, 100.0f);
    EuclideanDistanceHeuristic heuristic;
    DistanceCost cost_fn;

    for (const auto& [r_id, road] : net.roads()) {
        float edge_cost = cost_fn(road);
        for (const auto& [t_id, t_node] : net.intersections()) {
            float h_u = heuristic(net, road.from, t_id);
            float h_v = heuristic(net, road.to,   t_id);

            EXPECT_LE(h_u, edge_cost + h_v + 1e-3f)
                << "Heuristic violated consistency across road " << r_id
                << " towards target " << t_id;
        }
    }
}

TEST(RoutingTest, AStarProducesExactOptimalPathCostAsDijkstra) {
    auto net = make_grid_network(6, 6, 80.0f);
    DijkstraRouter dijkstra;
    AStarRouter    astar;

    // Test across 25 deterministic pseudo-random source-destination pairs
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, net.intersection_count() - 1);

    for (int i = 0; i < 25; ++i) {
        IntersectionId src = static_cast<IntersectionId>(dist(rng));
        IntersectionId dst = static_cast<IntersectionId>(dist(rng));

        auto res_d = dijkstra.find_route_with_info(net, src, dst);
        auto res_a = astar.find_route_with_info(net, src, dst);

        ASSERT_EQ(res_d.success, res_a.success);
        EXPECT_NEAR(res_d.total_cost, res_a.total_cost, 1e-3f)
            << "A* path cost differed from Dijkstra for src=" << src << ", dst=" << dst;
    }
}

TEST(RoutingTest, AStarExploresFewerNodesThanDijkstraOnHorizontalTraversal) {
    // In a 10x10 grid, traveling from (0,4) to (9,4) horizontally:
    // Dijkstra expands nodes outward in 2D concentric circles, visiting most of the grid.
    // A* is directed towards the goal along the horizontal axis, expanding far fewer nodes.
    auto net = make_grid_network(10, 10, 100.0f);

    IntersectionId src = static_cast<IntersectionId>(40); // (0,4)
    IntersectionId dst = static_cast<IntersectionId>(49); // (9,4)

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_d = dijkstra.find_route_with_info(net, src, dst);
    auto res_a = astar.find_route_with_info(net, src, dst);

    ASSERT_TRUE(res_d.success);
    ASSERT_TRUE(res_a.success);
    EXPECT_FLOAT_EQ(res_d.total_cost, res_a.total_cost);

    EXPECT_LT(res_a.nodes_visited, res_d.nodes_visited);
    EXPECT_LE(res_a.nodes_visited, 20u);
    EXPECT_GE(res_d.nodes_visited, 50u);
}

TEST(RoutingTest, ZeroHeuristicAStarMatchesDijkstraNodesVisited) {
    auto net = make_grid_network(5, 5, 100.0f);
    IntersectionId src = static_cast<IntersectionId>(0);
    IntersectionId dst = static_cast<IntersectionId>(24);

    DijkstraRouter dijkstra;
    AStarRouter    astar_zero{ DistanceCost{}, ZeroHeuristic{} };

    auto res_d = dijkstra.find_route_with_info(net, src, dst);
    auto res_z = astar_zero.find_route_with_info(net, src, dst);

    ASSERT_TRUE(res_d.success);
    ASSERT_TRUE(res_z.success);
    EXPECT_FLOAT_EQ(res_d.total_cost, res_z.total_cost);
    EXPECT_EQ(res_d.nodes_visited, res_z.nodes_visited);
}

// ═══════════════════════════════════════════════════════════════════
// Vehicle System & Spawner Integration Tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoutingTest, VehicleSpawnerUsesAStarByDefault) {
    auto net = make_test_map();
    VehicleSpawner spawner(42, 2.0f, 50);

    // Spawner default provider is AStarRouter
    auto opt = spawner.try_spawn(net, 0, 0.0f);
    ASSERT_TRUE(opt.has_value());
    EXPECT_TRUE(opt->is_valid());
    EXPECT_FALSE(opt->route.empty());
    EXPECT_EQ(opt->state, VehicleState::Moving);
}

TEST(RoutingTest, VehicleFollowsAStarRouteToDestination) {
    SimpleLinearNetwork ln;
    VehicleSpawner spawner(42, 1.0f, 10, std::make_shared<AStarRouter>());

    auto route = spawner.build_route(ln.net, ln.a, ln.d);
    ASSERT_EQ(route.size(), 3u);

    Vehicle v;
    v.id          = static_cast<VehicleId>(1);
    v.route       = route;
    v.route_index = 0;
    v.destination = ln.d;
    v.current_road = route[0];
    v.progress_m  = 0.f;
    v.speed_mps   = 13.89f;
    v.max_speed_mps = 13.89f;
    v.state       = VehicleState::Moving;

    VehicleMovementSystem sys;
    std::vector<Vehicle> vehicles{ v };

    // Advance vehicle through ticks until arrival at destination
    for (int step = 0; step < 50; ++step) {
        sys.update(vehicles, ln.net, 1.0f);
        if (vehicles[0].has_arrived()) break;
    }

    EXPECT_TRUE(vehicles[0].has_arrived());
    EXPECT_EQ(vehicles[0].state, VehicleState::Arrived);
}

TEST(RoutingTest, RouteStrategySwitchingWithoutVehicleModification) {
    SimpleLinearNetwork ln;
    VehicleSpawner spawner(42);

    // 1. A*
    spawner.set_route_provider(std::make_shared<AStarRouter>());
    auto route_astar = spawner.build_route(ln.net, ln.a, ln.d);
    ASSERT_EQ(route_astar.size(), 3u);

    // 2. Dijkstra
    spawner.set_route_provider(std::make_shared<DijkstraRouter>());
    auto route_dijkstra = spawner.build_route(ln.net, ln.a, ln.d);
    ASSERT_EQ(route_dijkstra.size(), 3u);

    // 3. Greedy
    spawner.set_route_provider(std::make_shared<GreedyRouteProvider>());
    auto route_greedy = spawner.build_route(ln.net, ln.a, ln.d);
    ASSERT_EQ(route_greedy.size(), 3u);

    EXPECT_EQ(route_astar, route_dijkstra);
    EXPECT_EQ(route_dijkstra, route_greedy);
}

} // namespace syntraq
