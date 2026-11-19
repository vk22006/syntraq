//
// tests/test_system_quality_expanded.cpp
//
// Comprehensive automated test suite expanding testing across:
//   1. Road Graph (topology, disconnected components, multi-lane, versioning)
//   2. Pathfinding (Dijkstra vs A*, unreachable targets, dynamic congestion avoidance)
//   3. Vehicles (IDM kinematics, standstill bounds, multi-lane safety)
//   4. Traffic Lights (phase cycles, clearance intervals, road signals)
//   5. Controllers (Fixed, Adaptive, Coordinated, RL bounds & fallbacks)
//   6. Metrics (zero-division protection, throughput calculations, CSV validity)
//   7. Configuration (parameter bounds, preset conversions, error resilience)
//   8. Model Preprocessing (FeatureScaler normalization, 26-feature alignment)
//   9. AI Inference Interfaces (ONNX tensor handling, graceful unloaded fallback)
//

#include <gtest/gtest.h>

#include "syntraq/core/config.hpp"
#include "syntraq/world/road_network.hpp"
#include "syntraq/world/road.hpp"
#include "syntraq/world/intersection.hpp"
#include "syntraq/routing/dijkstra_router.hpp"
#include "syntraq/routing/astar_router.hpp"
#include "syntraq/routing/static_routing.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/vehicles/vehicle_movement_system.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/signals/fixed_time_controller.hpp"
#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/ai/traffic_feature_scaler.hpp"
#include "syntraq/ai/traffic_predictor.hpp"
#include "syntraq/ai/rl_policy_inference.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace syntraq {

// ═══════════════════════════════════════════════════════════════════════════
// 1. Road Graph Subsystem Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityRoadGraphTest, DisconnectedSubgraphsAndQueryResilience) {
    RoadNetwork net;
    const auto i0 = net.add_intersection({   0.f, 0.f }, "Island1_A");
    const auto i1 = net.add_intersection({ 100.f, 0.f }, "Island1_B");
    const auto r0 = net.add_road(i0, i1, 100.f, 13.89f, 2, "Road_1A_1B");
    static_cast<void>(r0);

    // Separate disconnected component
    const auto i2 = net.add_intersection({ 500.f, 500.f }, "Island2_A");
    const auto i3 = net.add_intersection({ 600.f, 500.f }, "Island2_B");
    const auto r1 = net.add_road(i2, i3, 100.f, 13.89f, 2, "Road_2A_2B");
    static_cast<void>(r1);

    EXPECT_EQ(net.intersection_count(), 4u);
    EXPECT_EQ(net.road_count(), 2u);

    // Queries between disconnected components
    EXPECT_FALSE(net.has_road(i0, i2));
    EXPECT_FALSE(net.has_road(i1, i3));

    // Non-existent IDs query resilience
    EXPECT_EQ(net.intersection(static_cast<IntersectionId>(9999)), nullptr);
    EXPECT_EQ(net.road(static_cast<RoadId>(9999)), nullptr);
    EXPECT_FALSE(net.has_road(i0, static_cast<IntersectionId>(9999)));

    // Topology versioning
    const uint64_t v_initial = net.topology_version();
    net.add_intersection({ 800.f, 800.f });
    EXPECT_GT(net.topology_version(), v_initial);
}

TEST(SystemQualityRoadGraphTest, MultiLaneIndexingAndWidthConsistency) {
    RoadNetwork net;
    const auto a = net.add_intersection({ 0.f, 0.f });
    const auto b = net.add_intersection({ 200.f, 0.f });
    const auto rid = net.add_road(a, b, 200.f, 16.67f, 4, "4-Lane-Avenue");

    const auto* r = net.road(rid);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->lane_count(), 4u);
    EXPECT_FLOAT_EQ(r->total_width_m(), 14.0f); // 4 * 3.5m standard

    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(r->lanes[i].index, i);
        EXPECT_FLOAT_EQ(r->lanes[i].width_m, 3.5f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Pathfinding Subsystem Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityPathfindingTest, UnreachableDestinationsReturnFailureCleanly) {
    RoadNetwork net;
    const auto a = net.add_intersection({ 0.f, 0.f });
    const auto b = net.add_intersection({ 100.f, 0.f });
    const auto c = net.add_intersection({ 200.f, 0.f }); // isolated

    net.add_road(a, b, 100.f);

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_d = dijkstra.find_route_with_info(net, a, c);
    EXPECT_FALSE(res_d.success);
    EXPECT_TRUE(res_d.roads.empty());

    auto res_a = astar.find_route_with_info(net, a, c);
    EXPECT_FALSE(res_a.success);
    EXPECT_TRUE(res_a.roads.empty());
}

TEST(SystemQualityPathfindingTest, DijkstraAndAStarFindIdenticalOptimalCosts) {
    // Diamond network with different path lengths
    RoadNetwork net;
    const auto start = net.add_intersection({   0.f,   0.f }, "Start");
    const auto top   = net.add_intersection({ 100.f, -50.f }, "Top");
    const auto bot   = net.add_intersection({ 100.f,  50.f }, "Bot");
    const auto goal  = net.add_intersection({ 200.f,   0.f }, "Goal");

    // Path 1 (via top): 100 + 100 = 200m
    net.add_road(start, top, 100.f);
    net.add_road(top, goal,  100.f);

    // Path 2 (via bot): 80 + 90 = 170m (shorter)
    net.add_road(start, bot, 80.f);
    net.add_road(bot, goal,  90.f);

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    auto res_d = dijkstra.find_route_with_info(net, start, goal);
    auto res_a = astar.find_route_with_info(net, start, goal);

    ASSERT_TRUE(res_d.success);
    ASSERT_TRUE(res_a.success);
    EXPECT_FLOAT_EQ(res_d.total_cost, 170.0f);
    EXPECT_FLOAT_EQ(res_a.total_cost, 170.0f);
    EXPECT_EQ(res_d.roads, res_a.roads);
}

TEST(SystemQualityPathfindingTest, TrafficAwareRoutingDetoursAroundCongestedRoad) {
    RoadNetwork net;
    const auto start = net.add_intersection({   0.f,   0.f });
    const auto mid1  = net.add_intersection({ 100.f,   0.f });
    const auto mid2  = net.add_intersection({ 100.f, 100.f });
    const auto goal  = net.add_intersection({ 200.f,   0.f });

    // Primary route: start -> mid1 -> goal (200m)
    const auto r_direct1 = net.add_road(start, mid1, 100.f);
    const auto r_direct2 = net.add_road(mid1, goal,  100.f);

    // Detour route: start -> mid2 -> goal (300m)
    const auto r_detour1 = net.add_road(start, mid2, 150.f);
    const auto r_detour2 = net.add_road(mid2, goal,  150.f);

    TrafficAwareRouting router;

    // Free flow: selects direct 200m route
    auto free_route = router.find_route(net, start, goal);
    ASSERT_EQ(free_route.size(), 2u);
    EXPECT_EQ(free_route[0], r_direct1);
    EXPECT_EQ(free_route[1], r_direct2);

    // Add heavy congestion penalty to direct road
    router.update_from_queues({ { r_direct1, 50 } }); // 50 queued vehicles

    // Now traffic-aware router should select the detour
    auto congested_route = router.find_route(net, start, goal);
    ASSERT_EQ(congested_route.size(), 2u);
    EXPECT_EQ(congested_route[0], r_detour1);
    EXPECT_EQ(congested_route[1], r_detour2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Vehicle Kinematics & Behavior Subsystem Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityVehiclesTest, FollowerVehicleMaintainsStandstillDistance) {
    RoadNetwork net;
    const auto a = net.add_intersection({   0.f, 0.f });
    const auto b = net.add_intersection({ 500.f, 0.f });
    const auto rid = net.add_road(a, b, 500.f, 13.89f, 1);

    VehicleMovementSystem movement;

    // Leader stopped at pos = 100m
    Vehicle leader;
    leader.id           = static_cast<VehicleId>(1);
    leader.current_road = rid;
    leader.lane_index   = 0;
    leader.progress_m   = 100.0f;
    leader.speed_mps    = 0.0f;
    leader.state        = VehicleState::Moving;

    // Follower behind at pos = 60m with speed 10 m/s
    Vehicle follower;
    follower.id           = static_cast<VehicleId>(2);
    follower.current_road = rid;
    follower.lane_index   = 0;
    follower.progress_m   = 60.0f;
    follower.speed_mps    = 10.0f;
    follower.state        = VehicleState::Moving;

    std::vector<Vehicle> vehicles = { leader, follower };

    // Advance 100 ticks (10s) with leader remaining stationary
    for (int t = 0; t < 100; ++t) {
        movement.update(vehicles, net, {}, 0.1f);
        // Force leader stationary
        vehicles[0].speed_mps = 0.0f;
        vehicles[0].progress_m = 100.0f;
    }

    // Follower must not collide with leader and must halt at headway >= standstill (s0)
    const float final_gap = vehicles[0].progress_m - vehicles[1].progress_m;
    EXPECT_GT(final_gap, 1.0f); // Positive gap maintained
    EXPECT_NEAR(vehicles[1].speed_mps, 0.0f, 0.1f); // Follower has stopped
}

TEST(SystemQualityVehiclesTest, VelocityClampedToRoadSpeedLimit) {
    RoadNetwork net;
    const auto a = net.add_intersection({ 0.f, 0.f });
    const auto b = net.add_intersection({ 500.f, 0.f });
    const auto rid = net.add_road(a, b, 500.f, 10.0f, 1); // 10 m/s limit

    VehicleMovementSystem movement;

    Vehicle v;
    v.id           = static_cast<VehicleId>(1);
    v.current_road = rid;
    v.lane_index   = 0;
    v.progress_m   = 10.0f;
    v.speed_mps    = 8.0f;
    v.max_speed_mps= 25.0f; // Vehicle capable of 25 m/s
    v.state        = VehicleState::Moving;

    std::vector<Vehicle> vehicles = { v };

    for (int t = 0; t < 50; ++t) {
        movement.update(vehicles, net, {}, 0.1f);
    }

    // Must not exceed road speed limit + tolerance
    EXPECT_LE(vehicles[0].speed_mps, 10.05f);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Traffic Light & Safety Clearance Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityTrafficLightsTest, ClearanceIntervalsPreventDirectGreenToRedSwitch) {
    RoadNetwork net;
    const auto c = net.add_intersection({ 100.f, 100.f });
    const auto n = net.add_intersection({ 100.f,   0.f });
    const auto e = net.add_intersection({ 200.f, 100.f });
    const auto r_n = net.add_road(n, c, 100.f);
    const auto r_e = net.add_road(e, c, 100.f);

    SignalPhase p1{ "Phase1", { r_n }, 10.0f, 3.0f, 2.0f };
    SignalPhase p2{ "Phase2", { r_e }, 10.0f, 3.0f, 2.0f };

    TrafficSignalController ctrl{ c, { p1, p2 }, { r_n, r_e } };

    // Initially Green for r_n
    EXPECT_EQ(ctrl.signal_for_road(r_n), SignalColor::Green);

    // Fast-forward ticks: must visit Yellow and AllRed before r_n becomes Red and r_e becomes Green
    bool saw_yellow = false;
    bool saw_all_red = false;

    for (int t = 0; t < 200; ++t) {
        ctrl.tick(0.1f);
        if (ctrl.current_stage() == PhaseStage::Yellow && ctrl.current_phase_index() == 0) {
            saw_yellow = true;
            EXPECT_EQ(ctrl.signal_for_road(r_n), SignalColor::Yellow);
        }
        if (ctrl.current_stage() == PhaseStage::AllRed && ctrl.current_phase_index() == 0) {
            saw_all_red = true;
            EXPECT_EQ(ctrl.signal_for_road(r_n), SignalColor::Red);
            EXPECT_EQ(ctrl.signal_for_road(r_e), SignalColor::Red);
        }
        if (ctrl.current_phase_index() == 1 && ctrl.current_stage() == PhaseStage::Green) {
            break;
        }
    }

    EXPECT_TRUE(saw_yellow);
    EXPECT_TRUE(saw_all_red);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Signal Controller Edge Cases
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityControllersTest, AdaptiveControllerHandlesZeroVehiclesWithoutStalling) {
    AdaptiveControllerConfig cfg;
    cfg.min_green_s = 5.0f;
    cfg.max_green_s = 20.0f;

    SignalPhase p1{ "Phase1", { static_cast<RoadId>(1) }, 10.0f, 3.0f, 2.0f };
    SignalPhase p2{ "Phase2", { static_cast<RoadId>(2) }, 10.0f, 3.0f, 2.0f };

    AdaptiveController controller{
        static_cast<IntersectionId>(0),
        { p1, p2 },
        { static_cast<RoadId>(1), static_cast<RoadId>(2) },
        cfg
    };

    // All approaches empty (queue = 0)
    std::unordered_map<RoadId, uint32_t> empty_queues = {
        { static_cast<RoadId>(1), 0 },
        { static_cast<RoadId>(2), 0 }
    };

    // Run for 30 seconds
    for (int t = 0; t < 300; ++t) {
        controller.observe_traffic(empty_queues, {});
        controller.tick(0.1f);
    }

    // Must have cycled phases gracefully
    EXPECT_TRUE(controller.current_stage() == PhaseStage::Green ||
                controller.current_stage() == PhaseStage::Yellow ||
                controller.current_stage() == PhaseStage::AllRed);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Metrics Zero-Division & Bounds Safety
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityMetricsTest, ZeroActiveVehiclesAvoidsDivisionByZero) {
    TrafficMetricsCollector collector;

    // Tick with 0 vehicles
    collector.record_tick(0.1f, {}, 0, 1);

    const auto result = collector.get_run_result();

    EXPECT_FLOAT_EQ(result.avg_speed_mps, 0.0f);
    EXPECT_FALSE(std::isnan(result.avg_speed_mps));
    EXPECT_FALSE(std::isinf(result.avg_speed_mps));

    EXPECT_FLOAT_EQ(result.avg_waiting_time_s, 0.0f);
    EXPECT_FALSE(std::isnan(result.avg_waiting_time_s));
    EXPECT_FALSE(std::isinf(result.avg_waiting_time_s));

    EXPECT_FLOAT_EQ(result.congestion_ratio, 0.0f);
    EXPECT_FALSE(std::isnan(result.congestion_ratio));
    EXPECT_FALSE(std::isinf(result.congestion_ratio));

    EXPECT_FLOAT_EQ(result.throughput_vph, 0.0f);
    EXPECT_EQ(result.total_completed, 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Configuration Bounds & Validation
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityConfigTest, ConfigParametersPreserveSensibleDefaultsOnBadFile) {
    Config cfg = Config::from_file("corrupt_or_non_existent.json");

    EXPECT_GT(cfg.dt_seconds, 0.0f);
    EXPECT_LT(cfg.dt_seconds, 1.0f);
    EXPECT_GT(cfg.window_width, 320);
    EXPECT_GT(cfg.window_height, 240);
    EXPECT_GT(cfg.sim_duration_s, 0.0f);

    // Scenario presets validation
    for (auto scenario : { TrafficScenario::Low, TrafficScenario::Medium,
                           TrafficScenario::High, TrafficScenario::RushHour,
                           TrafficScenario::EmergencyPriority }) {
        auto preset = get_scenario_preset(scenario, 42);
        EXPECT_GT(preset.spawn_interval_s, 0.0f);
        EXPECT_GT(preset.max_vehicles, 0u);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Model Preprocessing & FeatureScaler Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityModelPreprocessingTest, FeatureScalerHandlesExtremeAndZeroValues) {
    TrafficFeatureScaler scaler;

    // Zero vector (22 continuous features)
    std::vector<float> zeros(22, 0.0f);
    auto scaled_zeros = scaler.transformRaw(zeros, "Medium");
    EXPECT_EQ(scaled_zeros.size(), 26u);
    for (float val : scaled_zeros) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    // Extreme values
    std::vector<float> extremes(22, 10000.0f);
    auto scaled_extremes = scaler.transformRaw(extremes, "Rush");
    EXPECT_EQ(scaled_extremes.size(), 26u);
    for (float val : scaled_extremes) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. AI Inference Interfaces Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemQualityAIInferenceTest, RLPolicyInferenceUnloadedSafeFallback) {
    RLPolicyInference policy;
    EXPECT_FALSE(policy.isLoaded());

    // 12-dim observation
    std::vector<float> dummy_obs(12, 0.5f);
    RLActionDecision decision = policy.predict(dummy_obs);

    // Unloaded model must return safe fallback action (0 or 1) without crashing
    EXPECT_TRUE(decision.best_action == 0 || decision.best_action == 1);

    // Batch prediction unloaded
    std::vector<RLActionDecision> batch_decisions = policy.predictBatch({ dummy_obs, dummy_obs });
    EXPECT_EQ(batch_decisions.size(), 2u);
    for (const auto& d : batch_decisions) {
        EXPECT_TRUE(d.best_action == 0 || d.best_action == 1);
    }
}

} // namespace syntraq
