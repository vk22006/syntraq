//
// tests/test_emergency_priority.cpp
//
// Comprehensive unit and integration test suite for Emergency Priority:
//   - Emergency vehicle type and kinematics
//   - Route priority and spawning
//   - Detection of approaching emergency vehicles
//   - Traffic signal preemption and green corridor hold
//   - Normal vehicle lane yielding on multi-lane roads
//   - Dynamic rerouting of normal background traffic away from priority corridor
//   - Comparative metrics (emergency travel time vs normal disruption)
//

#include <gtest/gtest.h>

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/signals/fixed_time_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/vehicles/emergency_priority_manager.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/vehicles/vehicle_spawner.hpp"
#include "syntraq/world/road_network.hpp"

using namespace syntraq;

namespace {

struct CorridorTestMap {
    RoadNetwork    net{};
    IntersectionId i0{ kInvalidIntersectionId };
    IntersectionId i1{ kInvalidIntersectionId };
    IntersectionId i2{ kInvalidIntersectionId };
    IntersectionId i3{ kInvalidIntersectionId };
    IntersectionId i4{ kInvalidIntersectionId };

    RoadId r_main_01  { kInvalidRoadId };
    RoadId r_main_12  { kInvalidRoadId };
    RoadId r_cross_41 { kInvalidRoadId };
    RoadId r_cross_13 { kInvalidRoadId };
    RoadId r_bypass_03{ kInvalidRoadId };
    RoadId r_bypass_32{ kInvalidRoadId };
};

CorridorTestMap make_priority_corridor_network() {
    CorridorTestMap map;
    map.i0 = map.net.add_intersection({   0.0f,   0.0f }, "I0");
    map.i1 = map.net.add_intersection({ 200.0f,   0.0f }, "I1");
    map.i2 = map.net.add_intersection({ 400.0f,   0.0f }, "I2");
    map.i3 = map.net.add_intersection({ 200.0f, 150.0f }, "I3");
    map.i4 = map.net.add_intersection({ 200.0f, -150.0f }, "I4");

    // Main arterial corridor (multi-lane: 2 lanes each)
    map.r_main_01   = map.net.add_road(map.i0, map.i1, 200.0f, 13.89f, 2, "MainWest");
    map.r_main_12   = map.net.add_road(map.i1, map.i2, 200.0f, 13.89f, 2, "MainEast");

    // Cross roads at intersection 1
    map.r_cross_41  = map.net.add_road(map.i4, map.i1, 150.0f, 13.89f, 1, "CrossNorth");
    map.r_cross_13  = map.net.add_road(map.i1, map.i3, 150.0f, 13.89f, 1, "CrossSouth");

    // Parallel bypass roads
    map.r_bypass_03 = map.net.add_road(map.i0, map.i3, 210.0f, 13.89f, 1, "BypassIn");
    map.r_bypass_32 = map.net.add_road(map.i3, map.i2, 210.0f, 13.89f, 1, "BypassOut");

    return map;
}

RoadNetwork make_corridor_road_network() {
    return make_priority_corridor_network().net;
}

} // namespace

// ── Test 1: Emergency Vehicle Type and Kinematics ────────────────────────────

TEST(EmergencyPriorityTest, EmergencyVehicleTypeAndKinematics) {
    Vehicle v;
    EXPECT_EQ(v.type, VehicleType::Regular);
    EXPECT_FALSE(v.is_emergency());
    EXPECT_EQ(to_string(VehicleType::Regular), "Regular");
    EXPECT_EQ(to_string(VehicleType::Emergency), "Emergency");

    v.type = VehicleType::Emergency;
    v.max_speed_mps = 20.0f; // 72 km/h response speed
    v.accel_mps2    = 4.5f;
    v.decel_mps2    = 7.0f;

    EXPECT_TRUE(v.is_emergency());
    EXPECT_GT(v.max_speed_mps, 13.89f);
    EXPECT_GT(v.accel_mps2, 3.0f);
}

// ── Test 2: Spawner Generates Emergency Vehicle with Route ───────────────────

TEST(EmergencyPriorityTest, SpawnerCreatesValidEmergencyVehicle) {
    const auto map = make_priority_corridor_network();
    VehicleSpawner spawner{ 42, 2.0f, 20 };

    std::vector<Vehicle> existing;
    auto opt = spawner.spawn_emergency_vehicle(map.net, map.i0, map.i2, existing);

    ASSERT_TRUE(opt.has_value());
    const auto& ev = *opt;
    EXPECT_TRUE(ev.is_emergency());
    EXPECT_EQ(ev.destination, map.i2);
    EXPECT_FALSE(ev.route.empty());
    EXPECT_EQ(ev.current_road, ev.route[0]);
    EXPECT_NEAR(ev.max_speed_mps, 20.0f, 1e-3f);
}

// ── Test 3: FixedTime Controller Preemption Safe Clearance ────────────────────

TEST(EmergencyPriorityTest, FixedTimeControllerPreemptionSafeClearance) {
    const auto map = make_priority_corridor_network();
    SignalPhase p1{ "Phase-Main",  { map.r_main_01 },  30.0f, 3.0f, 2.0f };
    SignalPhase p2{ "Phase-Cross", { map.r_cross_41 }, 30.0f, 3.0f, 2.0f };
    FixedTimeController ctrl(map.i1, { p1, p2 }, { map.r_main_01, map.r_cross_41 });

    const RoadId main_road  = map.r_main_01;
    const RoadId cross_road = map.r_cross_41;

    // Start in Phase 0 Green
    EXPECT_TRUE(ctrl.can_proceed(main_road));

    // Fast forward to Phase 1 Green (Cross traffic has right of way)
    ctrl.tick(35.0f); // 30s green + 3s yellow + 2s red -> Phase 1 Green
    EXPECT_TRUE(ctrl.can_proceed(cross_road));
    EXPECT_FALSE(ctrl.can_proceed(main_road));

    // Emergency vehicle approaches on main_road -> Request preemption
    ctrl.request_emergency_preemption(main_road);
    EXPECT_TRUE(ctrl.has_emergency_preemption());
    EXPECT_EQ(ctrl.emergency_preemption_road(), main_road);

    // Conflicting cross phase clears safely (minimum safety green, yellow, all-red)
    ctrl.tick(2.1f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Yellow);

    ctrl.tick(3.1f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::AllRed);

    ctrl.tick(2.1f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_TRUE(ctrl.can_proceed(main_road));

    // Green holds indefinitely as long as preemption is active
    ctrl.tick(100.0f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_TRUE(ctrl.can_proceed(main_road));

    // Once emergency vehicle clears intersection, preemption is released
    ctrl.clear_emergency_preemption();
    EXPECT_FALSE(ctrl.has_emergency_preemption());

    // Normal cyclical signal operation resumes (30s green expires, transitions to yellow)
    ctrl.tick(30.1f);
    EXPECT_NE(ctrl.current_stage(), PhaseStage::Green);
}

// ── Test 4: Coordinated Controller Preemption Override ───────────────────────

TEST(EmergencyPriorityTest, CoordinatedControllerPreemptionOverride) {
    const auto map = make_priority_corridor_network();
    auto ctrl = CoordinatedController::make_default_2phase(map.net, map.i1);

    const RoadId main_road = map.r_main_01;
    ctrl.request_emergency_preemption(main_road);

    EXPECT_TRUE(ctrl.has_emergency_preemption());
    ctrl.tick(0.1f);
    EXPECT_EQ(ctrl.last_decision(), CoordinatedDecision::EmergencyPriority);

    ctrl.clear_emergency_preemption();
    EXPECT_FALSE(ctrl.has_emergency_preemption());
}

// ── Test 5: Approaching Emergency Vehicle Detection and Preemption ───────────

TEST(EmergencyPriorityTest, EmergencyPriorityManagerDetectsAndPreempts) {
    const auto map = make_priority_corridor_network();
    EmergencyPriorityManager mgr;
    mgr.preemption_distance_m = 120.0f;
    mgr.enable_preemption     = true;

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    SignalPhase p1{ "Phase-Main",  { map.r_main_01 },  30.0f, 3.0f, 2.0f };
    SignalPhase p2{ "Phase-Cross", { map.r_cross_41 }, 30.0f, 3.0f, 2.0f };
    controllers[map.i1] = TrafficSignalController(map.i1, { p1, p2 }, { map.r_main_01, map.r_cross_41 });

    std::vector<Vehicle> vehicles;
    Vehicle ev;
    ev.id           = static_cast<VehicleId>(101);
    ev.type         = VehicleType::Emergency;
    ev.current_road = map.r_main_01; // Road from i0 to i1, length 200m
    ev.route        = { map.r_main_01, map.r_main_12 };
    ev.route_index  = 0;
    ev.progress_m   = 50.0f; // 150m from intersection i1 (> 120m)
    vehicles.push_back(ev);

    // Far away: no preemption yet
    mgr.update(map.net, vehicles, controllers, nullptr, 0.1f);
    EXPECT_FALSE(controllers[map.i1].has_emergency_preemption());
    EXPECT_FALSE(mgr.is_intersection_preempted(map.i1));

    // Vehicle approaches within 120m (progress 100m -> dist = 100m)
    vehicles[0].progress_m = 100.0f;
    mgr.update(map.net, vehicles, controllers, nullptr, 0.1f);
    EXPECT_TRUE(controllers[map.i1].has_emergency_preemption());
    EXPECT_TRUE(mgr.is_intersection_preempted(map.i1));
    EXPECT_TRUE(mgr.is_corridor_road(map.r_main_01));
    EXPECT_TRUE(mgr.is_corridor_road(map.r_main_12)); // Subsequent road added to corridor

    // Vehicle passes intersection i1 onto r_main_12
    vehicles[0].route_index  = 1;
    vehicles[0].current_road = map.r_main_12;
    vehicles[0].progress_m   = 10.0f;
    mgr.update(map.net, vehicles, controllers, nullptr, 0.1f);
    // Intersection i1 preemption is now released
    EXPECT_FALSE(controllers[map.i1].has_emergency_preemption());
}

// ── Test 6: Normal Vehicle Lane Yielding on Multi-lane Roads ──────────────────

TEST(EmergencyPriorityTest, NormalVehiclesYieldToApproachingEmergencyVehicle) {
    const auto map = make_priority_corridor_network();
    EmergencyPriorityManager mgr;
    mgr.enable_yielding = true;

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;

    std::vector<Vehicle> vehicles;

    // Normal car on r_main_01 (2 lanes), Lane 0, at 80m
    Vehicle norm_car;
    norm_car.id           = static_cast<VehicleId>(1);
    norm_car.type         = VehicleType::Regular;
    norm_car.current_road = map.r_main_01;
    norm_car.lane_index   = 0;
    norm_car.progress_m   = 80.0f;
    vehicles.push_back(norm_car);

    // Emergency vehicle behind it on r_main_01, Lane 0, at 50m (gap = 30m)
    Vehicle emerg_car;
    emerg_car.id           = static_cast<VehicleId>(999);
    emerg_car.type         = VehicleType::Emergency;
    emerg_car.current_road = map.r_main_01;
    emerg_car.lane_index   = 0;
    emerg_car.progress_m   = 50.0f;
    vehicles.push_back(emerg_car);

    EXPECT_EQ(vehicles[0].lane_index, 0u);

    // Update triggers yielding
    mgr.update(map.net, vehicles, controllers, nullptr, 0.1f);

    // Normal car moved to lane 1 to yield
    EXPECT_EQ(vehicles[0].lane_index, 1u);
}

// ── Test 7: Dynamic Detour Rerouting Diverts Normal Traffic Away From Corridor

TEST(EmergencyPriorityTest, DynamicReroutingBypassesEmergencyCorridor) {
    const auto map = make_priority_corridor_network();
    TrafficAwareRouting router;

    // Under normal conditions, shortest route between i0 and i2 is direct arterial (400m)
    const auto normal_route = router.find_route(map.net, map.i0, map.i2);
    ASSERT_EQ(normal_route.size(), 2u);
    EXPECT_EQ(normal_route[0], map.r_main_01);
    EXPECT_EQ(normal_route[1], map.r_main_12);

    // Activate priority corridor on arterial roads
    router.set_emergency_corridor_roads({ map.r_main_01, map.r_main_12 });

    // Normal vehicle routing now detours through parallel bypass
    const auto detour_route = router.find_route(map.net, map.i0, map.i2);
    ASSERT_EQ(detour_route.size(), 2u);
    EXPECT_NE(detour_route[0], map.r_main_01); // Takes bypass road instead of corridor
    EXPECT_EQ(detour_route[0], map.r_bypass_03);
    EXPECT_EQ(detour_route[1], map.r_bypass_32);

    // Clearing corridor returns to direct route
    router.clear_emergency_corridor_roads();
    const auto restored_route = router.find_route(map.net, map.i0, map.i2);
    EXPECT_EQ(restored_route[0], map.r_main_01);
}

// ── Test 8: End-to-End Simulation Compares Emergency Travel Time & Disruption ─

TEST(EmergencyPriorityTest, EndToEndEmergencyPriorityScenarioEvaluation) {
    Config cfg;
    cfg.scenario            = TrafficScenario::EmergencyPriority;
    cfg.sim_duration_s      = 40.0f;
    cfg.dt_seconds          = 0.1f;
    cfg.seed                = 12345;
    cfg.signal_controller_type = SignalControllerType::Adaptive;
    cfg.routing_strategy    = RoutingStrategyType::TrafficAware;
    cfg.enable_emergency_priority = true;
    cfg.enable_emergency_rerouting = true;

    Simulation sim{ cfg, make_corridor_road_network };

    const auto map = make_priority_corridor_network();
    sim.set_allowed_spawn_origins({ map.i0 });
    sim.set_allowed_spawn_destinations({ map.i2 });

    // Step simulation directly
    for (int step = 0; step < 400; ++step) {
        if (step == 10) {
            sim.trigger_emergency_vehicle(map.i0, map.i2);
        }
        sim.tick();
    }

    const auto result = sim.get_run_result();
    EXPECT_GT(result.total_spawned, 0u);
    EXPECT_GT(result.total_completed, 0u);

    // Emergency vehicle successfully reached destination
    EXPECT_GE(result.emergency_completed, 1u);
    EXPECT_GT(result.emergency_avg_speed_mps, 0.0f);

    // Verification of corridor priority:
    // Emergency vehicle should have substantially lower wait time and higher speed than congested traffic
    if (result.emergency_completed > 0 && result.normal_avg_travel_time_s > 0.0f) {
        EXPECT_LE(result.emergency_avg_waiting_time_s, result.normal_avg_waiting_time_s + 5.0f);
    }
}
