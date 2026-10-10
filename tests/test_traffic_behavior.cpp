//
// tests/test_traffic_behavior.cpp
//
// Milestone 6 — Traffic Behavior Tests
// Covers:
//   - Stopping distance and following distance
//   - Car-following behavior and collision avoidance
//   - Intersection queue formation and dissipation
//   - Lane occupancy checks (at spawner and road transitions)
//   - Turn behavior (speed reduction on sharp turns)
//   - Repeatable traffic scenarios (Low, Medium, High, Rush Hour)
//   - Deterministic seed repeatability
//   - Fixed-time signal controller as baseline
//

#include "syntraq/core/config.h"
#include "syntraq/signals/traffic_signal_controller.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/vehicles/vehicle_movement_system.h"
#include "syntraq/vehicles/vehicle_spawner.h"
#include "syntraq/world/road_network.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace syntraq {
namespace {

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Create a straight 2-way road track between A and B
struct StraightTestTrack {
    RoadNetwork    network;
    IntersectionId a{ kInvalidIntersectionId };
    IntersectionId b{ kInvalidIntersectionId };
    RoadId         road{ kInvalidRoadId };
    RoadId         return_road{ kInvalidRoadId };

    static StraightTestTrack create(float length_m = 200.0f, float speed_limit_mps = 13.89f, uint32_t lanes = 1) {
        StraightTestTrack t;
        t.a           = t.network.add_intersection({ 0.0f, 0.0f }, "A");
        t.b           = t.network.add_intersection({ length_m, 0.0f }, "B");
        t.road        = t.network.add_road(t.a, t.b, length_m, speed_limit_mps, lanes, "Track");
        t.return_road = t.network.add_road(t.b, t.a, length_m, speed_limit_mps, lanes, "ReturnTrack");
        return t;
    }
};

/// Create a 90-degree corner junction: A -> B -> C
/// A = (0, 0), B = (100, 0), C = (100, 100) -> 90 degree right turn at B
struct CornerJunction {
    RoadNetwork    network;
    IntersectionId a{ kInvalidIntersectionId };
    IntersectionId b{ kInvalidIntersectionId };
    IntersectionId c{ kInvalidIntersectionId };
    RoadId         road_ab{ kInvalidRoadId };
    RoadId         road_bc{ kInvalidRoadId };

    static CornerJunction create() {
        CornerJunction j;
        j.a = j.network.add_intersection({   0.0f,   0.0f }, "A");
        j.b = j.network.add_intersection({ 100.0f,   0.0f }, "B");
        j.c = j.network.add_intersection({ 100.0f, 100.0f }, "C");
        j.road_ab = j.network.add_road(j.a, j.b, 100.0f, 13.89f, 1, "AB");
        j.road_bc = j.network.add_road(j.b, j.c, 100.0f, 13.89f, 1, "BC");
        return j;
    }
};

Vehicle make_test_vehicle(VehicleId id, RoadId road, float progress_m, float speed_mps, uint32_t lane = 0) {
    Vehicle v;
    v.id           = id;
    v.route        = { road };
    v.route_index  = 0;
    v.current_road = road;
    v.lane_index   = lane;
    v.progress_m   = progress_m;
    v.speed_mps    = speed_mps;
    v.max_speed_mps = 13.89f;
    v.accel_mps2   = 3.0f;
    v.decel_mps2   = 5.0f;
    v.length_m     = 4.5f;
    v.min_gap_m    = 2.5f;
    v.time_headway_s = 1.2f;
    v.state        = VehicleState::Moving;
    return v;
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Car-Following Behavior & Collision Avoidance
// ═════════════════════════════════════════════════════════════════════════════

TEST(TrafficBehaviorTest, CarFollowingMaintainsSafeGapAndMatchesSpeed) {
    auto track = StraightTestTrack::create(300.0f, 15.0f);
    VehicleMovementSystem movement;

    // Leader moving at 8.0 m/s at 80m
    auto leader = make_test_vehicle(VehicleId{ 0 }, track.road, 80.0f, 8.0f);
    leader.max_speed_mps = 8.0f; // Limit leader cruise speed

    // Follower moving at 15.0 m/s at 20m
    auto follower = make_test_vehicle(VehicleId{ 1 }, track.road, 20.0f, 15.0f);

    std::vector<Vehicle> vehicles = { leader, follower };

    // Run for 15 seconds (150 ticks of 0.1s)
    for (int t = 0; t < 150; ++t) {
        movement.update(vehicles, track.network, 0.1f);

        // Follower front bumper must NEVER penetrate leader rear bumper
        const float leader_rear = vehicles[0].progress_m - vehicles[0].length_m;
        EXPECT_LE(vehicles[1].progress_m, leader_rear);

        // Check buffer is respected
        const float gap = leader_rear - vehicles[1].progress_m;
        EXPECT_GE(gap, 0.0f);
    }

    // Steady state check: follower should match leader speed closely
    EXPECT_NEAR(vehicles[1].speed_mps, vehicles[0].speed_mps, 0.5f);
}

TEST(TrafficBehaviorTest, CollisionAvoidanceEmergencyBrakingBehindSuddenStop) {
    auto track = StraightTestTrack::create(200.0f, 13.89f);
    VehicleMovementSystem movement;

    // Leader is stopped at 90m
    auto leader = make_test_vehicle(VehicleId{ 0 }, track.road, 90.0f, 0.0f);
    leader.state = VehicleState::Stopped;
    leader.accel_mps2 = 0.0f; // Remain stopped

    // Follower approaches at high speed (13.89 m/s) from 40m
    auto follower = make_test_vehicle(VehicleId{ 1 }, track.road, 40.0f, 13.89f);

    std::vector<Vehicle> vehicles = { leader, follower };

    for (int t = 0; t < 100; ++t) {
        movement.update(vehicles, track.network, 0.1f);

        const float leader_rear = vehicles[0].progress_m - vehicles[0].length_m;
        EXPECT_LE(vehicles[1].progress_m, leader_rear);
    }

    // Follower must have come to a complete stop behind leader
    EXPECT_EQ(vehicles[1].state, VehicleState::Stopped);
    EXPECT_FLOAT_EQ(vehicles[1].speed_mps, 0.0f);

    const float leader_rear = vehicles[0].progress_m - vehicles[0].length_m;
    const float final_gap = leader_rear - vehicles[1].progress_m;
    EXPECT_GE(final_gap, vehicles[1].min_gap_m - 0.1f);
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Intersection Queue Formation & Orderly Dissipation
// ═════════════════════════════════════════════════════════════════════════════

TEST(TrafficBehaviorTest, TwoVehicleQueueFormsAtRedSignal) {
    auto track = StraightTestTrack::create(100.0f, 13.89f);
    VehicleMovementSystem movement;

    // Red signal at intersection B
    SignalPhase red_phase{ "All-Red", {}, 30.0f, 0.0f, 0.0f };
    TrafficSignalController controller(track.b, { red_phase }, { track.road });
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[track.b] = controller;

    // Two vehicles approaching red signal
    auto v0 = make_test_vehicle(VehicleId{ 0 }, track.road, 60.0f, 10.0f);
    auto v1 = make_test_vehicle(VehicleId{ 1 }, track.road, 30.0f, 10.0f);

    std::vector<Vehicle> vehicles = { v0, v1 };

    // Run for 12 seconds
    for (int t = 0; t < 120; ++t) {
        movement.update(vehicles, track.network, controllers, 0.1f);
    }

    const float stop_line_m = 100.0f - 2.0f; // 98.0m

    // Both vehicles must be stopped
    EXPECT_EQ(vehicles[0].state, VehicleState::Stopped);
    EXPECT_EQ(vehicles[1].state, VehicleState::Stopped);
    EXPECT_FLOAT_EQ(vehicles[0].speed_mps, 0.0f);
    EXPECT_FLOAT_EQ(vehicles[1].speed_mps, 0.0f);

    // Lead vehicle stopped at stop line
    EXPECT_NEAR(vehicles[0].progress_m, stop_line_m, 0.2f);

    // Second vehicle stopped behind lead vehicle
    const float expected_v1_pos = stop_line_m - vehicles[0].length_m - vehicles[1].min_gap_m; // 98 - 4.5 - 2.5 = 91.0m
    EXPECT_NEAR(vehicles[1].progress_m, expected_v1_pos, 0.6f);

    // No overlap
    const float v0_rear = vehicles[0].progress_m - vehicles[0].length_m;
    EXPECT_LE(vehicles[1].progress_m, v0_rear);
}

TEST(TrafficBehaviorTest, MultiVehicleQueueFormsOrderedStack) {
    auto track = StraightTestTrack::create(120.0f, 13.89f);
    VehicleMovementSystem movement;

    // Red signal at end
    SignalPhase red_phase{ "All-Red", {}, 30.0f, 0.0f, 0.0f };
    TrafficSignalController controller(track.b, { red_phase }, { track.road });
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[track.b] = controller;

    // 4 vehicles queued behind one another
    std::vector<Vehicle> vehicles;
    for (uint32_t i = 0; i < 4; ++i) {
        vehicles.push_back(make_test_vehicle(VehicleId{ i }, track.road, 70.0f - static_cast<float>(i) * 18.0f, 8.0f));
    }

    // Run for 15 seconds
    for (int t = 0; t < 150; ++t) {
        movement.update(vehicles, track.network, controllers, 0.1f);
    }

    // All 4 vehicles must be stopped with strictly decreasing positions
    for (size_t i = 0; i < vehicles.size(); ++i) {
        EXPECT_EQ(vehicles[i].state, VehicleState::Stopped);
        EXPECT_FLOAT_EQ(vehicles[i].speed_mps, 0.0f);

        if (i > 0) {
            // Strictly behind previous vehicle with safe gap
            const float leader_rear = vehicles[i - 1].progress_m - vehicles[i - 1].length_m;
            EXPECT_LE(vehicles[i].progress_m, leader_rear);
            const float gap = leader_rear - vehicles[i].progress_m;
            EXPECT_GE(gap, vehicles[i].min_gap_m - 0.2f);
        }
    }
}

TEST(TrafficBehaviorTest, QueueDissipatesWhenSignalTurnsGreen) {
    auto track = StraightTestTrack::create(100.0f, 13.89f);
    auto c = track.network.add_intersection({ 200.0f, 0.0f }, "C");
    auto road_bc = track.network.add_road(track.b, c, 100.0f, 13.89f, 1, "TrackBC");
    VehicleMovementSystem movement;

    // Start with red signal
    SignalPhase red_phase{ "Red", {}, 10.0f, 0.0f, 0.0f };
    SignalPhase green_phase{ "Green", { track.road }, 20.0f, 0.0f, 0.0f };
    TrafficSignalController controller(track.b, { red_phase, green_phase }, { track.road });
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[track.b] = controller;

    // Two vehicles in queue with 2-road route
    auto v0 = make_test_vehicle(VehicleId{ 0 }, track.road, 98.0f, 0.0f);
    v0.route = { track.road, road_bc };
    v0.destination = c;
    v0.state = VehicleState::Stopped;

    auto v1 = make_test_vehicle(VehicleId{ 1 }, track.road, 91.0f, 0.0f);
    v1.route = { track.road, road_bc };
    v1.destination = c;
    v1.state = VehicleState::Stopped;

    std::vector<Vehicle> vehicles = { v0, v1 };

    // 1. Tick while RED: both remain stopped
    movement.update(vehicles, track.network, controllers, 0.1f);
    EXPECT_EQ(vehicles[0].state, VehicleState::Stopped);
    EXPECT_EQ(vehicles[1].state, VehicleState::Stopped);

    // 2. Signal changes to GREEN
    controllers[track.b] = TrafficSignalController(track.b, { green_phase }, { track.road });

    // 3. Update with green for 1.5 seconds (15 ticks)
    for (int t = 0; t < 15; ++t) {
        movement.update(vehicles, track.network, controllers, 0.1f);
    }

    // Both vehicles must now be moving forward
    EXPECT_EQ(vehicles[0].state, VehicleState::Moving);
    EXPECT_GT(vehicles[0].speed_mps, 0.0f);

    EXPECT_EQ(vehicles[1].state, VehicleState::Moving);
    EXPECT_GT(vehicles[1].speed_mps, 0.0f);
    EXPECT_GT(vehicles[1].progress_m, 91.0f);
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Lane Occupancy Checks
// ═════════════════════════════════════════════════════════════════════════════

TEST(TrafficBehaviorTest, SpawnerRejectsSpawnWhenEntryLaneIsOccupied) {
    auto track = StraightTestTrack::create(100.0f, 13.89f, /* lanes = */ 1);
    VehicleSpawner spawner(/* seed = */ 42, /* spawn_interval_s = */ 1.0f, /* max = */ 10);

    // Vehicle currently occupying the entrance buffer of road (progress = 2.0m)
    auto blocking_vehicle = make_test_vehicle(VehicleId{ 0 }, track.road, 2.0f, 0.0f);
    std::vector<Vehicle> existing = { blocking_vehicle };

    // Try to spawn: should be rejected due to lane occupancy
    auto spawned = spawner.try_spawn(track.network, existing, 1.5f);
    EXPECT_FALSE(spawned.has_value());

    // Once blocking vehicle moves beyond entrance buffer (progress = 25.0m):
    existing[0].progress_m = 25.0f;
    spawned = spawner.try_spawn(track.network, existing, 2.0f);
    EXPECT_TRUE(spawned.has_value());
}

TEST(TrafficBehaviorTest, RoadTransitionBlocksWhenNextRoadEntranceIsOccupied) {
    auto corner = CornerJunction::create();
    VehicleMovementSystem movement;

    // Vehicle 1 on road_ab approaching intersection B
    auto v_entering = make_test_vehicle(VehicleId{ 1 }, corner.road_ab, 97.0f, 5.0f);
    v_entering.route = { corner.road_ab, corner.road_bc };
    v_entering.destination = corner.c;

    // Vehicle 0 stopped right at the start of road_bc (blocking entry)
    auto v_blocking = make_test_vehicle(VehicleId{ 0 }, corner.road_bc, 2.0f, 0.0f);
    v_blocking.state = VehicleState::Stopped;
    v_blocking.accel_mps2 = 0.0f;

    std::vector<Vehicle> vehicles = { v_blocking, v_entering };

    // Update simulation
    for (int t = 0; t < 20; ++t) {
        movement.update(vehicles, corner.network, 0.1f);
    }

    // v_entering must hold at road_ab stop line (98m) and NOT enter road_bc while entrance is blocked
    EXPECT_EQ(vehicles[1].current_road, corner.road_ab);
    EXPECT_LE(vehicles[1].progress_m, 98.0f);
    EXPECT_EQ(vehicles[1].state, VehicleState::Stopped);
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Turn Behavior (Speed Reduction)
// ═════════════════════════════════════════════════════════════════════════════

TEST(TrafficBehaviorTest, VehicleReducesSpeedBeforeSharpPerpendicularTurn) {
    auto corner = CornerJunction::create();
    VehicleMovementSystem movement;

    // Fast vehicle approaching 90-degree turn
    auto v = make_test_vehicle(VehicleId{ 0 }, corner.road_ab, 70.0f, 13.89f);
    v.route = { corner.road_ab, corner.road_bc };
    v.destination = corner.c;

    std::vector<Vehicle> vehicles = { v };

    float min_speed_near_turn = 13.89f;
    for (int t = 0; t < 35; ++t) {
        movement.update(vehicles, corner.network, 0.1f);
        if (vehicles[0].current_road == corner.road_ab && vehicles[0].progress_m >= 95.0f) {
            min_speed_near_turn = std::min(min_speed_near_turn, vehicles[0].speed_mps);
        } else if (vehicles[0].current_road == corner.road_bc) {
            min_speed_near_turn = std::min(min_speed_near_turn, vehicles[0].speed_mps);
        }
    }

    // Approaching / negotiating 90 deg turn: speed must be significantly reduced to turn speed limit (<= 5.0 m/s)
    EXPECT_LE(min_speed_near_turn, 5.0f);
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Repeatable Traffic Scenarios & Presets
// ═════════════════════════════════════════════════════════════════════════════

TEST(TrafficBehaviorTest, ScenarioPresetsProvideCorrectParameters) {
    const auto low  = get_scenario_preset(TrafficScenario::Low, 100);
    const auto med  = get_scenario_preset(TrafficScenario::Medium, 100);
    const auto high = get_scenario_preset(TrafficScenario::High, 100);
    const auto rush = get_scenario_preset(TrafficScenario::RushHour, 100);

    EXPECT_GT(low.spawn_interval_s, med.spawn_interval_s);
    EXPECT_GT(med.spawn_interval_s, high.spawn_interval_s);
    EXPECT_GT(high.spawn_interval_s, rush.spawn_interval_s);

    EXPECT_LT(low.max_vehicles, med.max_vehicles);
    EXPECT_LT(med.max_vehicles, high.max_vehicles);
    EXPECT_LT(high.max_vehicles, rush.max_vehicles);

    EXPECT_EQ(low.seed, 100u);
    EXPECT_EQ(rush.seed, 100u);
}

TEST(TrafficBehaviorTest, RepeatableScenarioProducesBitIdenticalSimulationState) {
    Config cfg1;
    cfg1.scenario                 = TrafficScenario::High;
    cfg1.seed                     = 9999;
    cfg1.enable_traffic_signals   = true;
    cfg1.headless                 = true;

    Config cfg2 = cfg1;

    Simulation sim1{ cfg1 };
    Simulation sim2{ cfg2 };

    sim1.set_scenario(TrafficScenario::High, 9999);
    sim2.set_scenario(TrafficScenario::High, 9999);

    // Run both simulations for 25 seconds
    sim1.run_for(25.0f);
    sim2.run_for(25.0f);

    // Exact state match
    EXPECT_EQ(sim1.state().tick, sim2.state().tick);
    EXPECT_FLOAT_EQ(sim1.state().elapsed_s, sim2.state().elapsed_s);
    EXPECT_EQ(sim1.state().total_spawned, sim2.state().total_spawned);
    EXPECT_EQ(sim1.state().total_arrived, sim2.state().total_arrived);
    EXPECT_EQ(sim1.state().active_vehicles, sim2.state().active_vehicles);
    EXPECT_EQ(sim1.state().queued_vehicles, sim2.state().queued_vehicles);
    EXPECT_EQ(sim1.state().max_queue_len, sim2.state().max_queue_len);

    ASSERT_EQ(sim1.vehicles().size(), sim2.vehicles().size());
    for (size_t i = 0; i < sim1.vehicles().size(); ++i) {
        const auto& v1 = sim1.vehicles()[i];
        const auto& v2 = sim2.vehicles()[i];
        EXPECT_EQ(v1.id, v2.id);
        EXPECT_EQ(v1.current_road, v2.current_road);
        EXPECT_EQ(v1.lane_index, v2.lane_index);
        EXPECT_FLOAT_EQ(v1.progress_m, v2.progress_m);
        EXPECT_FLOAT_EQ(v1.speed_mps, v2.speed_mps);
        EXPECT_EQ(v1.state, v2.state);
    }
}

TEST(TrafficBehaviorTest, FixedTimeControllerBaselineClearsTrafficQueues) {
    Config cfg;
    cfg.scenario               = TrafficScenario::Medium;
    cfg.seed                   = 42;
    cfg.enable_traffic_signals = true;
    cfg.headless               = true;

    Simulation sim{ cfg };

    // Advance 40 seconds across multiple signal cycles
    sim.run_for(40.0f);

    // Baseline fixed-time controllers must be active
    EXPECT_FALSE(sim.signal_controllers().empty());

    // Both spawns and arrivals must have occurred
    EXPECT_GT(sim.state().total_spawned, 0u);
    EXPECT_GT(sim.state().total_arrived, 0u);
}

} // namespace
} // namespace syntraq
