#include <gtest/gtest.h>

#include "syntraq/signals/signal_phase.h"
#include "syntraq/signals/traffic_light.h"
#include "syntraq/signals/traffic_signal_controller.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/vehicles/vehicle_movement_system.h"
#include "syntraq/world/road_network.h"

#include <cmath>

namespace syntraq {
namespace {

// Helper to construct a minimal 2-approach intersection for testing
struct TestSignalJunction {
    RoadNetwork    network;
    IntersectionId center{ kInvalidIntersectionId };
    IntersectionId north { kInvalidIntersectionId };
    IntersectionId east  { kInvalidIntersectionId };
    RoadId         road_ns{ kInvalidRoadId }; // North -> Center
    RoadId         road_ew{ kInvalidRoadId }; // East -> Center

    static TestSignalJunction create() {
        TestSignalJunction j;
        j.center  = j.network.add_intersection({ 100.f, 100.f }, "Center");
        j.north   = j.network.add_intersection({ 100.f,   0.f }, "North");
        j.east    = j.network.add_intersection({ 200.f, 100.f }, "East");

        j.road_ns = j.network.add_road(j.north, j.center, 100.f, 13.89f, 1, "North->Center");
        j.road_ew = j.network.add_road(j.east,  j.center, 100.f, 13.89f, 1, "East->Center");
        return j;
    }
};

// ── 1. Signal State Transitions ──────────────────────────────────────────────

TEST(SignalStateTransitionsTest, FollowsGreenYellowAllRedCycle) {
    auto j = TestSignalJunction::create();

    SignalPhase p1{ "Phase-NS", { j.road_ns }, 10.0f, 3.0f, 2.0f };
    SignalPhase p2{ "Phase-EW", { j.road_ew }, 12.0f, 4.0f, 1.0f };

    TrafficSignalController controller(j.center, { p1, p2 }, { j.road_ns, j.road_ew });

    // Initial state: Phase 0, Green
    EXPECT_EQ(controller.current_phase_index(), 0u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_TRUE(controller.can_proceed(j.road_ns));
    EXPECT_FALSE(controller.can_proceed(j.road_ew));
    EXPECT_EQ(controller.signal_for_road(j.road_ns), SignalColor::Green);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Red);

    // Advance 10.0s -> Yellow
    controller.tick(10.0f);
    EXPECT_EQ(controller.current_phase_index(), 0u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(controller.signal_for_road(j.road_ns), SignalColor::Yellow);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Red);
    EXPECT_FALSE(controller.can_proceed(j.road_ns));

    // Advance 3.0s -> AllRed
    controller.tick(3.0f);
    EXPECT_EQ(controller.current_phase_index(), 0u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::AllRed);
    EXPECT_EQ(controller.signal_for_road(j.road_ns), SignalColor::Red);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Red);
    EXPECT_FALSE(controller.can_proceed(j.road_ns));
    EXPECT_FALSE(controller.can_proceed(j.road_ew));

    // Advance 2.0s -> Phase 1, Green
    controller.tick(2.0f);
    EXPECT_EQ(controller.current_phase_index(), 1u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_EQ(controller.signal_for_road(j.road_ns), SignalColor::Red);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Green);
    EXPECT_TRUE(controller.can_proceed(j.road_ew));
    EXPECT_FALSE(controller.can_proceed(j.road_ns));

    // Advance 12.0s -> Phase 1, Yellow
    controller.tick(12.0f);
    EXPECT_EQ(controller.current_phase_index(), 1u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Yellow);

    // Advance 4.0s -> Phase 1, AllRed
    controller.tick(4.0f);
    EXPECT_EQ(controller.current_phase_index(), 1u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::AllRed);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Red);

    // Advance 1.0s -> Cycles back to Phase 0, Green
    controller.tick(1.0f);
    EXPECT_EQ(controller.current_phase_index(), 0u);
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_EQ(controller.signal_for_road(j.road_ns), SignalColor::Green);
    EXPECT_EQ(controller.signal_for_road(j.road_ew), SignalColor::Red);
}

// ── 2. Phase Timing Accuracy & Invariance ────────────────────────────────────

TEST(SignalTimingTest, AccurateTimestepAccumulation) {
    auto j = TestSignalJunction::create();

    SignalPhase p1{ "Phase-NS", { j.road_ns }, 5.0f, 2.0f, 1.0f };
    TrafficSignalController controller(j.center, { p1 }, { j.road_ns });

    EXPECT_FLOAT_EQ(controller.cycle_duration_s(), 8.0f);

    constexpr float dt = 0.1f;
    // Tick 49 times = 4.9s (still in Green)
    for (int i = 0; i < 49; ++i) {
        controller.tick(dt);
    }
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_NEAR(controller.stage_remaining_s(), 0.1f, 1e-4f);

    // 50th tick = 5.0s -> Yellow
    controller.tick(dt);
    EXPECT_EQ(controller.current_stage(), PhaseStage::Yellow);

    // Tick 20 times = 2.0s -> AllRed
    for (int i = 0; i < 20; ++i) {
        controller.tick(dt);
    }
    EXPECT_EQ(controller.current_stage(), PhaseStage::AllRed);

    // Tick 10 times = 1.0s -> Green again
    for (int i = 0; i < 10; ++i) {
        controller.tick(dt);
    }
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
}

TEST(SignalTimingTest, DeterministicSubTickEquivalence) {
    auto j = TestSignalJunction::create();

    SignalPhase p1{ "Phase-NS", { j.road_ns }, 10.0f, 3.0f, 2.0f };
    SignalPhase p2{ "Phase-EW", { j.road_ew }, 10.0f, 3.0f, 2.0f };

    TrafficSignalController c1(j.center, { p1, p2 }, { j.road_ns, j.road_ew });
    TrafficSignalController c2(j.center, { p1, p2 }, { j.road_ns, j.road_ew });

    // c1 updated by 0.1s steps (150 ticks = 15.0s)
    for (int i = 0; i < 150; ++i) {
        c1.tick(0.1f);
    }

    // c2 updated by 0.05s steps (300 ticks = 15.0s)
    for (int i = 0; i < 300; ++i) {
        c2.tick(0.05f);
    }

    EXPECT_EQ(c1.current_phase_index(), c2.current_phase_index());
    EXPECT_EQ(c1.current_stage(), c2.current_stage());
    EXPECT_NEAR(c1.stage_elapsed_s(), c2.stage_elapsed_s(), 1e-4f);
    EXPECT_EQ(c1.signal_for_road(j.road_ns), c2.signal_for_road(j.road_ns));
    EXPECT_EQ(c1.signal_for_road(j.road_ew), c2.signal_for_road(j.road_ew));
}

// ── 3. Invalid Configurations ────────────────────────────────────────────────

TEST(SignalConfigurationTest, RejectsInvalidConfigurations) {
    std::string err;

    // 1. Invalid intersection ID
    TrafficSignalController c_bad_id(kInvalidIntersectionId, { SignalPhase{ "P", { RoadId{1} }, 10.f, 3.f, 2.f } });
    EXPECT_FALSE(c_bad_id.is_valid());
    EXPECT_FALSE(c_bad_id.validate(&err));
    EXPECT_FALSE(err.empty());

    // 2. Empty phases
    TrafficSignalController c_no_phases(IntersectionId{1}, {});
    EXPECT_FALSE(c_no_phases.is_valid());
    EXPECT_FALSE(c_no_phases.validate(&err));

    // 3. Phase with empty green roads
    SignalPhase p_empty_roads{ "P", {}, 10.0f, 3.0f, 2.0f };
    TrafficSignalController c_empty_roads(IntersectionId{1}, { p_empty_roads });
    EXPECT_FALSE(c_empty_roads.is_valid());
    EXPECT_FALSE(c_empty_roads.validate(&err));

    // 4. Phase with zero green duration
    SignalPhase p_zero_green{ "P", { RoadId{1} }, 0.0f, 3.0f, 2.0f };
    TrafficSignalController c_zero_green(IntersectionId{1}, { p_zero_green });
    EXPECT_FALSE(c_zero_green.is_valid());

    // 5. Phase with negative green duration
    SignalPhase p_neg_green{ "P", { RoadId{1} }, -5.0f, 3.0f, 2.0f };
    TrafficSignalController c_neg_green(IntersectionId{1}, { p_neg_green });
    EXPECT_FALSE(c_neg_green.is_valid());

    // 6. Phase with negative yellow duration
    SignalPhase p_neg_yellow{ "P", { RoadId{1} }, 10.0f, -2.0f, 2.0f };
    TrafficSignalController c_neg_yellow(IntersectionId{1}, { p_neg_yellow });
    EXPECT_FALSE(c_neg_yellow.is_valid());

    // 7. Phase with negative all-red duration
    SignalPhase p_neg_allred{ "P", { RoadId{1} }, 10.0f, 3.0f, -1.0f };
    TrafficSignalController c_neg_allred(IntersectionId{1}, { p_neg_allred });
    EXPECT_FALSE(c_neg_allred.is_valid());
}

TEST(SignalConfigurationTest, ReconfigurableTimings) {
    auto j = TestSignalJunction::create();
    SignalPhase p1{ "Phase-NS", { j.road_ns }, 10.0f, 3.0f, 2.0f };
    TrafficSignalController controller(j.center, { p1 }, { j.road_ns });

    EXPECT_FLOAT_EQ(controller.current_stage_duration(), 10.0f);

    // Reconfigure timings
    controller.set_phase_timings(0, 15.0f, 4.0f, 2.5f);
    EXPECT_FLOAT_EQ(controller.current_stage_duration(), 15.0f);
    EXPECT_FLOAT_EQ(controller.cycle_duration_s(), 21.5f);
}

// ── 4. Vehicle Response to Red/Green Signals ─────────────────────────────────

TEST(VehicleSignalResponseTest, VehicleStopsAtRedSignal) {
    auto j = TestSignalJunction::create();

    // Signal controller where NS road is RED
    SignalPhase p_ew{ "Phase-EW", { j.road_ew }, 10.0f, 3.0f, 2.0f };
    TrafficSignalController controller(j.center, { p_ew }, { j.road_ns, j.road_ew });
    EXPECT_FALSE(controller.can_proceed(j.road_ns));

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[j.center] = controller;

    // Create a vehicle travelling along road_ns toward center
    Vehicle v;
    v.id           = VehicleId{ 1 };
    v.route        = { j.road_ns };
    v.route_index  = 0;
    v.current_road = j.road_ns;
    v.progress_m   = 50.0f; // 50m along 100m road
    v.speed_mps    = 13.89f;
    v.destination  = j.center;
    v.state        = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    VehicleMovementSystem movement;

    // Update for 10.0 seconds (100 ticks of 0.1s)
    for (int i = 0; i < 100; ++i) {
        movement.update(vehicles, j.network, controllers, 0.1f);
    }

    // Vehicle should have decelerated and stopped before the stop line (100m - 2m = 98m)
    ASSERT_EQ(vehicles.size(), 1u);
    const auto& updated_v = vehicles[0];
    EXPECT_EQ(updated_v.state, VehicleState::Stopped);
    EXPECT_FLOAT_EQ(updated_v.speed_mps, 0.0f);
    EXPECT_LE(updated_v.progress_m, 98.0f);
    EXPECT_GT(updated_v.progress_m, 90.0f); // Reached the vicinity of the stop line
}

TEST(VehicleSignalResponseTest, VehicleRemainsStoppedWhileSignalStaysRed) {
    auto j = TestSignalJunction::create();

    SignalPhase p_ew{ "Phase-EW", { j.road_ew }, 20.0f, 3.0f, 2.0f };
    TrafficSignalController controller(j.center, { p_ew }, { j.road_ns, j.road_ew });
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[j.center] = controller;

    // Place vehicle already stopped at the stop line (98m)
    Vehicle v;
    v.id           = VehicleId{ 1 };
    v.route        = { j.road_ns };
    v.current_road = j.road_ns;
    v.progress_m   = 98.0f;
    v.speed_mps    = 0.0f;
    v.destination  = j.center;
    v.state        = VehicleState::Stopped;

    std::vector<Vehicle> vehicles{ v };
    VehicleMovementSystem movement;

    // Run for 5 seconds while red
    for (int i = 0; i < 50; ++i) {
        movement.update(vehicles, j.network, controllers, 0.1f);
    }

    EXPECT_EQ(vehicles[0].state, VehicleState::Stopped);
    EXPECT_FLOAT_EQ(vehicles[0].speed_mps, 0.0f);
    EXPECT_FLOAT_EQ(vehicles[0].progress_m, 98.0f);
}

TEST(VehicleSignalResponseTest, VehicleResumesAndCrossesWhenSignalTurnsGreen) {
    auto j = TestSignalJunction::create();

    // Add continuation road from center -> south
    auto south = j.network.add_intersection({ 100.f, 200.f }, "South");
    auto road_continuation = j.network.add_road(j.center, south, 100.f, 13.89f, 1, "Center->South");

    // Start with RED on road_ns
    SignalPhase p_ew{ "Phase-EW", { j.road_ew }, 10.0f, 3.0f, 2.0f };
    SignalPhase p_ns{ "Phase-NS", { j.road_ns }, 10.0f, 3.0f, 2.0f };
    TrafficSignalController controller(j.center, { p_ew, p_ns }, { j.road_ns, j.road_ew });

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[j.center] = controller;

    // Vehicle stopped at stop line on road_ns
    Vehicle v;
    v.id           = VehicleId{ 1 };
    v.route        = { j.road_ns, road_continuation };
    v.current_road = j.road_ns;
    v.progress_m   = 98.0f;
    v.speed_mps    = 0.0f;
    v.destination  = south;
    v.state        = VehicleState::Stopped;

    std::vector<Vehicle> vehicles{ v };
    VehicleMovementSystem movement;

    // 1. Tick while red: remains stopped
    movement.update(vehicles, j.network, controllers, 0.1f);
    EXPECT_EQ(vehicles[0].state, VehicleState::Stopped);

    // 2. Change controller to Phase-NS (Green for road_ns)
    controllers[j.center] = TrafficSignalController(j.center, { p_ns, p_ew }, { j.road_ns, j.road_ew });
    EXPECT_TRUE(controllers[j.center].can_proceed(j.road_ns));

    // 3. Tick with green: vehicle accelerates and transitions across intersection to continuation road
    for (int i = 0; i < 30; ++i) { // 3.0 seconds
        movement.update(vehicles, j.network, controllers, 0.1f);
    }

    ASSERT_EQ(vehicles.size(), 1u);
    EXPECT_EQ(vehicles[0].state, VehicleState::Moving);
    EXPECT_GT(vehicles[0].speed_mps, 0.0f);
    // Should now be on road_continuation
    EXPECT_EQ(vehicles[0].current_road, road_continuation);
    EXPECT_EQ(vehicles[0].route_index, 1u);
}

TEST(VehicleSignalResponseTest, VehiclePassesThroughGreenSignalWithoutStopping) {
    auto j = TestSignalJunction::create();

    auto south = j.network.add_intersection({ 100.f, 200.f }, "South");
    auto road_continuation = j.network.add_road(j.center, south, 100.f, 13.89f, 1, "Center->South");

    // NS road has GREEN
    SignalPhase p_ns{ "Phase-NS", { j.road_ns }, 20.0f, 3.0f, 2.0f };
    TrafficSignalController controller(j.center, { p_ns }, { j.road_ns, j.road_ew });
    EXPECT_TRUE(controller.can_proceed(j.road_ns));

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    controllers[j.center] = controller;

    Vehicle v;
    v.id           = VehicleId{ 1 };
    v.route        = { j.road_ns, road_continuation };
    v.current_road = j.road_ns;
    v.progress_m   = 80.0f;
    v.speed_mps    = 13.89f;
    v.destination  = south;
    v.state        = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    VehicleMovementSystem movement;

    // Run for 2.0s -> advances through intersection without stopping
    for (int i = 0; i < 20; ++i) {
        movement.update(vehicles, j.network, controllers, 0.1f);
        EXPECT_NE(vehicles[0].state, VehicleState::Stopped);
    }

    EXPECT_EQ(vehicles[0].current_road, road_continuation);
}

// ── 5. Simulation Integration & Reset ────────────────────────────────────────

TEST(SimulationSignalsIntegrationTest, DefaultGridInitializesSignalControllers) {
    Config cfg;
    cfg.enable_traffic_signals = true;
    Simulation sim{ cfg };

    // The 4x3 test grid has 12 intersections.
    // Intersections with > 1 incoming roads receive a baseline controller.
    const auto& controllers = sim.signal_controllers();
    EXPECT_FALSE(controllers.empty());

    // Center intersections on the 4x3 grid have 4 incoming roads
    for (const auto& [id, controller] : controllers) {
        EXPECT_TRUE(controller.is_valid());
        EXPECT_GE(controller.phase_count(), 1u);
    }
}

TEST(SimulationSignalsIntegrationTest, ResetRestoresSignalState) {
    Config cfg;
    Simulation sim{ cfg };

    // Advance 25 seconds
    sim.run_for(25.0f);

    // Reset restores t=0 and initial signal phase
    sim.reset();
    EXPECT_FLOAT_EQ(sim.state().elapsed_s, 0.0f);
    EXPECT_EQ(sim.state().tick, 0u);

    for (const auto& [id, controller] : sim.signal_controllers()) {
        EXPECT_EQ(controller.current_phase_index(), 0u);
        EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
        EXPECT_FLOAT_EQ(controller.stage_elapsed_s(), 0.0f);
    }
}

} // namespace
} // namespace syntraq
