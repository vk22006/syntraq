//
// tests/test_adaptive_controller.cpp
//
// Unit tests for AdaptiveController and FixedTimeController:
// - ITrafficSignalController interface polymorphism
// - Mandatory minimum green safety constraints
// - Demand-responsive green extension
// - Early gap-out phase shortening
// - Opposing approach starvation prevention
// - Max green duration enforcement
// - Deterministic explainability and telemetry tracking
//

#include <gtest/gtest.h>

#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/signals/fixed_time_controller.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"

#include <memory>
#include <vector>

namespace syntraq {
namespace {

struct TestJunction {
    IntersectionId center{ 1 };
    RoadId         road_ns{ 10 }; // North-South
    RoadId         road_ew{ 20 }; // East-West

    static TestJunction create() {
        return TestJunction{};
    }

    SignalPhase phase_ns(float green = 10.0f, float yellow = 3.0f, float all_red = 2.0f) const {
        return SignalPhase{ "Phase-NS", { road_ns }, green, yellow, all_red };
    }

    SignalPhase phase_ew(float green = 10.0f, float yellow = 3.0f, float all_red = 2.0f) const {
        return SignalPhase{ "Phase-EW", { road_ew }, green, yellow, all_red };
    }
};

// ── Test 1: Interface Compliance & Polymorphism ──────────────────────────────

TEST(AdaptiveControllerTest, BothControllersImplementSameInterface) {
    auto j = TestJunction::create();

    std::unique_ptr<ITrafficSignalController> fixed =
        std::make_unique<FixedTimeController>(j.center, std::vector<SignalPhase>{ j.phase_ns(), j.phase_ew() });
    std::unique_ptr<ITrafficSignalController> adaptive =
        std::make_unique<AdaptiveController>(j.center, std::vector<SignalPhase>{ j.phase_ns(), j.phase_ew() });

    EXPECT_EQ(fixed->type(), SignalControllerType::FixedTime);
    EXPECT_EQ(fixed->type_name(), "FixedTime");
    EXPECT_EQ(adaptive->type(), SignalControllerType::Adaptive);
    EXPECT_EQ(adaptive->type_name(), "Adaptive");

    EXPECT_EQ(fixed->intersection_id(), j.center);
    EXPECT_EQ(adaptive->intersection_id(), j.center);
    EXPECT_EQ(fixed->phase_count(), 2u);
    EXPECT_EQ(adaptive->phase_count(), 2u);
}

// ── Test 2: Minimum Green Safety Constraint ──────────────────────────────────

TEST(AdaptiveControllerTest, MinimumGreenSafetyIsStrictlyEnforced) {
    auto j = TestJunction::create();
    AdaptiveControllerConfig cfg;
    cfg.min_green_s     = 4.0f;
    cfg.nominal_green_s = 10.0f;
    cfg.max_green_s     = 25.0f;

    AdaptiveController controller(
        j.center,
        { j.phase_ns(), j.phase_ew() },
        { j.road_ns, j.road_ew },
        cfg);

    // Provide zero demand on current NS, but high demand on opposing EW
    std::unordered_map<RoadId, uint32_t> queues{ { j.road_ns, 0 }, { j.road_ew, 20 } };
    controller.observe_traffic(queues);

    // Run for 3.0s (less than min_green 4.0s)
    for (int i = 0; i < 30; ++i) {
        controller.tick(0.1f);
    }

    // Must still be green because min_green_s (4.0s) has not elapsed
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_EQ(controller.current_phase_index(), 0u);
    EXPECT_EQ(controller.last_decision(), AdaptiveDecision::Hold);
    EXPECT_TRUE(controller.can_proceed(j.road_ns));
    EXPECT_FALSE(controller.last_decision_reason().empty());
}

// ── Test 3: Shorten Phase (Early Gap-Out) ─────────────────────────────────────

TEST(AdaptiveControllerTest, ShortensPhaseEarlyWhenQueueDissipates) {
    auto j = TestJunction::create();
    AdaptiveControllerConfig cfg;
    cfg.min_green_s     = 4.0f;
    cfg.nominal_green_s = 10.0f;
    cfg.max_green_s     = 25.0f;

    AdaptiveController controller(
        j.center,
        { j.phase_ns(), j.phase_ew() },
        { j.road_ns, j.road_ew },
        cfg);

    // Initial state: cars present on current NS
    std::unordered_map<RoadId, uint32_t> queues{ { j.road_ns, 5 }, { j.road_ew, 10 } };
    controller.observe_traffic(queues);

    // Advance past min_green (4.0s) to 4.5s
    for (int i = 0; i < 45; ++i) {
        controller.tick(0.1f);
    }
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);

    // Now current queue clears completely (0 veh), opposing queue is still 10 veh
    queues[j.road_ns] = 0;
    controller.observe_traffic(queues);
    controller.tick(0.1f);

    // Must shorten green and transition to Yellow immediately!
    EXPECT_EQ(controller.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(controller.last_decision(), AdaptiveDecision::ShortenPhase);
    EXPECT_EQ(controller.shortening_count(), 1u);
    EXPECT_NE(controller.last_decision_reason().find("Shortening green early"), std::string::npos);
}

// ── Test 4: Demand-Responsive Green Extension ────────────────────────────────

TEST(AdaptiveControllerTest, ExtendsGreenWhenCurrentApproachHasHeavyDemand) {
    auto j = TestJunction::create();
    AdaptiveControllerConfig cfg;
    cfg.min_green_s          = 4.0f;
    cfg.nominal_green_s      = 10.0f;
    cfg.max_green_s          = 20.0f;
    cfg.max_wait_threshold_s = 30.0f;

    AdaptiveController controller(
        j.center,
        { j.phase_ns(), j.phase_ew() },
        { j.road_ns, j.road_ew },
        cfg);

    // Heavy demand on current NS, zero demand on EW
    std::unordered_map<RoadId, uint32_t> queues{ { j.road_ns, 15 }, { j.road_ew, 0 } };
    controller.observe_traffic(queues);

    // Advance to nominal green (10.0s)
    for (int i = 0; i < 100; ++i) {
        controller.tick(0.1f);
    }

    // Fixed-time would switch to Yellow here, but Adaptive should EXTEND!
    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_EQ(controller.current_phase_index(), 0u);

    // Tick further into extension window (to 15.0s)
    for (int i = 0; i < 50; ++i) {
        controller.tick(0.1f);
    }

    EXPECT_EQ(controller.current_stage(), PhaseStage::Green);
    EXPECT_EQ(controller.last_decision(), AdaptiveDecision::ExtendGreen);
    EXPECT_GT(controller.extension_count(), 0u);
    EXPECT_NE(controller.last_decision_reason().find("Extending green"), std::string::npos);
}

// ── Test 5: Maximum Green Duration Cap ───────────────────────────────────────

TEST(AdaptiveControllerTest, MaxGreenCapTerminatesExtensionAtLimit) {
    auto j = TestJunction::create();
    AdaptiveControllerConfig cfg;
    cfg.min_green_s          = 4.0f;
    cfg.nominal_green_s      = 10.0f;
    cfg.max_green_s          = 18.0f;
    cfg.max_wait_threshold_s = 50.0f;

    AdaptiveController controller(
        j.center,
        { j.phase_ns(), j.phase_ew() },
        { j.road_ns, j.road_ew },
        cfg);

    // Persistent continuous queue on NS
    std::unordered_map<RoadId, uint32_t> queues{ { j.road_ns, 20 }, { j.road_ew, 5 } };
    controller.observe_traffic(queues);

    // Run up to max_green (18.0s)
    for (int i = 0; i < 185; ++i) {
        controller.tick(0.1f);
    }

    // Must have switched to Yellow because max_green was reached
    EXPECT_EQ(controller.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(controller.last_decision(), AdaptiveDecision::SwitchPhase);
    EXPECT_NE(controller.last_decision_reason().find("max green limit reached"), std::string::npos);
}

// ── Test 6: Starvation Prevention ────────────────────────────────────────────

TEST(AdaptiveControllerTest, StarvationThresholdForcesPhaseSwitch) {
    auto j = TestJunction::create();
    AdaptiveControllerConfig cfg;
    cfg.min_green_s          = 4.0f;
    cfg.nominal_green_s      = 10.0f;
    cfg.max_green_s          = 30.0f;
    cfg.max_wait_threshold_s = 15.0f; // Opposing vehicle waited 15s

    AdaptiveController controller(
        j.center,
        { j.phase_ns(), j.phase_ew() },
        { j.road_ns, j.road_ew },
        cfg);

    // Advance to 10.0s nominal green
    std::unordered_map<RoadId, uint32_t> queues{ { j.road_ns, 10 }, { j.road_ew, 2 } };
    std::unordered_map<RoadId, float>    waits{ { j.road_ns, 2.0f }, { j.road_ew, 16.0f } }; // EW waited 16s!
    controller.observe_traffic(queues, waits);

    for (int i = 0; i < 101; ++i) {
        controller.tick(0.1f);
    }

    // Must switch to Yellow to prevent EW starvation even though NS has more queue
    EXPECT_EQ(controller.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(controller.last_decision(), AdaptiveDecision::SwitchPhase);
    EXPECT_NE(controller.last_decision_reason().find("starvation threshold"), std::string::npos);
}

// ── Test 7: Determinism & Repeatability ───────────────────────────────────────

TEST(AdaptiveControllerTest, ControllerIsStrictlyDeterministic) {
    auto j = TestJunction::create();
    AdaptiveController c1(j.center, { j.phase_ns(), j.phase_ew() }, { j.road_ns, j.road_ew });
    AdaptiveController c2(j.center, { j.phase_ns(), j.phase_ew() }, { j.road_ns, j.road_ew });

    std::unordered_map<RoadId, uint32_t> queues{ { j.road_ns, 7 }, { j.road_ew, 3 } };

    for (int tick = 0; tick < 200; ++tick) {
        queues[j.road_ns] = (tick % 17);
        queues[j.road_ew] = (tick % 11);

        c1.observe_traffic(queues);
        c2.observe_traffic(queues);

        c1.tick(0.1f);
        c2.tick(0.1f);

        ASSERT_EQ(c1.current_stage(), c2.current_stage());
        ASSERT_EQ(c1.current_phase_index(), c2.current_phase_index());
        ASSERT_FLOAT_EQ(c1.stage_elapsed_s(), c2.stage_elapsed_s());
        ASSERT_EQ(c1.last_decision(), c2.last_decision());
        ASSERT_EQ(c1.last_decision_reason(), c2.last_decision_reason());
    }
}

// ── Test 8: Simulation Adaptive Mode Integration ─────────────────────────────

TEST(AdaptiveControllerTest, SimulationRunsWithAdaptiveSignalControllers) {
    Config cfg;
    cfg.scenario               = TrafficScenario::Low;
    cfg.seed                   = 42u;
    cfg.sim_duration_s         = 10.0f;
    cfg.headless               = true;
    cfg.signal_controller_type = SignalControllerType::Adaptive;

    Simulation sim{ cfg };
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Adaptive);

    // Verify all signal controllers in simulation are Adaptive
    for (const auto& [id, controller] : sim.signal_controllers()) {
        EXPECT_EQ(controller.type(), SignalControllerType::Adaptive);
        EXPECT_TRUE(controller.is<AdaptiveController>());
    }

    // Run simulation for 10 seconds without errors
    sim.run_for(10.0f);
    EXPECT_FLOAT_EQ(sim.state().elapsed_s, 10.0f);
    EXPECT_GT(sim.state().total_spawned, 0u);
}

} // namespace
} // namespace syntraq
