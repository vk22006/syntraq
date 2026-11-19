//
// tests/test_system_integration.cpp
//
// Integration tests covering:
//   Simulation -> Controller -> Traffic Signal Behavior
//
// Verifies closed-loop interaction where simulation state feeds controller
// observations, controllers make actuated/coordinated/RL decisions, and traffic
// signals enforce physical safety intervals (minimum green, yellow, all-red)
// directly influencing vehicle queuing and clearance dynamics.
//

#include <gtest/gtest.h>

#include "syntraq/core/config.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/signals/signal_phase.hpp"
#include "syntraq/signals/traffic_light.hpp"
#include "syntraq/signals/fixed_time_controller.hpp"
#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/vehicles/vehicle.hpp"
#include "syntraq/vehicles/emergency_priority_manager.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace syntraq {
namespace {

RoadNetwork create_cross_intersection_network() {
    RoadNetwork net;
    const auto center = net.add_intersection({ 100.0f, 100.0f }, "Center");
    const auto north  = net.add_intersection({ 100.0f,   0.0f }, "North");
    const auto south  = net.add_intersection({ 100.0f, 200.0f }, "South");
    const auto east   = net.add_intersection({ 200.0f, 100.0f }, "East");
    const auto west   = net.add_intersection({   0.0f, 100.0f }, "West");

    // Inbound roads
    net.add_road(north, center, 100.0f, 13.89f, 2, "North->Center");
    net.add_road(south, center, 100.0f, 13.89f, 2, "South->Center");
    net.add_road(east,  center, 100.0f, 13.89f, 2, "East->Center");
    net.add_road(west,  center, 100.0f, 13.89f, 2, "West->Center");

    // Outbound roads
    net.add_road(center, north, 100.0f, 13.89f, 2, "Center->North");
    net.add_road(center, south, 100.0f, 13.89f, 2, "Center->South");
    net.add_road(center, east,  100.0f, 13.89f, 2, "Center->East");
    net.add_road(center, west,  100.0f, 13.89f, 2, "Center->West");

    return net;
}

RoadNetwork create_corridor_network() {
    RoadNetwork net;
    const auto i0 = net.add_intersection({   0.0f, 0.0f }, "I0");
    const auto i1 = net.add_intersection({ 150.0f, 0.0f }, "I1");
    const auto i2 = net.add_intersection({ 300.0f, 0.0f }, "I2");

    // Forward corridor
    net.add_road(i0, i1, 150.0f, 13.89f, 2, "I0->I1");
    net.add_road(i1, i2, 150.0f, 13.89f, 2, "I1->I2");

    // Backward corridor
    net.add_road(i2, i1, 150.0f, 13.89f, 2, "I2->I1");
    net.add_road(i1, i0, 150.0f, 13.89f, 2, "I1->I0");

    // Cross approaches for I1
    const auto north = net.add_intersection({ 150.0f, -100.0f }, "I1_North");
    const auto south = net.add_intersection({ 150.0f,  100.0f }, "I1_South");
    net.add_road(north, i1, 100.0f, 13.89f, 2, "North->I1");
    net.add_road(i1, north, 100.0f, 13.89f, 2, "I1->North");
    net.add_road(south, i1, 100.0f, 13.89f, 2, "South->I1");
    net.add_road(i1, south, 100.0f, 13.89f, 2, "I1->South");

    return net;
}

Config make_test_config(uint32_t seed = 42) {
    Config cfg;
    cfg.seed                  = seed;
    cfg.headless              = true;
    cfg.dt_seconds            = 0.1f;
    cfg.sim_duration_s        = 120.0f;
    cfg.enable_rl_model       = false; // safe heuristic fallback for RL
    cfg.signal_controller_type = SignalControllerType::FixedTime;
    return cfg;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. Fixed-Time Simulation -> Signal Cycling Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemIntegrationTest, FixedTimeControllerDrivesDeterministicSignalStages) {
    Config cfg = make_test_config();
    cfg.signal_controller_type = SignalControllerType::FixedTime;
    Simulation sim{ cfg, &create_cross_intersection_network };

    auto* ctrl = sim.signal_controller(static_cast<IntersectionId>(0));
    ASSERT_NE(ctrl, nullptr);

    const auto p0_green_s = ctrl->phases()[0].green_duration_s;
    const auto p0_yellow_s = ctrl->phases()[0].yellow_duration_s;
    const auto p0_all_red_s = ctrl->phases()[0].all_red_duration_s;

    EXPECT_EQ(ctrl->current_phase_index(), 0u);
    EXPECT_EQ(ctrl->current_stage(), PhaseStage::Green);

    // Tick forward through Phase 0 Green
    const uint32_t ticks_green = static_cast<uint32_t>(std::floor((p0_green_s - 0.2f) / cfg.dt_seconds));
    for (uint32_t i = 0; i < ticks_green; ++i) {
        sim.tick();
    }
    EXPECT_EQ(ctrl->current_phase_index(), 0u);
    EXPECT_EQ(ctrl->current_stage(), PhaseStage::Green);

    // Cross into Yellow stage
    const uint32_t ticks_to_yellow = static_cast<uint32_t>(std::ceil(0.5f / cfg.dt_seconds));
    for (uint32_t i = 0; i < ticks_to_yellow; ++i) {
        sim.tick();
    }
    EXPECT_EQ(ctrl->current_phase_index(), 0u);
    EXPECT_EQ(ctrl->current_stage(), PhaseStage::Yellow);

    // Advance past Yellow into All-Red stage
    const uint32_t ticks_to_all_red = static_cast<uint32_t>(std::ceil((p0_yellow_s + 0.2f) / cfg.dt_seconds));
    for (uint32_t i = 0; i < ticks_to_all_red; ++i) {
        sim.tick();
    }
    EXPECT_EQ(ctrl->current_phase_index(), 0u);
    EXPECT_EQ(ctrl->current_stage(), PhaseStage::AllRed);

    // Advance past All-Red into Phase 1 Green
    const uint32_t ticks_to_phase1 = static_cast<uint32_t>(std::ceil((p0_all_red_s + 0.2f) / cfg.dt_seconds));
    for (uint32_t i = 0; i < ticks_to_phase1; ++i) {
        sim.tick();
    }
    EXPECT_EQ(ctrl->current_phase_index(), 1u);
    EXPECT_EQ(ctrl->current_stage(), PhaseStage::Green);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Adaptive Controller Queue Actuation & Gap-Out Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemIntegrationTest, AdaptiveControllerActuatesBasedOnSimulationQueues) {
    Config cfg = make_test_config();
    cfg.signal_controller_type = SignalControllerType::Adaptive;
    cfg.scenario               = TrafficScenario::Medium;
    Simulation sim{ cfg, &create_cross_intersection_network };

    auto* ctrl = sim.signal_controller(static_cast<IntersectionId>(0));
    ASSERT_NE(ctrl, nullptr);
    EXPECT_EQ(ctrl->type(), SignalControllerType::Adaptive);

    // Run simulation for 200 ticks (20 seconds)
    for (int t = 0; t < 200; ++t) {
        sim.tick();
        // Invariant: Controller stage must always be Green, Yellow, or All-Red
        const auto stage = ctrl->current_stage();
        EXPECT_TRUE(stage == PhaseStage::Green || stage == PhaseStage::Yellow || stage == PhaseStage::AllRed);

        // Invariant: An active controller must never grant green to non-phase approaches
        const auto* cur_phase = ctrl->current_phase();
        ASSERT_NE(cur_phase, nullptr);
        for (const auto& [rid, road] : sim.network().roads()) {
            if (road.to == static_cast<IntersectionId>(0)) {
                const bool is_active = cur_phase->contains_road(rid);
                if (!is_active) {
                    EXPECT_EQ(ctrl->signal_for_road(rid), SignalColor::Red);
                    EXPECT_FALSE(ctrl->can_proceed(rid));
                }
            }
        }
    }

    // Verify adaptive controller successfully managed queues without stalling
    EXPECT_GT(sim.state().elapsed_s, 19.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Coordinated Corridor Green-Wave & Spillback Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemIntegrationTest, CoordinatedControllerExchangesCorridorStateAndPreventsGridlock) {
    Config cfg = make_test_config();
    cfg.signal_controller_type = SignalControllerType::Coordinated;
    cfg.scenario               = TrafficScenario::High;
    Simulation sim{ cfg, &create_corridor_network };

    // Set allowed origin-destination across the corridor
    sim.set_allowed_spawn_origins({ static_cast<IntersectionId>(0) });
    sim.set_allowed_spawn_destinations({ static_cast<IntersectionId>(2) });

    // Run corridor simulation for 400 ticks (40 seconds)
    for (int t = 0; t < 400; ++t) {
        sim.tick();

        // Check telemetry across all 3 corridor intersections
        for (uint32_t i = 0; i <= 2; ++i) {
            const auto* c = sim.signal_controller(static_cast<IntersectionId>(i));
            if (c) {
                EXPECT_TRUE(c->current_stage() == PhaseStage::Green ||
                            c->current_stage() == PhaseStage::Yellow ||
                            c->current_stage() == PhaseStage::AllRed);
            }
        }
    }

    // High traffic corridor must remain active and progress vehicles
    EXPECT_GT(sim.state().active_vehicles + sim.state().total_arrived, 0u);
    EXPECT_TRUE(sim.state().running);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Reinforcement Learning Controller Integration & Fallback Policy
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemIntegrationTest, RLControllerGeneratesValidObservationsAndAppliesActions) {
    Config cfg = make_test_config();
    cfg.signal_controller_type = SignalControllerType::ReinforcementLearning;
    cfg.enable_rl_model        = false; // tests robust heuristic fallback
    Simulation sim{ cfg, &create_cross_intersection_network };

    auto* ctrl = sim.signal_controller(static_cast<IntersectionId>(0));
    ASSERT_NE(ctrl, nullptr);
    EXPECT_EQ(ctrl->type(), SignalControllerType::ReinforcementLearning);

    // Verify observation extraction for the intersection
    std::vector<float> obs = sim.compute_rl_observation(static_cast<IntersectionId>(0));
    EXPECT_EQ(obs.size(), 12u);
    for (float val : obs) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
        EXPECT_GE(val, 0.0f);
    }

    // Run simulation for 300 ticks (30 seconds)
    for (int t = 0; t < 300; ++t) {
        sim.tick();

        // Check observation extraction in active simulation
        if (t % 50 == 0) {
            obs = sim.compute_rl_observation(static_cast<IntersectionId>(0));
            EXPECT_EQ(obs.size(), 12u);
            for (float val : obs) {
                EXPECT_FALSE(std::isnan(val));
                EXPECT_FALSE(std::isinf(val));
            }
        }
    }

    // RL controller must produce decisions without crashing or getting stuck
    EXPECT_GT(sim.state().elapsed_s, 29.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Emergency Priority Preemption & Safe Recovery Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemIntegrationTest, EmergencyVehiclePreemptsSignalsAndResumesNormalOperation) {
    Config cfg = make_test_config();
    cfg.signal_controller_type    = SignalControllerType::Adaptive;
    cfg.enable_emergency_priority = true;
    cfg.emergency_preemption_distance_m = 120.0f;
    Simulation sim{ cfg, &create_cross_intersection_network };

    // Run normally for 50 ticks to populate some baseline traffic
    for (int t = 0; t < 50; ++t) {
        sim.tick();
    }

    // Trigger emergency vehicle from North to South
    const bool spawned = sim.trigger_emergency_vehicle(
        static_cast<IntersectionId>(1), // North
        static_cast<IntersectionId>(2)  // South
    );
    EXPECT_TRUE(spawned);

    // Advance simulation while emergency vehicle travels
    bool saw_preemption = false;
    for (int t = 0; t < 250; ++t) {
        sim.tick();

        if (!sim.emergency_manager().status().preempted_intersections.empty()) {
            saw_preemption = true;
            // Preempted controller safely transitions stages and holds green for priority approach
            const auto* ctrl = sim.signal_controller(static_cast<IntersectionId>(0));
            if (ctrl) {
                const auto stage = ctrl->current_stage();
                EXPECT_TRUE(stage == PhaseStage::Green || stage == PhaseStage::Yellow || stage == PhaseStage::AllRed);
            }
        }
    }

    EXPECT_TRUE(saw_preemption);

    // Run further to ensure full passage and recovery
    for (int t = 0; t < 100; ++t) {
        sim.tick();
    }

    // System must remain operational with running simulation
    EXPECT_TRUE(sim.state().running);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Dynamic Runtime Controller Switching Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST(SystemIntegrationTest, DynamicControllerSwitchingPreservesStateAndNetworkIntegrity) {
    Config cfg = make_test_config();
    cfg.signal_controller_type = SignalControllerType::FixedTime;
    cfg.scenario               = TrafficScenario::Medium;
    Simulation sim{ cfg, &create_cross_intersection_network };

    // 1. Run under FixedTime for 50 ticks
    for (int t = 0; t < 50; ++t) sim.tick();
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::FixedTime);

    // 2. Switch to Adaptive
    sim.set_signal_controller_type(SignalControllerType::Adaptive);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Adaptive);
    for (int t = 0; t < 50; ++t) sim.tick();

    // 3. Switch to Coordinated
    sim.set_signal_controller_type(SignalControllerType::Coordinated);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Coordinated);
    for (int t = 0; t < 50; ++t) sim.tick();

    // 4. Switch to Reinforcement Learning
    sim.set_signal_controller_type(SignalControllerType::ReinforcementLearning);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::ReinforcementLearning);
    for (int t = 0; t < 50; ++t) sim.tick();

    // 5. Switch back to FixedTime
    sim.set_signal_controller_type(SignalControllerType::FixedTime);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::FixedTime);
    for (int t = 0; t < 50; ++t) sim.tick();

    // Total elapsed time must be 25.0s (250 ticks * 0.1s)
    EXPECT_NEAR(sim.state().elapsed_s, 25.0f, 0.05f);
    EXPECT_TRUE(sim.state().running);
}

} // namespace syntraq
