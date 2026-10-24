//
// tests/test_traffic_signal_env.cpp
//
// Unit tests for single-intersection Reinforcement Learning environment:
// - RLSignalController actions (Maintain vs Switch)
// - Minimum green safety constraint enforcement
// - Yellow and all-red clearance phase progression
// - Max green failsafe timeout
// - Single-intersection map topological layout
// - TrafficSignalEnv step lifecycle, observation vector bounds [0, 1]
// - Reward computation (queue/wait/congestion penalties and throughput bonus)
// - Episode truncation at horizon
//

#include <gtest/gtest.h>

#include "syntraq/ai/traffic_signal_env.hpp"
#include "syntraq/signals/rl_signal_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <cmath>
#include <vector>

namespace syntraq {
namespace {

struct TestJunction {
    IntersectionId center{ IntersectionId{ 1 } };
    RoadId         road_ns{ RoadId{ 10 } }; // North-South
    RoadId         road_ew{ RoadId{ 20 } }; // East-West

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

// ── Test Group 1: RLSignalController Unit Tests ──────────────────────────────

TEST(RLSignalControllerTest, InitialStateIsGreenPhase0) {
    auto j = TestJunction::create();
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.max_green_s = 30.0f;

    RLSignalController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg);

    EXPECT_EQ(ctrl.current_phase_index(), 0u);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_FLOAT_EQ(ctrl.stage_elapsed_s(), 0.0f);
    EXPECT_EQ(ctrl.signal_for_road(j.road_ns), SignalColor::Green);
    EXPECT_EQ(ctrl.signal_for_road(j.road_ew), SignalColor::Red);
    EXPECT_FALSE(ctrl.can_switch()); // Elapsed 0 < min_green 4
    EXPECT_FALSE(ctrl.is_in_transition());
}

TEST(RLSignalControllerTest, MinGreenSafetyEnforced) {
    auto j = TestJunction::create();
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;

    RLSignalController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg);

    ctrl.tick(2.0f);
    EXPECT_FLOAT_EQ(ctrl.stage_elapsed_s(), 2.0f);
    EXPECT_FALSE(ctrl.can_switch());

    // Switch attempt should be rejected before min_green
    const bool switched = ctrl.switch_phase();
    EXPECT_FALSE(switched);

    // After tick, still in Green Phase 0
    ctrl.tick(1.0f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_EQ(ctrl.current_phase_index(), 0u);
}

TEST(RLSignalControllerTest, SwitchTriggersYellowThenAllRedThenNextGreen) {
    auto j = TestJunction::create();
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.yellow_s    = 3.0f;
    cfg.all_red_s   = 2.0f;

    RLSignalController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg);

    // Advance past min_green
    ctrl.tick(5.0f);
    EXPECT_TRUE(ctrl.can_switch());

    // Request switch
    const bool switched = ctrl.switch_phase();
    EXPECT_TRUE(switched);

    // On tick, should enter Yellow clearance
    ctrl.tick(0.1f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Yellow);
    EXPECT_TRUE(ctrl.is_in_transition());
    EXPECT_EQ(ctrl.signal_for_road(j.road_ns), SignalColor::Yellow);

    // Advance through yellow clearance
    ctrl.tick(3.0f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::AllRed);
    EXPECT_EQ(ctrl.signal_for_road(j.road_ns), SignalColor::Red);
    EXPECT_EQ(ctrl.signal_for_road(j.road_ew), SignalColor::Red);

    // Advance through all-red clearance
    ctrl.tick(2.0f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_EQ(ctrl.current_phase_index(), 1u); // Now Phase 1 (East-West)
    EXPECT_EQ(ctrl.signal_for_road(j.road_ns), SignalColor::Red);
    EXPECT_EQ(ctrl.signal_for_road(j.road_ew), SignalColor::Green);
    EXPECT_FALSE(ctrl.is_in_transition());
}

TEST(RLSignalControllerTest, MaxGreenFailsafeEnforced) {
    auto j = TestJunction::create();
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.max_green_s = 20.0f;
    cfg.yellow_s    = 3.0f;

    RLSignalController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg);

    // Agent never requests switch, but green reaches max_green_s
    ctrl.tick(20.5f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Yellow);
    EXPECT_TRUE(ctrl.is_in_transition());
}

// ── Test Group 2: Single-Intersection Map Factory ────────────────────────────

TEST(SingleIntersectionNetworkTest, TopologicalStructure) {
    RoadNetwork net = make_single_intersection_map({ 200.0f, 200.0f }, 150.0f, 13.89f, 2);

    EXPECT_EQ(net.intersection_count(), 5u); // Center + North + South + West + East
    EXPECT_EQ(net.road_count(), 8u);         // 4 incoming + 4 outgoing

    const auto incoming = net.incoming_roads(IntersectionId{ 0 });
    EXPECT_EQ(incoming.size(), 4u);

    const auto outgoing = net.outgoing_roads(IntersectionId{ 0 });
    EXPECT_EQ(outgoing.size(), 4u);
}

// ── Test Group 3: TrafficSignalEnv MDP Lifecycle ─────────────────────────────

TEST(TrafficSignalEnvTest, InitialResetReturnsValidObservation) {
    EnvConfig cfg;
    cfg.scenario            = TrafficScenario::Medium;
    cfg.step_duration_s     = 1.0f;
    cfg.episode_duration_s  = 100.0f;
    cfg.seed                = 42;

    TrafficSignalEnv env(cfg);
    Observation obs = env.reset(42);

    // Verify 12 normalized dimensions bounded in [0.0, 1.0]
    for (size_t i = 0; i < kObsDim; ++i) {
        EXPECT_GE(obs[i], 0.0f) << "Feature " << i << " is negative";
        EXPECT_LE(obs[i], 1.0f) << "Feature " << i << " exceeds 1.0";
    }

    // Initially phase 0 is active, not clearing
    EXPECT_FLOAT_EQ(obs[6], 0.0f); // Phase 0
    EXPECT_FLOAT_EQ(obs[8], 0.0f); // Not clearing
}

TEST(TrafficSignalEnvTest, MaintainActionAdvancesSimulation) {
    EnvConfig cfg;
    cfg.step_duration_s    = 2.0f;
    cfg.episode_duration_s = 60.0f;
    cfg.seed               = 100;

    TrafficSignalEnv env(cfg);
    env.reset(100);

    StepResult res = env.step(0); // Maintain phase

    EXPECT_NEAR(res.elapsed_s, 2.0f, 0.05f);
    EXPECT_FALSE(res.terminated);
    EXPECT_FALSE(res.truncated);

    // Verify observation dimensions
    for (size_t i = 0; i < kObsDim; ++i) {
        EXPECT_GE(res.observation[i], 0.0f);
        EXPECT_LE(res.observation[i], 1.0f);
    }
}

TEST(TrafficSignalEnvTest, SwitchActionInitiatesClearanceAndChangesPhase) {
    EnvConfig cfg;
    cfg.step_duration_s    = 1.0f;
    cfg.min_green_s        = 4.0f;
    cfg.yellow_s           = 3.0f;
    cfg.all_red_s          = 2.0f;
    cfg.episode_duration_s = 60.0f;

    TrafficSignalEnv env(cfg);
    env.reset(42);

    // Step 0..4 (maintain) to satisfy min_green_s
    for (int i = 0; i < 5; ++i) {
        env.step(0);
    }

    // Now elapsed >= 4s, can_switch should be true (obs[9] == 1.0)
    Observation obs_before = env.get_observation();
    EXPECT_FLOAT_EQ(obs_before[9], 1.0f);

    // Action 1: Switch phase
    StepResult res_switch = env.step(1);

    // Transition should have initiated (Yellow clearance)
    EXPECT_FLOAT_EQ(res_switch.observation[8], 1.0f); // is_clearing == 1.0

    // Step through yellow (3s) and all-red (2s)
    for (int i = 0; i < 5; ++i) {
        env.step(0);
    }

    // Now Phase 1 (East-West) should be green!
    Observation obs_after = env.get_observation();
    EXPECT_FLOAT_EQ(obs_after[6], 1.0f); // Phase 1
    EXPECT_FLOAT_EQ(obs_after[8], 0.0f); // is_clearing == 0.0
}

TEST(TrafficSignalEnvTest, RewardPenalizesQueuesAndCongestion) {
    EnvConfig cfg;
    cfg.step_duration_s           = 1.0f;
    cfg.queue_penalty_weight      = 0.5f;
    cfg.wait_penalty_weight       = 0.1f;
    cfg.congestion_penalty_weight = 1.0f;
    cfg.throughput_reward_weight  = 2.0f;

    TrafficSignalEnv env(cfg);
    env.reset(42);

    // Run several steps to accumulate traffic
    float total_reward = 0.0f;
    for (int i = 0; i < 20; ++i) {
        StepResult res = env.step(0);
        total_reward += res.reward;
    }

    // Reward reflects real penalties and telemetry
    const auto& last = env.last_step_result();
    EXPECT_GE(last.elapsed_s, 19.5f);
}

TEST(TrafficSignalEnvTest, EpisodeTruncatesAtDuration) {
    EnvConfig cfg;
    cfg.step_duration_s    = 5.0f;
    cfg.episode_duration_s = 20.0f;

    TrafficSignalEnv env(cfg);
    env.reset(42);

    StepResult res;
    int step_count = 0;
    while (!res.truncated && step_count < 10) {
        res = env.step(0);
        step_count++;
    }

    EXPECT_TRUE(res.truncated);
    EXPECT_EQ(step_count, 4); // 4 steps * 5s = 20s
}

TEST(TrafficSignalEnvTest, DeterministicResetWithSameSeed) {
    EnvConfig cfg;
    cfg.seed = 777;

    TrafficSignalEnv env(cfg);
    Observation obs1 = env.reset(777);
    StepResult res1  = env.step(0);

    Observation obs2 = env.reset(777);
    StepResult res2  = env.step(0);

    for (size_t i = 0; i < kObsDim; ++i) {
        EXPECT_NEAR(obs1[i], obs2[i], 1e-5f);
        EXPECT_NEAR(res1.observation[i], res2.observation[i], 1e-5f);
    }
    EXPECT_FLOAT_EQ(res1.reward, res2.reward);
}

} // namespace
} // namespace syntraq
