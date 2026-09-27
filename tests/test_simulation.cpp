//
// tests/test_simulation.cpp
//
// Unit tests for the Simulation class.
// These run headlessly — no window, no Raylib, no rendering.
//

#include "syntraq/simulation/simulation.h"
#include "syntraq/core/config.h"
#include <gtest/gtest.h>
#include <cmath>

namespace syntraq {

// ── Helpers ──────────────────────────────────────────────────────────────

static Config make_test_config() {
    Config cfg;
    cfg.dt_seconds  = 0.1f;
    cfg.headless    = true;
    cfg.seed        = 1u;
    return cfg;
}

// ── Initial state ────────────────────────────────────────────────────────

TEST(SimulationTest, InitialStateIsZero) {
    Simulation sim{ make_test_config() };
    EXPECT_EQ(sim.state().tick,      0u);
    EXPECT_FLOAT_EQ(sim.state().elapsed_s, 0.0f);
    EXPECT_TRUE(sim.state().running);
}

// ── Single tick ──────────────────────────────────────────────────────────

TEST(SimulationTest, SingleTickAdvancesTime) {
    Simulation sim{ make_test_config() };
    sim.tick();
    EXPECT_EQ(sim.state().tick, 1u);
    EXPECT_NEAR(sim.state().elapsed_s, 0.1f, 1e-5f);
}

// ── Multiple ticks ───────────────────────────────────────────────────────

TEST(SimulationTest, TenTicksAdvanceCorrectly) {
    Simulation sim{ make_test_config() };
    for (int i = 0; i < 10; ++i) sim.tick();
    EXPECT_EQ(sim.state().tick, 10u);
    EXPECT_NEAR(sim.state().elapsed_s, 1.0f, 1e-4f);
}

// ── run_for ──────────────────────────────────────────────────────────────

TEST(SimulationTest, RunForExecutesCorrectNumberOfTicks) {
    Config cfg = make_test_config();
    cfg.dt_seconds = 0.1f;
    Simulation sim{ cfg };
    sim.run_for(1.0f);
    // Should have ticked at least 10 times (1.0 / 0.1)
    EXPECT_GE(sim.state().tick, 10u);
    EXPECT_GE(sim.state().elapsed_s, 1.0f);
}

TEST(SimulationTest, RunForZeroDurationDoesNothing) {
    Simulation sim{ make_test_config() };
    sim.run_for(0.0f);
    EXPECT_EQ(sim.state().tick, 0u);
}

// ── Determinism ──────────────────────────────────────────────────────────

TEST(SimulationTest, SameSeedProducesSameState) {
    Config cfg = make_test_config();
    cfg.seed = 99u;

    Simulation sim1{ cfg };
    Simulation sim2{ cfg };

    for (int i = 0; i < 50; ++i) { sim1.tick(); sim2.tick(); }

    EXPECT_EQ(sim1.state().tick,      sim2.state().tick);
    EXPECT_FLOAT_EQ(sim1.state().elapsed_s, sim2.state().elapsed_s);
}

// ── Reset ────────────────────────────────────────────────────────────────

TEST(SimulationTest, ResetRestoresInitialState) {
    Simulation sim{ make_test_config() };
    for (int i = 0; i < 20; ++i) sim.tick();
    EXPECT_GT(sim.state().tick, 0u);

    sim.reset();
    EXPECT_EQ(sim.state().tick,      0u);
    EXPECT_FLOAT_EQ(sim.state().elapsed_s, 0.0f);
    EXPECT_TRUE(sim.state().running);
}

// ── Pause / Resume ───────────────────────────────────────────────────────

TEST(SimulationTest, PausePreventsSimulationAdvancement) {
    Simulation sim{ make_test_config() };
    sim.set_paused(true);
    EXPECT_TRUE(sim.is_paused());

    // Calling update or tick while paused should not advance
    sim.update(1.0f);
    EXPECT_EQ(sim.state().tick, 0u);
    EXPECT_FLOAT_EQ(sim.state().elapsed_s, 0.0f);

    sim.tick();
    EXPECT_EQ(sim.state().tick, 0u);
    EXPECT_FLOAT_EQ(sim.state().elapsed_s, 0.0f);
}

TEST(SimulationTest, ResumeAllowsSimulationAdvancement) {
    Simulation sim{ make_test_config() };
    sim.set_paused(true);
    sim.set_paused(false);
    EXPECT_FALSE(sim.is_paused());

    sim.update(0.5f);
    // dt is 0.1s -> 5 ticks
    EXPECT_EQ(sim.state().tick, 5u);
    EXPECT_NEAR(sim.state().elapsed_s, 0.5f, 1e-4f);
}

// ── Speed Multiplier ─────────────────────────────────────────────────────

TEST(SimulationTest, SpeedMultiplierBehavesCorrectly) {
    // 0.5x speed: 1.0s real time -> 0.5s sim time -> 5 ticks
    {
        Simulation sim{ make_test_config() };
        sim.set_time_scale(0.5f);
        EXPECT_FLOAT_EQ(sim.time_scale(), 0.5f);
        sim.update(1.0f);
        EXPECT_EQ(sim.state().tick, 5u);
        EXPECT_NEAR(sim.state().elapsed_s, 0.5f, 1e-4f);
    }
    // 2.0x speed: 1.0s real time -> 2.0s sim time -> 20 ticks
    {
        Simulation sim{ make_test_config() };
        sim.set_time_scale(2.0f);
        EXPECT_FLOAT_EQ(sim.time_scale(), 2.0f);
        sim.update(1.0f);
        EXPECT_EQ(sim.state().tick, 20u);
        EXPECT_NEAR(sim.state().elapsed_s, 2.0f, 1e-3f);
    }
    // 5.0x speed: 1.0s real time -> 5.0s sim time -> 50 ticks
    {
        Simulation sim{ make_test_config() };
        sim.set_time_scale(5.0f);
        EXPECT_FLOAT_EQ(sim.time_scale(), 5.0f);
        sim.update(1.0f);
        EXPECT_EQ(sim.state().tick, 50u);
        EXPECT_NEAR(sim.state().elapsed_s, 5.0f, 1e-3f);
    }
}

// ── Fixed Timestep Determinism ───────────────────────────────────────────

TEST(SimulationTest, FixedTimestepIndependentOfFrameRate) {
    Config cfg = make_test_config();
    cfg.seed = 42u;

    Simulation sim_60fps{ cfg };
    Simulation sim_120fps{ cfg };

    // Simulate 1.0s real time at 60 FPS (60 steps of ~1/60s)
    const float dt60 = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        sim_60fps.update(dt60);
    }

    // Simulate 1.0s real time at 120 FPS (120 steps of ~1/120s)
    const float dt120 = 1.0f / 120.0f;
    for (int i = 0; i < 120; ++i) {
        sim_120fps.update(dt120);
    }

    // Both should have executed exactly 10 ticks (1.0s / 0.1s dt)
    EXPECT_EQ(sim_60fps.state().tick, 10u);
    EXPECT_EQ(sim_120fps.state().tick, 10u);
    EXPECT_FLOAT_EQ(sim_60fps.state().elapsed_s, sim_120fps.state().elapsed_s);
    EXPECT_EQ(sim_60fps.state().total_spawned, sim_120fps.state().total_spawned);
}

// ── Config preservation ──────────────────────────────────────────────────

TEST(SimulationTest, ConfigIsPreservedAfterReset) {
    Config cfg = make_test_config();
    cfg.seed = 777u;
    Simulation sim{ cfg };
    sim.reset();
    EXPECT_EQ(sim.config().seed, 777u);
}

} // namespace syntraq
