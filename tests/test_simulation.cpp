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

// ── Config preservation ──────────────────────────────────────────────────

TEST(SimulationTest, ConfigIsPreservedAfterReset) {
    Config cfg = make_test_config();
    cfg.seed = 777u;
    Simulation sim{ cfg };
    sim.reset();
    EXPECT_EQ(sim.config().seed, 777u);
}

} // namespace syntraq
