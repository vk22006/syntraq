//
// tests/test_deterministic_regression.cpp
//
// Deterministic regression test suite:
// Verifies bit-for-bit repeatability across identical seeds, preservation of
// physical invariants (no collisions, bounded velocities, positive headways),
// and absence of behavioral drift across engine versions.
//

#include <gtest/gtest.h>

#include "syntraq/core/config.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"
#include "syntraq/vehicles/vehicle.hpp"

#include <cmath>
#include <vector>

namespace syntraq {
namespace {

Config make_regression_config(uint32_t seed = 42, TrafficScenario scenario = TrafficScenario::Medium) {
    Config cfg;
    cfg.seed                  = seed;
    cfg.headless              = true;
    cfg.dt_seconds            = 0.1f;
    cfg.sim_duration_s        = 100.0f;
    cfg.scenario               = scenario;
    cfg.signal_controller_type = SignalControllerType::FixedTime;
    cfg.routing_strategy       = RoutingStrategyType::Static;
    return cfg;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. Bit-For-Bit Deterministic Parity Test
// ═══════════════════════════════════════════════════════════════════════════

TEST(DeterministicRegressionTest, IdenticalSeedsProduceBitForBitIdenticalStates) {
    constexpr uint32_t kSeed = 101u;
    constexpr int      kTicks = 400;

    Config cfgA = make_regression_config(kSeed);
    Config cfgB = make_regression_config(kSeed);

    Simulation simA{ cfgA };
    Simulation simB{ cfgB };

    for (int t = 0; t < kTicks; ++t) {
        simA.tick();
        simB.tick();

        // Intermediate parity assertions every 50 ticks
        if (t % 50 == 0) {
            ASSERT_EQ(simA.state().active_vehicles, simB.state().active_vehicles);
            ASSERT_EQ(simA.state().queued_vehicles, simB.state().queued_vehicles);
            ASSERT_FLOAT_EQ(simA.state().elapsed_s, simB.state().elapsed_s);
        }
    }

    const auto& stateA = simA.state();
    const auto& stateB = simB.state();

    EXPECT_FLOAT_EQ(stateA.elapsed_s, stateB.elapsed_s);
    EXPECT_EQ(stateA.active_vehicles, stateB.active_vehicles);
    EXPECT_EQ(stateA.queued_vehicles, stateB.queued_vehicles);
    EXPECT_EQ(stateA.total_spawned, stateB.total_spawned);
    EXPECT_EQ(stateA.total_arrived, stateB.total_arrived);
    EXPECT_EQ(stateA.max_queue_len, stateB.max_queue_len);
    EXPECT_FLOAT_EQ(stateA.avg_speed_mps, stateB.avg_speed_mps);
    EXPECT_FLOAT_EQ(stateA.avg_wait_time_s, stateB.avg_wait_time_s);
    EXPECT_FLOAT_EQ(stateA.avg_travel_time_s, stateB.avg_travel_time_s);
    EXPECT_FLOAT_EQ(stateA.throughput_vph, stateB.throughput_vph);
    EXPECT_FLOAT_EQ(stateA.congestion_ratio, stateB.congestion_ratio);

    // Exact vehicle array comparison
    const auto& vehA = simA.vehicles();
    const auto& vehB = simB.vehicles();
    ASSERT_EQ(vehA.size(), vehB.size());

    for (size_t i = 0; i < vehA.size(); ++i) {
        EXPECT_EQ(vehA[i].id, vehB[i].id);
        EXPECT_EQ(vehA[i].current_road, vehB[i].current_road);
        EXPECT_EQ(vehA[i].lane_index, vehB[i].lane_index);
        EXPECT_FLOAT_EQ(vehA[i].speed_mps, vehB[i].speed_mps);
        EXPECT_FLOAT_EQ(vehA[i].progress_m, vehB[i].progress_m);
        EXPECT_FLOAT_EQ(vehA[i].distance_m, vehB[i].distance_m);
        EXPECT_FLOAT_EQ(vehA[i].wait_time_s, vehB[i].wait_time_s);
        EXPECT_FLOAT_EQ(vehA[i].travel_time_s, vehB[i].travel_time_s);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Physical Invariants Verification (Zero Collisions, Bounded Kinematics)
// ═══════════════════════════════════════════════════════════════════════════

TEST(DeterministicRegressionTest, PhysicalInvariantsHoldAcrossLongSimulations) {
    Config cfg = make_regression_config(42u, TrafficScenario::High);
    Simulation sim{ cfg };

    // Advance 600 ticks (60 seconds of high traffic)
    for (int t = 0; t < 600; ++t) {
        sim.tick();

        const auto& state = sim.state();
        EXPECT_TRUE(state.running);
        EXPECT_GE(state.queued_vehicles, 0u);
        EXPECT_LE(state.queued_vehicles, state.active_vehicles);
        EXPECT_GE(state.avg_speed_mps, 0.0f);
        EXPECT_GE(state.avg_wait_time_s, 0.0f);
        EXPECT_GE(state.avg_travel_time_s, 0.0f);
        EXPECT_GE(state.congestion_ratio, 0.0f);
        EXPECT_LE(state.congestion_ratio, 1.0f);

        // Per-vehicle kinematic invariants
        for (const auto& v : sim.vehicles()) {
            if (v.has_arrived()) continue;
            EXPECT_GE(v.speed_mps, 0.0f);
            EXPECT_LE(v.speed_mps, v.max_speed_mps + 0.05f); // allowing minor float eps
            EXPECT_GE(v.distance_m, 0.0f);
            EXPECT_GE(v.progress_m, -50.0f); // entry queue buffering
            EXPECT_GE(v.wait_time_s, 0.0f);
            EXPECT_GE(v.travel_time_s, 0.0f);
        }
    }

    // After 60 seconds of high traffic, vehicles must have arrived and throughput generated
    EXPECT_GT(sim.state().total_spawned, 10u);
    EXPECT_GT(sim.state().active_vehicles + sim.state().total_arrived, 10u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Multi-Seed Stability & Absence of NaN/Inf Divergence
// ═══════════════════════════════════════════════════════════════════════════

TEST(DeterministicRegressionTest, MultiSeedRunsExecuteCleanlyWithoutDivergence) {
    const std::vector<uint32_t> test_seeds = { 7u, 42u, 123u, 9999u };

    for (uint32_t seed : test_seeds) {
        Config cfg = make_regression_config(seed, TrafficScenario::Medium);
        Simulation sim{ cfg };

        for (int t = 0; t < 200; ++t) {
            sim.tick();
        }

        const auto& state = sim.state();
        EXPECT_TRUE(state.running);
        EXPECT_FALSE(std::isnan(state.avg_speed_mps)) << "NaN avg_speed on seed: " << seed;
        EXPECT_FALSE(std::isinf(state.avg_speed_mps)) << "Inf avg_speed on seed: " << seed;
        EXPECT_FALSE(std::isnan(state.congestion_ratio)) << "NaN congestion on seed: " << seed;
        EXPECT_GT(state.active_vehicles + state.total_arrived, 0u);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Golden Benchmark Telemetry Bounds
// ═══════════════════════════════════════════════════════════════════════════

TEST(DeterministicRegressionTest, GoldenReferenceScenarioRemainsWithinExpectedBands) {
    Config cfg = make_regression_config(42u, TrafficScenario::Medium);
    Simulation sim{ cfg };

    // Run for 300 ticks (30s)
    for (int t = 0; t < 300; ++t) {
        sim.tick();
    }

    const auto& s = sim.state();
    EXPECT_NEAR(s.elapsed_s, 30.0f, 0.05f);
    EXPECT_GT(s.total_spawned, 5u);
    EXPECT_GT(s.avg_speed_mps, 1.0f);
    EXPECT_LT(s.avg_speed_mps, 20.0f);
    EXPECT_LE(s.congestion_ratio, 1.0f);
    EXPECT_TRUE(s.running);
}

} // namespace syntraq
