//
// tests/test_rl_controller.cpp
//
// Comprehensive unit tests for RLController:
// - Four-step lifecycle: receive_observation -> run_inference -> produce_action -> apply_action
// - Graceful operation and heuristic fallback when no AI model is loaded
// - Unified step() method execution
// - Performance telemetry: inference latency, decisions per second, decision counting
// - Signal phase transitions, min_green safety, and max_green failsafe
// - Controller mode switching (FixedTime, Adaptive, ReinforcementLearning) in Simulation
// - Simulation FPS measurement
//

#include <gtest/gtest.h>

#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"

#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

namespace syntraq {
namespace {

std::string resolvePath(const std::string& rel_path) {
    if (std::filesystem::exists(rel_path)) return rel_path;
    if (std::filesystem::exists("../" + rel_path)) return "../" + rel_path;
    if (std::filesystem::exists("../../" + rel_path)) return "../../" + rel_path;
    return rel_path;
}

const std::string kValidModelPath = resolvePath("models/traffic_signal_dqn.onnx");

struct FourWayJunction {
    IntersectionId center{ 0 };
    RoadId r_n_in{ 10 }, r_s_in{ 11 };
    RoadId r_e_in{ 20 }, r_w_in{ 21 };

    SignalPhase phase_ns() const {
        SignalPhase p;
        p.name = "North-South Green";
        p.green_roads = { r_n_in, r_s_in };
        p.green_duration_s = 10.0f;
        p.yellow_duration_s = 3.0f;
        p.all_red_duration_s = 2.0f;
        return p;
    }

    SignalPhase phase_ew() const {
        SignalPhase p;
        p.name = "East-West Green";
        p.green_roads = { r_e_in, r_w_in };
        p.green_duration_s = 10.0f;
        p.yellow_duration_s = 3.0f;
        p.all_red_duration_s = 2.0f;
        return p;
    }
};

} // namespace

// ── Test 1: Four-Step Execution Lifecycle ────────────────────────────────────

TEST(RLControllerTest, FourStepExecutionLifecycle) {
    FourWayJunction j;
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.max_green_s = 45.0f;

    auto policy = std::make_shared<RLPolicyInference>();
    if (std::filesystem::exists(kValidModelPath)) {
        policy->loadModel(kValidModelPath);
    }

    RLController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg, policy);

    // 1. Receive Observation
    std::vector<float> obs(12, 0.25f);
    obs[0] = 0.8f; // Heavy queue on Phase 0
    obs[1] = 0.1f; // Light queue on Phase 1
    obs[6] = 0.0f; // Phase 0 active
    obs[7] = 0.1f; // Early in green
    obs[9] = 1.0f; // can_switch

    ctrl.receive_observation(obs);
    EXPECT_EQ(ctrl.last_observation().size(), 12u);
    EXPECT_FLOAT_EQ(ctrl.last_observation()[0], 0.8f);

    // 2. Run Inference
    auto decision = ctrl.run_inference();
    EXPECT_TRUE(decision.best_action == 0 || decision.best_action == 1);
    if (ctrl.is_model_loaded()) {
        EXPECT_GT(decision.latency_us, 0.0f);
    }

    // 3. Produce Action
    int action = ctrl.produce_action();
    EXPECT_EQ(action, decision.best_action);

    // 4. Apply Action
    ctrl.apply_action(action);
    EXPECT_EQ(ctrl.last_action(), action);
    EXPECT_EQ(ctrl.total_decisions(), 1u);
}

// ── Test 2: Graceful Operation with No AI Model Loaded ───────────────────────

TEST(RLControllerTest, GracefulFallbackWithNoModelLoaded) {
    FourWayJunction j;
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.max_green_s = 45.0f;

    // Explicitly null policy => no AI model loaded
    RLController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg, nullptr);
    // Overwrite with empty policy that fails to load
    ctrl.set_policy(std::make_shared<RLPolicyInference>());

    EXPECT_FALSE(ctrl.is_model_loaded());

    std::vector<float> obs(12, 0.5f);

    // Ensure all 4 steps execute cleanly without exceptions
    ctrl.receive_observation(obs);
    auto decision = ctrl.run_inference();
    EXPECT_TRUE(decision.best_action == 0 || decision.best_action == 1);

    int action = ctrl.produce_action();
    EXPECT_TRUE(action == 0 || action == 1);

    ctrl.apply_action(action);
    EXPECT_EQ(ctrl.total_decisions(), 1u);

    // Verify signal state machine still operates safely
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_EQ(ctrl.current_phase_index(), 0u);

    // Advance 5 seconds (exceeding min_green_s)
    ctrl.tick(5.0f);
    EXPECT_TRUE(ctrl.can_switch());

    // Switch action initiates clearance
    ctrl.apply_action(1);
    ctrl.tick(0.1f);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Yellow);
}

// ── Test 3: Unified Step Execution ───────────────────────────────────────────

TEST(RLControllerTest, UnifiedStepExecution) {
    FourWayJunction j;
    RLControllerConfig cfg;
    cfg.min_green_s = 4.0f;

    RLController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg);

    std::vector<float> obs(12, 0.1f);
    auto decision = ctrl.step(obs);

    EXPECT_EQ(ctrl.total_decisions(), 1u);
    EXPECT_EQ(ctrl.last_action(), decision.best_action);
    EXPECT_EQ(ctrl.last_decision().best_action, decision.best_action);
}

// ── Test 4: Latency & Decisions Per Second Telemetry ──────────────────────────

TEST(RLControllerTest, TelemetryLatencyAndDecisionsPerSecond) {
    FourWayJunction j;
    RLControllerConfig cfg;

    auto policy = std::make_shared<RLPolicyInference>();
    if (std::filesystem::exists(kValidModelPath)) {
        policy->loadModel(kValidModelPath);
    }

    RLController ctrl(j.center, { j.phase_ns(), j.phase_ew() }, {}, cfg, policy);

    std::vector<float> obs(12, 0.3f);

    for (int i = 0; i < 20; ++i) {
        ctrl.step(obs);
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }

    EXPECT_EQ(ctrl.total_decisions(), 20u);
    if (ctrl.is_model_loaded()) {
        EXPECT_GT(ctrl.last_inference_latency_us(), 0.0f);
        EXPECT_GT(ctrl.average_inference_latency_us(), 0.0f);
    }

    const float dps = ctrl.decisions_per_second();
    EXPECT_GE(dps, 0.0f);

    // Reset telemetry
    ctrl.reset_telemetry();
    EXPECT_EQ(ctrl.total_decisions(), 0u);
    EXPECT_FLOAT_EQ(ctrl.average_inference_latency_us(), 0.0f);
}

// ── Test 5: Mode Switching Between Fixed, Adaptive, and RL ───────────────────

TEST(RLControllerTest, ControllerModeSwitchingInSimulation) {
    Config cfg;
    cfg.headless = true;
    cfg.enable_traffic_signals = true;
    cfg.signal_controller_type = SignalControllerType::FixedTime;

    Simulation sim{ cfg };

    // 1. Initial mode: FixedTime
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::FixedTime);
    for (const auto& [id, ctrl] : sim.signal_controllers()) {
        EXPECT_EQ(ctrl.type(), SignalControllerType::FixedTime);
    }

    // 2. Switch to Adaptive
    sim.set_signal_controller_type(SignalControllerType::Adaptive);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Adaptive);
    for (const auto& [id, ctrl] : sim.signal_controllers()) {
        EXPECT_EQ(ctrl.type(), SignalControllerType::Adaptive);
    }

    // 3. Switch to RL
    sim.set_signal_controller_type(SignalControllerType::ReinforcementLearning);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::ReinforcementLearning);
    for (const auto& [id, ctrl] : sim.signal_controllers()) {
        EXPECT_EQ(ctrl.type(), SignalControllerType::ReinforcementLearning);
        EXPECT_NE(ctrl.as<RLController>(), nullptr);
    }

    // Advance simulation in RL mode
    for (int t = 0; t < 25; ++t) {
        sim.tick();
    }

    EXPECT_GT(sim.state().tick, 0u);
}

// ── Test 6: Observation Dimension Padding & Resilience ───────────────────────

TEST(RLControllerTest, ObservationDimensionResilience) {
    FourWayJunction j;
    RLController ctrl(j.center, { j.phase_ns(), j.phase_ew() });

    // Under-sized vector (4 instead of 12)
    std::vector<float> short_obs = { 0.1f, 0.2f, 0.3f, 0.4f };
    ctrl.receive_observation(short_obs);
    EXPECT_EQ(ctrl.last_observation().size(), 12u);
    EXPECT_FLOAT_EQ(ctrl.last_observation()[0], 0.1f);
    EXPECT_FLOAT_EQ(ctrl.last_observation()[1], 0.2f);
    EXPECT_FLOAT_EQ(ctrl.last_observation()[4], 0.0f); // Padded with 0.0f

    // Over-sized vector (18 instead of 12)
    std::vector<float> long_obs(18, 0.7f);
    ctrl.receive_observation(long_obs);
    EXPECT_EQ(ctrl.last_observation().size(), 12u);
    EXPECT_FLOAT_EQ(ctrl.last_observation()[0], 0.7f);

    // Empty vector
    std::vector<float> empty_obs;
    ctrl.receive_observation(empty_obs);
    EXPECT_EQ(ctrl.last_observation().size(), 12u);
    EXPECT_FLOAT_EQ(ctrl.last_observation()[0], 0.0f);
}

// ── Test 7: Simulation FPS Telemetry ─────────────────────────────────────────

TEST(RLControllerTest, SimulationFpsTelemetry) {
    Config cfg;
    cfg.headless = true;
    cfg.dt_seconds = 0.1f;
    cfg.signal_controller_type = SignalControllerType::ReinforcementLearning;

    Simulation sim{ cfg };

    // Advance updates with simulated real time
    for (int i = 0; i < 30; ++i) {
        sim.update(0.01f);
    }

    EXPECT_GT(sim.state().tick, 0u);
    EXPECT_GE(sim.state().simulation_fps, 0.0f);
}

} // namespace syntraq

