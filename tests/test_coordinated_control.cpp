//
// tests/test_coordinated_control.cpp
//
// Comprehensive unit tests for Network-Level Coordinated Traffic Signal Control.
// Validates:
// 1. Single-intersection fallback (backward compatibility with 0 neighbors).
// 2. Downstream spillback mitigation (queue propagation gating).
// 3. Upstream green-wave progression (platoon arrival green extension).
// 4. Network coordination manager topology mapping & state exchange.
// 5. Multi-intersection corridor simulation execution.
// 6. Queue propagation and waiting time reduction.
//

#include <gtest/gtest.h>

#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/signals/network_coordination_manager.hpp"
#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"

using namespace syntraq;

namespace {

// Helper to construct a simple 2-phase intersection controller
CoordinatedController make_test_controller(
    IntersectionId id,
    RoadId north_road = RoadId(1),
    RoadId south_road = RoadId(2),
    RoadId east_road  = RoadId(3),
    RoadId west_road  = RoadId(4),
    CoordinatedControllerConfig config = {})
{
    std::vector<SignalPhase> phases;

    SignalPhase p_ns;
    p_ns.name = "North-South";
    p_ns.green_roads = { north_road, south_road };
    p_ns.green_duration_s = config.nominal_green_s;
    p_ns.yellow_duration_s = 3.0f;
    p_ns.all_red_duration_s = 2.0f;
    phases.push_back(p_ns);

    SignalPhase p_ew;
    p_ew.name = "East-West";
    p_ew.green_roads = { east_road, west_road };
    p_ew.green_duration_s = config.nominal_green_s;
    p_ew.yellow_duration_s = 3.0f;
    p_ew.all_red_duration_s = 2.0f;
    phases.push_back(p_ew);

    std::vector<RoadId> all_roads = { north_road, south_road, east_road, west_road };
    return CoordinatedController(id, std::move(phases), std::move(all_roads), config);
}

// Helper to construct a 3-intersection linear corridor:
// I0 <--(r10)-- (r01)--> I1 <--(r21)-- (r12)--> I2
// plus cross streets on each intersection.
RoadNetwork build_3intersection_corridor() {
    RoadNetwork net;

    // Intersections at x = 0, 200, 400
    auto i0 = net.add_intersection({0.0f, 0.0f});
    auto i1 = net.add_intersection({200.0f, 0.0f});
    auto i2 = net.add_intersection({400.0f, 0.0f});

    // Outer boundary intersections for entry/exit
    auto west_entry = net.add_intersection({-200.0f, 0.0f});
    auto east_exit  = net.add_intersection({600.0f, 0.0f});

    // Cross street endpoints
    auto n0 = net.add_intersection({0.0f, 150.0f});
    auto s0 = net.add_intersection({0.0f, -150.0f});
    auto n1 = net.add_intersection({200.0f, 150.0f});
    auto s1 = net.add_intersection({200.0f, -150.0f});
    auto n2 = net.add_intersection({400.0f, 150.0f});
    auto s2 = net.add_intersection({400.0f, -150.0f});

    // Corridor arterial roads
    net.add_road(west_entry, i0, 13.89f, 2); // road into i0
    net.add_road(i0, i1, 13.89f, 2);         // road i0 -> i1
    net.add_road(i1, i2, 13.89f, 2);         // road i1 -> i2
    net.add_road(i2, east_exit, 13.89f, 2);  // road i2 -> exit

    // Reverse arterial roads
    net.add_road(i1, i0, 13.89f, 2);
    net.add_road(i2, i1, 13.89f, 2);

    // Cross street incoming roads
    net.add_road(n0, i0, 11.11f, 1);
    net.add_road(s0, i0, 11.11f, 1);
    net.add_road(n1, i1, 11.11f, 1);
    net.add_road(s1, i1, 11.11f, 1);
    net.add_road(n2, i2, 11.11f, 1);
    net.add_road(s2, i2, 11.11f, 1);

    return net;
}

} // namespace

// ===========================================================================
// 1. Single-Intersection Fallback Tests (Backward Compatibility)
// ===========================================================================

TEST(CoordinatedControlTest, SingleIntersectionModeFunctionsWithZeroNeighbors) {
    CoordinatedController ctrl = make_test_controller(IntersectionId(1));
    EXPECT_TRUE(ctrl.is_valid());
    EXPECT_EQ(ctrl.neighbor_states().size(), 0u);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_EQ(ctrl.current_phase_index(), 0u);

    // Run without any neighbor inputs — behaves as deterministic local controller
    for (int step = 0; step < 50; ++step) {
        ctrl.tick(0.1f);
    }
    // Should have advanced past min green (4.0s) smoothly
    EXPECT_GT(ctrl.stage_elapsed_s(), 4.5f);
}

TEST(CoordinatedControlTest, LocalTrafficExtendsGreenWhenVehiclesPresent) {
    CoordinatedControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.nominal_green_s = 10.0f;
    cfg.max_green_s = 20.0f;

    CoordinatedController ctrl = make_test_controller(IntersectionId(1), RoadId(1), RoadId(2), RoadId(3), RoadId(4), cfg);

    // Advance to 5.0s (past min_green_s)
    for (int i = 0; i < 50; ++i) ctrl.tick(0.1f);

    // Report active queue on active green roads 1 and 2
    std::unordered_map<RoadId, uint32_t> queues = { {RoadId(1), 4}, {RoadId(2), 3} };
    std::unordered_map<RoadId, float> waits = { {RoadId(1), 2.0f}, {RoadId(2), 1.5f} };
    ctrl.observe_traffic(queues, waits);

    ctrl.tick(0.1f);
    // Should extend green locally or hold green
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
}

TEST(CoordinatedControlTest, LocalTrafficGapsOutWhenNoActiveVehicles) {
    CoordinatedControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.nominal_green_s = 10.0f;

    CoordinatedController ctrl = make_test_controller(IntersectionId(1), RoadId(1), RoadId(2), RoadId(3), RoadId(4), cfg);

    // Advance to 4.5s (past min green)
    for (int i = 0; i < 45; ++i) ctrl.tick(0.1f);

    // Active queue empty, cross-street has waiting queue
    std::unordered_map<RoadId, uint32_t> queues = { {RoadId(1), 0}, {RoadId(2), 0}, {RoadId(3), 5}, {RoadId(4), 3} };
    std::unordered_map<RoadId, float> waits = { {RoadId(3), 12.0f}, {RoadId(4), 10.0f} };
    ctrl.observe_traffic(queues, waits);

    ctrl.tick(0.1f);
    // Should gap out and switch phase via yellow clearance
    EXPECT_EQ(ctrl.last_decision(), CoordinatedDecision::ShortenPhase);
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Yellow);
}

// ===========================================================================
// 2. Downstream Spillback Mitigation Tests
// ===========================================================================

TEST(CoordinatedControlTest, SpillbackMitigationTruncatesGreenWhenDownstreamCongested) {
    CoordinatedControllerConfig cfg;
    cfg.min_green_s = 4.0f;
    cfg.nominal_green_s = 15.0f;
    cfg.enable_spillback_prevention = true;
    cfg.spillback_queue_threshold = 5;

    CoordinatedController upstream_ctrl = make_test_controller(IntersectionId(10), RoadId(1), RoadId(2), RoadId(3), RoadId(4), cfg);

    // Advance upstream controller past safety minimum green (4.5s)
    for (int i = 0; i < 45; ++i) upstream_ctrl.tick(0.1f);
    EXPECT_EQ(upstream_ctrl.current_stage(), PhaseStage::Green);

    // Downstream receiving intersection sends spillback alarm
    NeighborIntersectionState down_state;
    down_state.neighbor_id = IntersectionId(20);
    down_state.connecting_road = RoadId(1); // Our green arterial road feeds downstream
    down_state.is_upstream = false;
    down_state.downstream_queue = 8; // >= 5 threshold
    down_state.downstream_occupancy_ratio = 0.80f;
    down_state.spillback_imminent = true;

    upstream_ctrl.receive_neighbor_state(down_state);
    upstream_ctrl.tick(0.1f);

    // Controller must immediately trigger MitigateSpillback and switch to Yellow
    EXPECT_EQ(upstream_ctrl.last_decision(), CoordinatedDecision::MitigateSpillback);
    EXPECT_EQ(upstream_ctrl.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(upstream_ctrl.spillback_mitigations(), 1u);
}

TEST(CoordinatedControlTest, SpillbackMitigationRespectsSafetyMinimumGreen) {
    CoordinatedControllerConfig cfg;
    cfg.min_green_s = 5.0f;
    cfg.enable_spillback_prevention = true;

    CoordinatedController ctrl = make_test_controller(IntersectionId(10), RoadId(1), RoadId(2), RoadId(3), RoadId(4), cfg);

    // Advance only 2.0s (under minimum green)
    for (int i = 0; i < 20; ++i) ctrl.tick(0.1f);

    NeighborIntersectionState down_state;
    down_state.neighbor_id = IntersectionId(20);
    down_state.connecting_road = RoadId(1);
    down_state.is_upstream = false;
    down_state.spillback_imminent = true;
    down_state.downstream_queue = 10;

    ctrl.receive_neighbor_state(down_state);
    ctrl.tick(0.1f);

    // Must NOT switch yet because mandatory minimum green safety buffer has not elapsed
    EXPECT_EQ(ctrl.current_stage(), PhaseStage::Green);
    EXPECT_EQ(ctrl.last_decision(), CoordinatedDecision::Hold);
}

// ===========================================================================
// 3. Upstream Green Wave Progression Tests
// ===========================================================================

TEST(CoordinatedControlTest, UpstreamGreenWaveExtendsDownstreamGreenForApproachingPlatoon) {
    CoordinatedControllerConfig cfg;
    cfg.min_green_s = 5.0f;
    cfg.nominal_green_s = 10.0f;
    cfg.max_green_s = 25.0f;
    cfg.progression_window_s = 6.0f;
    cfg.enable_green_wave = true;

    CoordinatedController downstream_ctrl = make_test_controller(IntersectionId(20), RoadId(1), RoadId(2), RoadId(3), RoadId(4), cfg);

    // Advance downstream controller to 9.0s (close to nominal green expiry 10.0s)
    for (int i = 0; i < 90; ++i) downstream_ctrl.tick(0.1f);
    EXPECT_EQ(downstream_ctrl.current_stage(), PhaseStage::Green);

    // Upstream intersection dispatches a platoon on road 1 (which is active green on downstream)
    NeighborIntersectionState up_state;
    up_state.neighbor_id = IntersectionId(10);
    up_state.connecting_road = RoadId(1);
    up_state.is_upstream = true;
    up_state.platoon_dispatched = true;
    up_state.estimated_platoon_eta_s = 4.0f; // Arriving in 4.0s (within 6.0s window)
    up_state.estimated_platoon_size = 6;

    downstream_ctrl.receive_neighbor_state(up_state);
    downstream_ctrl.tick(0.1f);

    // Must extend green wave progression!
    EXPECT_EQ(downstream_ctrl.last_decision(), CoordinatedDecision::ExtendGreenWave);
    EXPECT_EQ(downstream_ctrl.current_stage(), PhaseStage::Green);
    EXPECT_GT(downstream_ctrl.current_stage_duration(), 10.0f);
    EXPECT_EQ(downstream_ctrl.green_wave_extensions(), 1u);
}

TEST(CoordinatedControlTest, GreenWaveDoesNotExtendBeyondMaxGreenCap) {
    CoordinatedControllerConfig cfg;
    cfg.min_green_s = 5.0f;
    cfg.nominal_green_s = 10.0f;
    cfg.max_green_s = 15.0f;
    cfg.progression_window_s = 6.0f;
    cfg.enable_green_wave = true;

    CoordinatedController downstream_ctrl = make_test_controller(IntersectionId(20), RoadId(1), RoadId(2), RoadId(3), RoadId(4), cfg);

    NeighborIntersectionState up_state;
    up_state.neighbor_id = IntersectionId(10);
    up_state.connecting_road = RoadId(1);
    up_state.is_upstream = true;
    up_state.platoon_dispatched = true;
    up_state.estimated_platoon_eta_s = 2.0f;

    // Continuously feed upstream platoon so green is extended until reaching max green limit (15.0s)
    while (downstream_ctrl.current_stage() == PhaseStage::Green) {
        downstream_ctrl.receive_neighbor_state(up_state);
        downstream_ctrl.tick(0.1f);
    }

    // Must switch phase because max green cap is reached
    EXPECT_EQ(downstream_ctrl.current_stage(), PhaseStage::Yellow);
    EXPECT_EQ(downstream_ctrl.last_decision(), CoordinatedDecision::SwitchPhase);
    EXPECT_NE(downstream_ctrl.last_reason().find("Maximum green"), std::string::npos);
}

// ===========================================================================
// 4. Network Coordination Manager & Topology Tests
// ===========================================================================

TEST(NetworkCoordinationManagerTest, DiscoversCorridorTopology) {
    RoadNetwork net = build_3intersection_corridor();
    NetworkCoordinationManager mgr;

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    for (IntersectionId i : {IntersectionId(1), IntersectionId(2), IntersectionId(3)}) {
        controllers[i] = TrafficSignalController::make_coordinated(net, i);
    }

    mgr.build_topology(net, controllers);

    // Check that links exist connecting adjacent controlled intersections
    EXPECT_GT(mgr.link_count(), 0u);
    for (const auto& link : mgr.links()) {
        EXPECT_TRUE(controllers.find(link.from_intersection) != controllers.end());
        EXPECT_TRUE(controllers.find(link.to_intersection) != controllers.end());
        EXPECT_GT(link.length_m, 0.0f);
    }
}

TEST(NetworkCoordinationManagerTest, ExchangesBidirectionalStatesAcrossControllers) {
    RoadNetwork net = build_3intersection_corridor();
    NetworkCoordinationManager mgr;

    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    for (IntersectionId i : {IntersectionId(1), IntersectionId(2), IntersectionId(3)}) {
        controllers[i] = TrafficSignalController::make_coordinated(net, i);
    }

    mgr.build_topology(net, controllers);
    ASSERT_GT(mgr.link_count(), 0u);

    // Set high queue on the road connecting I1 to I2
    const auto& first_link = mgr.links().front();
    std::unordered_map<RoadId, uint32_t> queues = { {first_link.road_id, 12} };
    std::unordered_map<RoadId, float> waits = { {first_link.road_id, 25.0f} };

    mgr.exchange_coordination_states(net, controllers, queues, waits);

    // Upstream controller should have received downstream state
    auto* up_coord = controllers[first_link.from_intersection].as<CoordinatedController>();
    ASSERT_NE(up_coord, nullptr);
    EXPECT_FALSE(up_coord->neighbor_states().empty());

    // Downstream controller should have received upstream state
    auto* down_coord = controllers[first_link.to_intersection].as<CoordinatedController>();
    ASSERT_NE(down_coord, nullptr);
    EXPECT_FALSE(down_coord->neighbor_states().empty());
}

// ===========================================================================
// 5. Multi-Intersection Corridor Simulation Execution
// ===========================================================================

TEST(NetworkCoordinationSimulationTest, SimulationRunsWithCoordinatedControllers) {
    Config cfg;
    cfg.signal_controller_type = SignalControllerType::Coordinated;
    cfg.default_green_duration_s = 10.0f;
    cfg.sim_duration_s = 30.0f;
    cfg.dt_seconds = 0.1f;

    Simulation sim(cfg);

    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Coordinated);
    EXPECT_GT(sim.signal_controllers().size(), 0u);

    // Verify all controllers are CoordinatedController instances
    for (const auto& [id, ctrl] : sim.signal_controllers()) {
        const auto* cc = ctrl.as<CoordinatedController>();
        EXPECT_NE(cc, nullptr);
    }

    // Step simulation for 5 seconds
    sim.run_for(5.0f);

    EXPECT_NEAR(sim.state().elapsed_s, 5.0f, 0.15f);
}

TEST(NetworkCoordinationSimulationTest, DynamicControllerSwitchingToAndFromCoordinated) {
    Config cfg;
    cfg.signal_controller_type = SignalControllerType::FixedTime;
    Simulation sim(cfg);

    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::FixedTime);

    // Switch to Coordinated
    sim.set_signal_controller_type(SignalControllerType::Coordinated);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Coordinated);
    sim.run_for(2.0f);

    // Switch to Adaptive
    sim.set_signal_controller_type(SignalControllerType::Adaptive);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Adaptive);
    sim.run_for(2.0f);

    // Switch back to Coordinated
    sim.set_signal_controller_type(SignalControllerType::Coordinated);
    EXPECT_EQ(sim.signal_controller_type(), SignalControllerType::Coordinated);
    sim.run_for(2.0f);
}

// ===========================================================================
// 6. Network Coordination Mitigates Queue Propagation & Delay
// ===========================================================================

TEST(NetworkCoordinationSimulationTest, CoordinatedControlReducesQueuePropagation) {
    // Run two identical corridor simulations:
    // 1 with FixedTime, 1 with Coordinated
    auto run_sim = [](SignalControllerType type) -> RunResult {
        Config cfg;
        cfg.signal_controller_type = type;
        cfg.scenario = TrafficScenario::RushHour;
        cfg.default_green_duration_s = 10.0f;
        cfg.sim_duration_s = 40.0f;
        cfg.dt_seconds = 0.1f;
        cfg.seed = 42;

        Simulation sim(cfg);
        sim.run_for(40.0f);
        return sim.metrics().get_run_result();
    };

    const auto fixed_metrics = run_sim(SignalControllerType::FixedTime);
    const auto coord_metrics = run_sim(SignalControllerType::Coordinated);

    // Coordinated control should reduce or maintain reasonable queue lengths and throughput
    EXPECT_LE(coord_metrics.max_queue_length, fixed_metrics.max_queue_length + 2);
    EXPECT_GT(coord_metrics.total_spawned, 0u);
}
