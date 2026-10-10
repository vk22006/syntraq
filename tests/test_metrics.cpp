//
// tests/test_metrics.cpp
//
// Unit tests for the traffic metrics subsystem:
// - Vehicle count, average speed, average waiting time, average travel time
// - Queue length, throughput, congestion ratio
// - Reset functionality
// - RunResult creation, scenario & seed metadata
// - CSV exports (summary and timeseries)
// - End-to-end integration with Simulation
//

#include <gtest/gtest.h>

#include "syntraq/metrics/traffic_metrics.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/vehicles/vehicle.h"
#include "syntraq/world/road_network.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace syntraq;

namespace {

Vehicle make_mock_vehicle(uint32_t id,
                          uint32_t road_id,
                          uint32_t lane,
                          float progress,
                          float speed,
                          VehicleState state,
                          float travel_time_s = 0.0f,
                          float wait_time_s = 0.0f,
                          float distance_m = 0.0f) {
    Vehicle v;
    v.id            = static_cast<VehicleId>(id);
    v.current_road  = static_cast<RoadId>(road_id);
    v.lane_index    = lane;
    v.progress_m    = progress;
    v.speed_mps     = speed;
    v.state         = state;
    v.travel_time_s = travel_time_s;
    v.wait_time_s   = wait_time_s;
    v.distance_m    = distance_m;
    return v;
}

} // namespace

// ── Test 1: Vehicle Counts and Snapshot Basics ─────────────────────────────────

TEST(TrafficMetricsTest, VehicleCountsTrackCorrectly) {
    TrafficMetricsCollector metrics;
    metrics.set_scenario(TrafficScenario::Medium, "Medium Traffic");
    metrics.set_seed(12345u);

    std::vector<Vehicle> active;
    active.push_back(make_mock_vehicle(1, 10, 0, 50.0f, 10.0f, VehicleState::Moving));
    active.push_back(make_mock_vehicle(2, 10, 0, 20.0f, 12.0f, VehicleState::Moving));
    active.push_back(make_mock_vehicle(3, 10, 1, 10.0f, 0.0f,  VehicleState::Stopped));

    metrics.record_tick(0.1f, active, /*total_spawned=*/5, /*tick=*/1);

    const auto& snap = metrics.current_snapshot();
    EXPECT_EQ(snap.tick, 1u);
    EXPECT_NEAR(snap.timestamp_s, 0.1f, 1e-4f);
    EXPECT_EQ(snap.active_vehicles, 3u);
    EXPECT_EQ(snap.total_spawned, 5u);
    EXPECT_EQ(snap.total_completed, 0u);
}

// ── Test 2: Average Speed Calculation ─────────────────────────────────────────

TEST(TrafficMetricsTest, AverageSpeedCalculation) {
    TrafficMetricsCollector metrics;

    std::vector<Vehicle> active;
    active.push_back(make_mock_vehicle(1, 1, 0, 10.0f, 10.0f, VehicleState::Moving));
    active.push_back(make_mock_vehicle(2, 1, 0, 30.0f, 20.0f, VehicleState::Moving));

    metrics.record_tick(1.0f, active, 2, 1);

    // Snapshot avg speed: (10 + 20) / 2 = 15.0 m/s
    EXPECT_NEAR(metrics.current_snapshot().avg_speed_mps, 15.0f, 1e-3f);

    const auto result = metrics.get_run_result();
    EXPECT_NEAR(result.avg_speed_mps, 15.0f, 1e-3f);
    EXPECT_NEAR(result.avg_speed_kmh, 15.0f * 3.6f, 1e-3f);
}

// ── Test 3: Average Waiting Time and Travel Time ───────────────────────────────

TEST(TrafficMetricsTest, AverageWaitingAndTravelTimeCalculation) {
    TrafficMetricsCollector metrics;

    // Simulate arrival of two vehicles
    Vehicle arrived1 = make_mock_vehicle(1, 1, 0, 100.0f, 0.0f, VehicleState::Arrived,
                                         /*travel_time_s=*/20.0f, /*wait_time_s=*/5.0f, /*distance_m=*/200.0f);
    Vehicle arrived2 = make_mock_vehicle(2, 1, 0, 100.0f, 0.0f, VehicleState::Arrived,
                                         /*travel_time_s=*/40.0f, /*wait_time_s=*/15.0f, /*distance_m=*/400.0f);

    metrics.record_arrival(arrived1);
    metrics.record_arrival(arrived2);

    std::vector<Vehicle> empty_active;
    metrics.record_tick(1.0f, empty_active, 2, 1);

    const auto result = metrics.get_run_result();
    EXPECT_EQ(result.total_completed, 2u);
    // (20 + 40) / 2 = 30.0 s
    EXPECT_NEAR(result.avg_travel_time_s, 30.0f, 1e-3f);
    // (5 + 15) / 2 = 10.0 s
    EXPECT_NEAR(result.avg_waiting_time_s, 10.0f, 1e-3f);
}

// ── Test 4: Queue Length and Peak Queue Tracking ───────────────────────────────

TEST(TrafficMetricsTest, QueueLengthAndMaxQueueDetection) {
    TrafficMetricsCollector metrics;

    std::vector<Vehicle> active;
    // Lane (Road 5, Lane 0) has 3 queued vehicles
    active.push_back(make_mock_vehicle(1, 5, 0, 10.0f, 0.0f, VehicleState::Stopped));
    active.push_back(make_mock_vehicle(2, 5, 0, 17.0f, 0.2f, VehicleState::Moving)); // speed < 0.5 counts as queued
    active.push_back(make_mock_vehicle(3, 5, 0, 24.0f, 0.0f, VehicleState::Stopped));
    // Lane (Road 5, Lane 1) has 1 queued vehicle
    active.push_back(make_mock_vehicle(4, 5, 1, 10.0f, 0.1f, VehicleState::Stopped));
    // Another vehicle moving fast
    active.push_back(make_mock_vehicle(5, 6, 0, 50.0f, 12.0f, VehicleState::Moving));

    metrics.record_tick(1.0f, active, 5, 1);

    EXPECT_EQ(metrics.current_snapshot().queued_vehicles, 4u);
    EXPECT_EQ(metrics.current_snapshot().max_queue_length, 3u);

    // In a second tick, queue dissipates to 1 vehicle
    active.clear();
    active.push_back(make_mock_vehicle(1, 5, 0, 10.0f, 0.0f, VehicleState::Stopped));
    active.push_back(make_mock_vehicle(5, 6, 0, 80.0f, 12.0f, VehicleState::Moving));

    metrics.record_tick(1.0f, active, 5, 2);

    EXPECT_EQ(metrics.current_snapshot().queued_vehicles, 1u);
    EXPECT_EQ(metrics.current_snapshot().max_queue_length, 1u);

    // Peak queue should remember the maximum (3)
    const auto result = metrics.get_run_result();
    EXPECT_EQ(result.max_queue_length, 3u);
    // Average queued count across 2 ticks: (4 + 1) / 2 = 2.5
    EXPECT_NEAR(result.avg_queue_length, 2.5f, 1e-3f);
}

// ── Test 5: Throughput Calculation ────────────────────────────────────────────

TEST(TrafficMetricsTest, ThroughputCalculation) {
    TrafficMetricsCollector metrics;

    // Simulate 10 vehicles completing trips over 20 seconds
    for (uint32_t i = 1; i <= 10; ++i) {
        metrics.record_arrival(make_mock_vehicle(i, 1, 0, 100.0f, 0.0f, VehicleState::Arrived, 15.0f, 2.0f, 150.0f));
    }

    std::vector<Vehicle> active;
    metrics.record_tick(20.0f, active, 10, 1);

    const auto result = metrics.get_run_result();
    // 10 vehicles in 20 seconds = 0.5 veh/s
    EXPECT_NEAR(result.throughput_vps, 0.5f, 1e-3f);
    // 0.5 * 3600 = 1800.0 veh/h
    EXPECT_NEAR(result.throughput_vph, 1800.0f, 1e-2f);
}

// ── Test 6: Congestion Ratio Calculation ──────────────────────────────────────

TEST(TrafficMetricsTest, CongestionRatioUnderVaryingConditions) {
    TrafficMetricsCollector metrics;

    // Condition 1: Free flow (0 queued out of 4 active) -> ratio = 0.0
    std::vector<Vehicle> active;
    for (uint32_t i = 1; i <= 4; ++i) {
        active.push_back(make_mock_vehicle(i, 1, 0, (float)i * 20.0f, 10.0f, VehicleState::Moving));
    }
    metrics.record_tick(1.0f, active, 4, 1);
    EXPECT_NEAR(metrics.current_snapshot().congestion_ratio, 0.0f, 1e-4f);

    // Condition 2: Half queued (2 queued out of 4 active) -> ratio = 0.5
    active[0].state = VehicleState::Stopped;
    active[0].speed_mps = 0.0f;
    active[1].state = VehicleState::Stopped;
    active[1].speed_mps = 0.0f;
    metrics.record_tick(1.0f, active, 4, 2);
    EXPECT_NEAR(metrics.current_snapshot().congestion_ratio, 0.5f, 1e-4f);

    // Condition 3: Total gridlock (4 queued out of 4 active) -> ratio = 1.0
    active[2].state = VehicleState::Stopped;
    active[2].speed_mps = 0.0f;
    active[3].state = VehicleState::Stopped;
    active[3].speed_mps = 0.0f;
    metrics.record_tick(1.0f, active, 4, 3);
    EXPECT_NEAR(metrics.current_snapshot().congestion_ratio, 1.0f, 1e-4f);

    // Run average: (0.0 + 0.5 + 1.0) / 3 = 0.5
    const auto result = metrics.get_run_result();
    EXPECT_NEAR(result.congestion_ratio, 0.5f, 1e-3f);
}

// ── Test 7: Reset Cleans All Metrics ──────────────────────────────────────────

TEST(TrafficMetricsTest, MetricsResetRestoresZeroState) {
    TrafficMetricsCollector metrics;
    metrics.record_arrival(make_mock_vehicle(1, 1, 0, 100.0f, 0.0f, VehicleState::Arrived, 20.0f, 5.0f, 100.0f));

    std::vector<Vehicle> active;
    active.push_back(make_mock_vehicle(2, 1, 0, 10.0f, 0.0f, VehicleState::Stopped));
    metrics.record_tick(5.0f, active, 2, 1);

    EXPECT_GT(metrics.current_snapshot().total_completed, 0u);
    EXPECT_GT(metrics.timeseries().size(), 0u);

    metrics.reset();

    const auto result = metrics.get_run_result();
    EXPECT_EQ(result.total_ticks, 0u);
    EXPECT_NEAR(result.duration_s, 0.0f, 1e-4f);
    EXPECT_EQ(result.total_spawned, 0u);
    EXPECT_EQ(result.total_completed, 0u);
    EXPECT_EQ(result.active_remaining, 0u);
    EXPECT_NEAR(result.avg_speed_mps, 0.0f, 1e-4f);
    EXPECT_NEAR(result.avg_waiting_time_s, 0.0f, 1e-4f);
    EXPECT_NEAR(result.avg_travel_time_s, 0.0f, 1e-4f);
    EXPECT_EQ(result.max_queue_length, 0u);
    EXPECT_NEAR(result.throughput_vph, 0.0f, 1e-4f);
    EXPECT_NEAR(result.congestion_ratio, 0.0f, 1e-4f);
    EXPECT_TRUE(metrics.timeseries().empty());
}

// ── Test 8: RunResult Stores Scenario and Random Seed ─────────────────────────

TEST(TrafficMetricsTest, RunResultRecordsScenarioAndSeed) {
    TrafficMetricsCollector metrics;
    metrics.set_scenario(TrafficScenario::RushHour, "Rush Hour");
    metrics.set_seed(99999u);

    std::vector<Vehicle> active;
    metrics.record_tick(1.0f, active, 0, 1);

    const auto result = metrics.get_run_result();
    EXPECT_EQ(result.scenario, TrafficScenario::RushHour);
    EXPECT_EQ(result.scenario_name, "Rush Hour");
    EXPECT_EQ(result.seed, 99999u);
}

// ── Test 9: CSV Summary Export ────────────────────────────────────────────────

TEST(TrafficMetricsTest, CsvSummaryExportWritesValidFile) {
    const std::string path = "test_metrics_summary.csv";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    TrafficMetricsCollector metrics;
    metrics.set_scenario(TrafficScenario::High, "High Traffic");
    metrics.set_seed(42u);
    metrics.record_arrival(make_mock_vehicle(1, 1, 0, 100.0f, 0.0f, VehicleState::Arrived, 12.5f, 3.2f, 150.0f));

    std::vector<Vehicle> active;
    metrics.record_tick(5.0f, active, 1, 1);

    EXPECT_TRUE(metrics.export_summary_csv(path));
    EXPECT_TRUE(std::filesystem::exists(path));

    // Verify file contents
    std::ifstream file(path);
    std::string header_line;
    std::string data_line;
    ASSERT_TRUE(std::getline(file, header_line));
    ASSERT_TRUE(std::getline(file, data_line));
    file.close();

    EXPECT_EQ(header_line, RunResult::csv_header());
    EXPECT_NE(data_line.find("High Traffic"), std::string::npos);
    EXPECT_NE(data_line.find("42"), std::string::npos);

    std::filesystem::remove(path);
}

// ── Test 10: CSV Time-Series Export ───────────────────────────────────────────

TEST(TrafficMetricsTest, CsvTimeSeriesExportWritesChronologicalRows) {
    const std::string path = "test_metrics_timeseries.csv";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    TrafficMetricsCollector metrics;
    std::vector<Vehicle> active;
    active.push_back(make_mock_vehicle(1, 1, 0, 10.0f, 8.0f, VehicleState::Moving));

    metrics.record_tick(0.5f, active, 1, 1);
    metrics.record_tick(0.5f, active, 1, 2);
    metrics.record_tick(0.5f, active, 1, 3);

    EXPECT_TRUE(metrics.export_timeseries_csv(path));
    EXPECT_TRUE(std::filesystem::exists(path));

    std::ifstream file(path);
    std::string line;
    int row_count = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) row_count++;
    }
    file.close();

    // 1 header + 3 tick snapshots = 4 lines
    EXPECT_EQ(row_count, 4);

    std::filesystem::remove(path);
}

// ── Test 11: End-to-End Simulation Metrics Integration ────────────────────────

TEST(TrafficMetricsTest, SimulationTracksAndUpdatesMetricsDuringRun) {
    Config cfg;
    cfg.scenario                 = TrafficScenario::Medium;
    cfg.seed                     = 42u;
    cfg.vehicle_spawn_interval_s = 0.5f;
    cfg.max_vehicles             = 20;
    cfg.dt_seconds               = 0.1f;

    Simulation sim(cfg);
    sim.run_for(10.0f); // 100 ticks

    const auto& s = sim.state();
    EXPECT_GT(s.total_spawned, 0u);
    EXPECT_GT(s.active_vehicles, 0u);

    // Verify metrics subsystem matches Simulation state
    const auto result = sim.get_run_result();
    EXPECT_EQ(result.scenario, TrafficScenario::Medium);
    EXPECT_EQ(result.seed, 42u);
    EXPECT_NEAR(result.duration_s, 10.0f, 0.2f);
    EXPECT_EQ(result.total_spawned, s.total_spawned);
    EXPECT_EQ(result.total_completed, s.total_arrived);
    EXPECT_EQ(result.active_remaining, s.active_vehicles);

    // Average speed should be positive since vehicles are driving
    EXPECT_GT(result.avg_speed_mps, 0.0f);
    EXPECT_GT(result.avg_speed_kmh, 0.0f);

    // Simulation reset must reset metrics too
    sim.reset();
    const auto reset_result = sim.get_run_result();
    EXPECT_EQ(reset_result.total_spawned, 0u);
    EXPECT_EQ(reset_result.total_completed, 0u);
    EXPECT_EQ(reset_result.total_ticks, 0u);
}
