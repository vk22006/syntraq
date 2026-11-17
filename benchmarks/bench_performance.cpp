//
// benchmarks/bench_performance.cpp
//
// Comprehensive profiling and benchmark harness for the SyntraQ simulation engine.
// Measures:
//   1. Simulation tick time (mean / p95 / p99 us) and vehicles/second across workloads
//   2. Per-subsystem tick profiling breakdown (movement, routing, emergency, spawner, metrics, signals)
//   3. Pathfinding latency (Dijkstra, A*, StaticRouting, TrafficAwareRouting)
//   4. Native AI inference latency (RLPolicyInference & TrafficPredictor single + batch)
//   5. Memory usage (working set RSS, peak RSS, and object sizes)
//   6. AoS vs SoA (ECS-style) microbenchmark to empirically evaluate ECS suitability
//

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "syntraq/ai/rl_policy_inference.hpp"
#include "syntraq/ai/traffic_predictor.hpp"
#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/routing/astar_router.hpp"
#include "syntraq/routing/dijkstra_router.hpp"
#include "syntraq/routing/graph_generator.hpp"
#include "syntraq/routing/parallel_routing.hpp"
#include "syntraq/routing/static_routing.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/signals/network_coordination_manager.hpp"
#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/vehicles/emergency_priority_manager.hpp"
#include "syntraq/vehicles/vehicle_movement_system.hpp"
#include "syntraq/vehicles/vehicle_spawner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace syntraq {

namespace {

using Clock = std::chrono::steady_clock;

struct MemStats {
    double rss_mb{ 0.0 };
    double peak_rss_mb{ 0.0 };
};

MemStats get_memory_stats() {
    MemStats m{};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        m.rss_mb      = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        m.peak_rss_mb = static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#else
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        m.peak_rss_mb = static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
        m.peak_rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
        m.rss_mb = m.peak_rss_mb;
    }
#endif
    return m;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(idx));
    const auto hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return values[lo];
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

RoadNetwork make_multilane_grid(int cols, int rows, float spacing_m, uint32_t lanes) {
    RoadNetwork net;
    std::vector<std::vector<IntersectionId>> ids(
        static_cast<size_t>(rows),
        std::vector<IntersectionId>(static_cast<size_t>(cols)));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ids[static_cast<size_t>(r)][static_cast<size_t>(c)] =
                net.add_intersection(Vec2{ c * spacing_m, r * spacing_m },
                                     "I(" + std::to_string(c) + "," + std::to_string(r) + ")");
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols - 1; ++c) {
            net.add_road(ids[static_cast<size_t>(r)][static_cast<size_t>(c)],
                         ids[static_cast<size_t>(r)][static_cast<size_t>(c + 1)],
                         spacing_m, 13.89f, lanes);
            net.add_road(ids[static_cast<size_t>(r)][static_cast<size_t>(c + 1)],
                         ids[static_cast<size_t>(r)][static_cast<size_t>(c)],
                         spacing_m, 13.89f, lanes);
        }
    }
    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols; ++c) {
            net.add_road(ids[static_cast<size_t>(r)][static_cast<size_t>(c)],
                         ids[static_cast<size_t>(r + 1)][static_cast<size_t>(c)],
                         spacing_m, 13.89f, lanes);
            net.add_road(ids[static_cast<size_t>(r + 1)][static_cast<size_t>(c)],
                         ids[static_cast<size_t>(r)][static_cast<size_t>(c)],
                         spacing_m, 13.89f, lanes);
        }
    }
    return net;
}

std::string resolve_model_path(const std::string& rel) {
    if (std::filesystem::exists(rel)) return rel;
    if (std::filesystem::exists("../" + rel)) return "../" + rel;
    if (std::filesystem::exists("../../" + rel)) return "../../" + rel;
    return rel;
}

struct CsvRow {
    std::string category;
    std::string benchmark;
    std::string metric;
    double      value{ 0.0 };
    std::string unit;
};

struct WorkloadResult {
    std::string name;
    uint32_t    ticks{ 0 };
    double      mean_tick_us{ 0.0 };
    double      p95_tick_us{ 0.0 };
    double      p99_tick_us{ 0.0 };
    double      avg_active_vehicles{ 0.0 };
    uint32_t    peak_active_vehicles{ 0 };
    double      vehicles_per_sec{ 0.0 };
    double      sim_ticks_per_sec{ 0.0 };
    double      rss_mb{ 0.0 };
    double      peak_rss_mb{ 0.0 };
    uint32_t    total_arrived{ 0 };
    float       avg_speed_mps{ 0.0f };
};

WorkloadResult run_simulation_workload(
    const std::string& name,
    Config cfg,
    std::function<RoadNetwork()> factory,
    uint32_t num_ticks,
    uint32_t pre_spawn_vehicles,
    bool trigger_emergency)
{
    Simulation sim(cfg, std::move(factory));

    // Pre-populate vehicles so steady-state vehicle density is reached quickly
    if (pre_spawn_vehicles > 0) {
        const float orig_interval = sim.config().vehicle_spawn_interval_s;
        sim.set_spawn_interval(0.01f);
        for (uint32_t i = 0; i < pre_spawn_vehicles && sim.vehicles().size() < pre_spawn_vehicles; ++i) {
            sim.tick();
        }
        sim.set_spawn_interval(orig_interval);
    }

    if (trigger_emergency) {
        sim.trigger_emergency_vehicle();
    }

    std::vector<double> tick_times_us;
    tick_times_us.reserve(num_ticks);

    uint64_t total_vehicle_steps = 0;
    uint32_t peak_vehicles = 0;

    const auto wall_start = Clock::now();
    for (uint32_t t = 0; t < num_ticks; ++t) {
        if (trigger_emergency && t % 150 == 75) {
            sim.trigger_emergency_vehicle();
        }
        const auto t0 = Clock::now();
        sim.tick();
        const auto t1 = Clock::now();

        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        tick_times_us.push_back(us);

        const uint32_t active = static_cast<uint32_t>(sim.vehicles().size());
        total_vehicle_steps += active;
        if (active > peak_vehicles) peak_vehicles = active;
    }
    const auto wall_end = Clock::now();
    const double total_wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    const double sum_us = std::accumulate(tick_times_us.begin(), tick_times_us.end(), 0.0);
    const MemStats mem = get_memory_stats();

    WorkloadResult res;
    res.name                 = name;
    res.ticks                = num_ticks;
    res.mean_tick_us         = sum_us / static_cast<double>(num_ticks);
    res.p95_tick_us          = percentile(tick_times_us, 95.0);
    res.p99_tick_us          = percentile(tick_times_us, 99.0);
    res.avg_active_vehicles  = static_cast<double>(total_vehicle_steps) / static_cast<double>(num_ticks);
    res.peak_active_vehicles = peak_vehicles;
    res.vehicles_per_sec     = total_wall_s > 0.0 ? (static_cast<double>(total_vehicle_steps) / total_wall_s) : 0.0;
    res.sim_ticks_per_sec    = total_wall_s > 0.0 ? (static_cast<double>(num_ticks) / total_wall_s) : 0.0;
    res.rss_mb               = mem.rss_mb;
    res.peak_rss_mb          = mem.peak_rss_mb;
    res.total_arrived        = sim.state().total_arrived;
    res.avg_speed_mps        = sim.state().avg_speed_mps;
    return res;
}

void run_subsystem_profile(std::vector<CsvRow>& rows) {
    std::cout << "\n[2] SUBSYSTEM BREAKDOWN PROFILE (12x12 Grid, 1000 Active Vehicles, 200 Ticks)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    RoadNetwork net = make_multilane_grid(12, 12, 120.0f, 2);
    std::unordered_map<IntersectionId, TrafficSignalController> controllers;
    for (const auto& [id, inter] : net.intersections()) {
        if (net.incoming_roads(id).size() > 1) {
            CoordinatedControllerConfig cc_cfg;
            auto c = TrafficSignalController::make_coordinated(net, id, cc_cfg, 3.0f, 1.0f);
            if (c.is_valid()) controllers[id] = std::move(c);
        }
    }
    NetworkCoordinationManager coord_mgr;
    coord_mgr.build_topology(net, controllers);

    auto ta_router = std::make_shared<TrafficAwareRouting>();
    VehicleSpawner spawner(42, 0.01f, 1000, ta_router);
    VehicleMovementSystem movement;
    EmergencyPriorityManager emerg_mgr;
    TrafficMetricsCollector metrics;

    std::vector<Vehicle> vehicles;
    vehicles.reserve(1000);

    // Populate 1000 vehicles deterministically across grid
    float sim_t = 0.0f;
    for (uint32_t i = 0; i < 2000 && vehicles.size() < 1000; ++i) {
        sim_t += 0.02f;
        auto v = spawner.try_spawn(net, vehicles, sim_t);
        if (v.has_value()) vehicles.push_back(std::move(*v));
        if (i % 10 == 0) {
            movement.update(vehicles, net, controllers, 0.1f);
        }
    }
    // Add 3 emergency vehicles
    for (int e = 0; e < 3; ++e) {
        auto ev = spawner.spawn_emergency_vehicle(
            net,
            static_cast<IntersectionId>(e * 12),
            static_cast<IntersectionId>(143 - e * 12),
            vehicles);
        if (ev.has_value()) vehicles.push_back(std::move(*ev));
    }

    std::cout << "Populated " << vehicles.size() << " vehicles on "
              << net.intersection_count() << " intersections / " << net.road_count() << " roads.\n";

    spawner.spawn_interval_s = 0.1f;
    spawner.max_vehicles = 1200;

    double t_queue_agg_us   = 0.0;
    double t_coord_sig_us   = 0.0;
    double t_ta_update_us   = 0.0;
    double t_emergency_us   = 0.0;
    double t_spawner_us     = 0.0;
    double t_movement_us    = 0.0;
    double t_metrics_us     = 0.0;

    constexpr uint32_t kProfileTicks = 200;
    const float dt = 0.1f;

    for (uint32_t tick = 0; tick < kProfileTicks; ++tick) {
        sim_t += dt;

        // 1. Queue aggregation
        auto s0 = Clock::now();
        std::unordered_map<RoadId, uint32_t> road_queues;
        std::unordered_map<RoadId, float> road_max_waits;
        for (const auto& v : vehicles) {
            if (v.state == VehicleState::Arrived) continue;
            if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
                road_queues[v.current_road]++;
                if (v.wait_time_s > road_max_waits[v.current_road]) {
                    road_max_waits[v.current_road] = v.wait_time_s;
                }
            }
        }
        auto s1 = Clock::now();

        // 2. Coordination & Signal controllers
        coord_mgr.exchange_coordination_states(net, controllers, road_queues, road_max_waits);
        for (auto& [id, ctrl] : controllers) {
            ctrl.observe_traffic(road_queues, road_max_waits);
            ctrl.tick(dt);
        }
        auto s2 = Clock::now();

        // 3. TrafficAwareRouting condition update
        ta_router->update_from_simulation(net, vehicles);
        auto s3 = Clock::now();

        // 4. Emergency priority update
        emerg_mgr.update(net, vehicles, controllers, ta_router.get(), dt);
        auto s4 = Clock::now();

        // 5. Vehicle spawner
        auto spawned = spawner.try_spawn(net, vehicles, sim_t);
        if (spawned.has_value()) {
            vehicles.push_back(std::move(*spawned));
        }
        auto s5 = Clock::now();

        // 6. Vehicle movement system
        movement.update(vehicles, net, controllers, dt);
        auto s6 = Clock::now();

        // 7. Metrics collector
        metrics.record_tick(dt, vehicles, static_cast<uint32_t>(vehicles.size()), tick + 1);
        auto s7 = Clock::now();

        t_queue_agg_us += std::chrono::duration<double, std::micro>(s1 - s0).count();
        t_coord_sig_us += std::chrono::duration<double, std::micro>(s2 - s1).count();
        t_ta_update_us += std::chrono::duration<double, std::micro>(s3 - s2).count();
        t_emergency_us += std::chrono::duration<double, std::micro>(s4 - s3).count();
        t_spawner_us   += std::chrono::duration<double, std::micro>(s5 - s4).count();
        t_movement_us  += std::chrono::duration<double, std::micro>(s6 - s5).count();
        t_metrics_us   += std::chrono::duration<double, std::micro>(s7 - s6).count();
    }

    const double inv = 1.0 / static_cast<double>(kProfileTicks);
    const double avg_queue  = t_queue_agg_us * inv;
    const double avg_sig    = t_coord_sig_us * inv;
    const double avg_ta     = t_ta_update_us * inv;
    const double avg_emerg  = t_emergency_us * inv;
    const double avg_spawn  = t_spawner_us   * inv;
    const double avg_move   = t_movement_us  * inv;
    const double avg_met    = t_metrics_us   * inv;
    const double total_us   = avg_queue + avg_sig + avg_ta + avg_emerg + avg_spawn + avg_move + avg_met;

    auto print_sub = [&](const std::string& label, const std::string& key, double us) {
        const double pct = total_us > 0.0 ? (100.0 * us / total_us) : 0.0;
        std::cout << "  " << std::left << std::setw(34) << label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << us << " us/tick"
                  << " (" << std::setw(5) << std::setprecision(1) << pct << "%)\n";
        rows.push_back({ "subsystem", key, "mean_us", us, "us" });
    };

    print_sub("VehicleMovementSystem::update",     "movement",      avg_move);
    print_sub("VehicleSpawner::try_spawn",         "spawner",       avg_spawn);
    print_sub("TrafficAware::update_from_sim",     "ta_update",     avg_ta);
    print_sub("TrafficMetrics::record_tick",       "metrics",       avg_met);
    print_sub("Signals + Coordination",            "signals_coord", avg_sig);
    print_sub("EmergencyPriorityManager::update",  "emergency",     avg_emerg);
    print_sub("Queue Aggregation",                 "queue_agg",     avg_queue);
    print_sub("TOTAL (Measured Tick Sum)",         "total_sum",     total_us);
}

void run_pathfinding_benchmarks(std::vector<CsvRow>& rows) {
    std::cout << "\n[3] PATHFINDING BENCHMARKS\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // A) 25x25 Grid (625 nodes) - Cold & Repeated Queries
    RoadNetwork grid25 = make_grid_network(25, 25, 100.0f);
    std::mt19937 rng(2026);
    std::uniform_int_distribution<uint32_t> dist25(0, grid25.intersection_count() - 1);

    // 200 distinct random pairs
    std::vector<std::pair<IntersectionId, IntersectionId>> pairs25;
    pairs25.reserve(200);
    for (int i = 0; i < 200; ++i) {
        IntersectionId s = static_cast<IntersectionId>(dist25(rng));
        IntersectionId d = static_cast<IntersectionId>(dist25(rng));
        if (s == d) d = static_cast<IntersectionId>((static_cast<uint32_t>(d) + 1) % grid25.intersection_count());
        pairs25.emplace_back(s, d);
    }

    // 1000 repeated O-D queries drawn from 40 popular corridors (realistic urban demand)
    std::vector<std::pair<IntersectionId, IntersectionId>> repeated_queries;
    repeated_queries.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        repeated_queries.push_back(pairs25[static_cast<size_t>(i % 40)]);
    }

    DijkstraRouter dijkstra;
    AStarRouter astar;
    StaticRouting static_routing(StaticRouting::Algorithm::AStar);
    TrafficAwareRouting ta_routing;

    // 1. Dijkstra cold 25x25
    {
        auto t0 = Clock::now();
        size_t total_len = 0;
        for (const auto& [s, d] : pairs25) {
            total_len += dijkstra.find_route_with_info(grid25, s, d).roads.size();
        }
        auto t1 = Clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(pairs25.size());
        std::cout << "  Dijkstra 25x25 (200 random pairs)          : " << std::fixed << std::setprecision(2)
                  << std::setw(8) << avg_us << " us/query (edges=" << total_len << ")\n";
        rows.push_back({ "pathfinding", "dijkstra_25x25_random", "avg_us", avg_us, "us" });
    }

    // 2. A* cold 25x25
    {
        auto t0 = Clock::now();
        size_t total_len = 0;
        for (const auto& [s, d] : pairs25) {
            total_len += astar.find_route_with_info(grid25, s, d).roads.size();
        }
        auto t1 = Clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(pairs25.size());
        std::cout << "  A* 25x25 (200 random pairs)                : " << std::fixed << std::setprecision(2)
                  << std::setw(8) << avg_us << " us/query (edges=" << total_len << ")\n";
        rows.push_back({ "pathfinding", "astar_25x25_random", "avg_us", avg_us, "us" });
    }

    // 3. A* 50x50 (2500 nodes)
    {
        RoadNetwork grid50 = make_grid_network(50, 50, 100.0f);
        std::uniform_int_distribution<uint32_t> dist50(0, grid50.intersection_count() - 1);
        std::vector<std::pair<IntersectionId, IntersectionId>> pairs50;
        for (int i = 0; i < 60; ++i) {
            pairs50.emplace_back(static_cast<IntersectionId>(dist50(rng)),
                                 static_cast<IntersectionId>(dist50(rng)));
        }
        auto t0 = Clock::now();
        size_t total_len = 0;
        for (const auto& [s, d] : pairs50) {
            total_len += astar.find_route_with_info(grid50, s, d).roads.size();
        }
        auto t1 = Clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(pairs50.size());
        std::cout << "  A* 50x50 (60 random pairs)                 : " << std::fixed << std::setprecision(2)
                  << std::setw(8) << avg_us << " us/query (edges=" << total_len << ")\n";
        rows.push_back({ "pathfinding", "astar_50x50_random", "avg_us", avg_us, "us" });

        // 3b. Parallel batch A* 50x50 (same 60 random pairs across CPU cores)
        auto tp0 = Clock::now();
        auto par_res = find_routes_parallel(astar, grid50, pairs50);
        auto tp1 = Clock::now();
        size_t par_len = 0;
        for (const auto& r : par_res) par_len += r.roads.size();
        double par_avg_us = std::chrono::duration<double, std::micro>(tp1 - tp0).count() / static_cast<double>(pairs50.size());
        std::cout << "  A* 50x50 Parallel Batch (60 pairs, MT)     : " << std::fixed << std::setprecision(2)
                  << std::setw(8) << par_avg_us << " us/query (edges=" << par_len << ")\n";
        rows.push_back({ "pathfinding", "astar_50x50_parallel", "avg_us", par_avg_us, "us" });
    }

    // 4. StaticRouting repeated O-D workload (1000 queries across 40 pairs on 25x25)
    {
        auto t0 = Clock::now();
        size_t total_len = 0;
        for (const auto& [s, d] : repeated_queries) {
            total_len += static_routing.find_route(grid25, s, d).size();
        }
        auto t1 = Clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(repeated_queries.size());
        std::cout << "  StaticRouting 25x25 (1000 repeated O-D)    : " << std::fixed << std::setprecision(2)
                  << std::setw(8) << avg_us << " us/query (edges=" << total_len << ")\n";
        rows.push_back({ "pathfinding", "static_25x25_repeated", "avg_us", avg_us, "us" });
    }

    // 5. TrafficAwareRouting repeated workload (1000 queries, traffic updated every 50 queries)
    {
        std::unordered_map<RoadId, RoadTrafficCondition> conds;
        for (uint32_t r = 0; r < grid25.road_count(); r += 5) {
            conds[static_cast<RoadId>(r)] = RoadTrafficCondition{ 8, 5, 4.0f, 0.625f };
        }
        ta_routing.set_road_conditions(conds);

        auto t0 = Clock::now();
        size_t total_len = 0;
        for (size_t i = 0; i < repeated_queries.size(); ++i) {
            if (i > 0 && i % 50 == 0) {
                conds[static_cast<RoadId>((i * 3) % grid25.road_count())].queued_vehicles =
                    static_cast<uint32_t>((i % 10) + 1);
                ta_routing.set_road_conditions(conds);
            }
            const auto& [s, d] = repeated_queries[i];
            total_len += ta_routing.find_route(grid25, s, d).size();
        }
        auto t1 = Clock::now();
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(repeated_queries.size());
        std::cout << "  TrafficAware 25x25 (1000 queries, periodic): " << std::fixed << std::setprecision(2)
                  << std::setw(8) << avg_us << " us/query (edges=" << total_len << ")\n";
        rows.push_back({ "pathfinding", "traffic_aware_25x25_periodic", "avg_us", avg_us, "us" });
    }
}

void run_ai_inference_benchmarks(std::vector<CsvRow>& rows) {
    std::cout << "\n[4] AI INFERENCE LATENCY BENCHMARKS (ONNX Runtime Native C++)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // 1. RLPolicyInference
    RLPolicyInference rl;
    const std::string rl_path = resolve_model_path("models/traffic_signal_dqn.onnx");
    if (rl.loadModel(rl_path)) {
        std::vector<float> obs(RLPolicyInference::OBSERVATION_DIM, 0.35f);
        // Warmup
        for (int i = 0; i < 20; ++i) (void)rl.predict(obs);
        rl.resetMetrics();

        constexpr int kSingleRuns = 2000;
        std::vector<double> lat_us;
        lat_us.reserve(kSingleRuns);
        int action_sum = 0;
        for (int i = 0; i < kSingleRuns; ++i) {
            obs[0] = static_cast<float>(i % 20) / 20.0f;
            auto t0 = Clock::now();
            auto dec = rl.predict(obs);
            auto t1 = Clock::now();
            action_sum += dec.best_action;
            lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        const double mean_us = std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / static_cast<double>(lat_us.size());
        const double p95_us  = percentile(lat_us, 95.0);
        const double p99_us  = percentile(lat_us, 99.0);
        const double dps     = mean_us > 0.0 ? (1e6 / mean_us) : 0.0;

        std::cout << "  RLPolicy single (N=" << kSingleRuns << ", sum=" << action_sum << "): "
                  << std::fixed << std::setprecision(2)
                  << "mean=" << mean_us << " us | p95=" << p95_us << " us | p99=" << p99_us
                  << " us | " << std::setprecision(0) << dps << " decisions/s\n";
        rows.push_back({ "ai_inference", "rl_single", "mean_us", mean_us, "us" });
        rows.push_back({ "ai_inference", "rl_single", "p95_us", p95_us, "us" });
        rows.push_back({ "ai_inference", "rl_single", "decisions_per_sec", dps, "ops/s" });

        // Batch of 36 intersections
        constexpr size_t kBatchSize = 36;
        std::vector<std::vector<float>> batch_obs(kBatchSize, obs);
        for (size_t i = 0; i < kBatchSize; ++i) {
            batch_obs[i][0] = static_cast<float>(i % 10) / 10.0f;
            batch_obs[i][1] = static_cast<float>((i * 3) % 10) / 10.0f;
        }
        for (int i = 0; i < 10; ++i) (void)rl.predictBatch(batch_obs);

        constexpr int kBatchRuns = 200;
        std::vector<double> batch_lat_us;
        batch_lat_us.reserve(kBatchRuns);
        for (int i = 0; i < kBatchRuns; ++i) {
            auto t0 = Clock::now();
            auto decs = rl.predictBatch(batch_obs);
            auto t1 = Clock::now();
            action_sum += decs.empty() ? 0 : decs[0].best_action;
            batch_lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        const double b_mean = std::accumulate(batch_lat_us.begin(), batch_lat_us.end(), 0.0) / static_cast<double>(batch_lat_us.size());
        const double per_item = b_mean / static_cast<double>(kBatchSize);
        const double b_dps = per_item > 0.0 ? (1e6 / per_item) : 0.0;

        std::cout << "  RLPolicy batch-36 (N=" << kBatchRuns << ")           : "
                  << std::fixed << std::setprecision(2)
                  << "total=" << b_mean << " us | per-agent=" << per_item << " us | "
                  << std::setprecision(0) << b_dps << " decisions/s\n";
        rows.push_back({ "ai_inference", "rl_batch36", "total_us", b_mean, "us" });
        rows.push_back({ "ai_inference", "rl_batch36", "per_agent_us", per_item, "us" });
        rows.push_back({ "ai_inference", "rl_batch36", "decisions_per_sec", b_dps, "ops/s" });
    } else {
        std::cout << "  [SKIP] RLPolicyInference model not found at " << rl_path << "\n";
    }

    // 2. TrafficPredictor
    TrafficPredictor pred;
    const std::string pred_path = resolve_model_path("models/traffic_predictor.onnx");
    if (pred.loadModel(pred_path)) {
        std::vector<float> feats(TrafficFeatureScaler::TOTAL_FEATURES, 0.25f);
        for (int i = 0; i < 20; ++i) (void)pred.predict(feats);
        pred.resetMetrics();

        constexpr int kSingleRuns = 1000;
        std::vector<double> lat_us;
        lat_us.reserve(kSingleRuns);
        float pred_sum = 0.0f;
        for (int i = 0; i < kSingleRuns; ++i) {
            feats[0] = static_cast<float>(i % 20) / 20.0f;
            auto t0 = Clock::now();
            pred_sum += pred.predict(feats);
            auto t1 = Clock::now();
            lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        const double mean_us = std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / static_cast<double>(lat_us.size());
        const double p95_us  = percentile(lat_us, 95.0);
        std::cout << "  TrafficPredictor single (N=" << kSingleRuns << ", sum=" << std::setprecision(1) << pred_sum << "): "
                  << std::fixed << std::setprecision(2)
                  << "mean=" << mean_us << " us | p95=" << p95_us << " us\n";
        rows.push_back({ "ai_inference", "predictor_single", "mean_us", mean_us, "us" });
    } else {
        std::cout << "  [SKIP] TrafficPredictor model not found at " << pred_path << "\n";
    }
}

void run_ecs_evaluation_microbenchmark(std::vector<CsvRow>& rows) {
    std::cout << "\n[5] MEMORY FOOTPRINT & ECS (SoA vs AoS) WORKLOAD EVALUATION\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  sizeof(Vehicle)        : " << sizeof(Vehicle) << " bytes\n";
    std::cout << "  sizeof(Road)           : " << sizeof(Road) << " bytes\n";
    std::cout << "  sizeof(Intersection)   : " << sizeof(Intersection) << " bytes\n";
    std::cout << "  sizeof(MetricSnapshot) : " << sizeof(MetricSnapshot) << " bytes\n";

    constexpr size_t kN = 5000;
    constexpr int kIters = 1000;
    const float dt = 0.1f;

    // AoS layout (current std::vector<Vehicle>)
    std::vector<Vehicle> aos(kN);
    for (size_t i = 0; i < kN; ++i) {
        aos[i].speed_mps     = 8.0f;
        aos[i].max_speed_mps = 13.89f;
        aos[i].accel_mps2    = 3.0f;
        aos[i].progress_m    = static_cast<float>(i % 100);
    }

    auto t0 = Clock::now();
    for (int it = 0; it < kIters; ++it) {
        for (size_t i = 0; i < kN; ++i) {
            auto& v = aos[i];
            if (v.speed_mps < v.max_speed_mps) {
                v.speed_mps = std::min(v.max_speed_mps, v.speed_mps + v.accel_mps2 * dt);
            }
            const float step = v.speed_mps * dt;
            v.progress_m    += step;
            v.distance_m    += step;
            v.travel_time_s += dt;
        }
    }
    auto t1 = Clock::now();
    const double aos_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(kIters);

    // SoA layout (ECS-style packed float arrays)
    std::vector<float> speed(kN, 8.0f), max_speed(kN, 13.89f), accel(kN, 3.0f);
    std::vector<float> progress(kN, 0.0f), dist(kN, 0.0f), travel(kN, 0.0f);
    for (size_t i = 0; i < kN; ++i) progress[i] = static_cast<float>(i % 100);

    auto t2 = Clock::now();
    for (int it = 0; it < kIters; ++it) {
        for (size_t i = 0; i < kN; ++i) {
            float s = speed[i];
            const float ms = max_speed[i];
            if (s < ms) s = std::min(ms, s + accel[i] * dt);
            speed[i] = s;
            const float step = s * dt;
            progress[i] += step;
            dist[i]     += step;
            travel[i]   += dt;
        }
    }
    auto t3 = Clock::now();
    const double soa_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / static_cast<double>(kIters);

    // Prevent dead-code elimination
    volatile float sink = aos[0].progress_m + progress[0];
    (void)sink;

    std::cout << "  Pure kinematic integration (5,000 vehicles):\n";
    std::cout << "    AoS (std::vector<Vehicle>) : " << std::fixed << std::setprecision(2) << aos_us << " us/tick\n";
    std::cout << "    SoA (ECS-style arrays)     : " << std::fixed << std::setprecision(2) << soa_us << " us/tick"
              << " (delta: " << (aos_us - soa_us) << " us, "
              << (soa_us > 0.0 ? aos_us / soa_us : 1.0) << "x)\n";

    rows.push_back({ "ecs_eval", "aos_5000_kinematics", "mean_us", aos_us, "us" });
    rows.push_back({ "ecs_eval", "soa_5000_kinematics", "mean_us", soa_us, "us" });
    rows.push_back({ "memory", "sizeof_vehicle", "bytes", static_cast<double>(sizeof(Vehicle)), "B" });
}

} // namespace

} // namespace syntraq

int main(int argc, char** argv) {
    using namespace syntraq;

    std::string csv_path;
    std::string label = "run";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--label" && i + 1 < argc) {
            label = argv[++i];
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "            SyntraQ Performance Profiler & Benchmark Suite [" << label << "]\n";
    std::cout << "================================================================================\n\n";

    std::vector<CsvRow> csv_rows;

    std::cout << "[1] END-TO-END SIMULATION WORKLOADS\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Workload A: Standard 4x3 Rush Hour (Adaptive + Static Routing)
    {
        Config cfg;
        cfg.scenario = TrafficScenario::RushHour;
        cfg.seed = 42;
        cfg.signal_controller_type = SignalControllerType::Adaptive;
        cfg.routing_strategy = RoutingStrategyType::Static;
        auto r = run_simulation_workload("W1_Standard_RushHour_4x3", cfg,
                                         [] { return make_test_map(); },
                                         1000, 0, false);
        std::cout << "  " << std::left << std::setw(32) << r.name
                  << " | tick=" << std::right << std::setw(8) << std::fixed << std::setprecision(2) << r.mean_tick_us << " us"
                  << " (p95=" << std::setw(7) << r.p95_tick_us << ")"
                  << " | " << std::setw(9) << std::setprecision(0) << r.vehicles_per_sec << " veh/s"
                  << " | avgVeh=" << std::setprecision(1) << r.avg_active_vehicles
                  << " | RSS=" << std::setprecision(1) << r.rss_mb << " MB"
                  << " | arrived=" << r.total_arrived << "\n";
        csv_rows.push_back({ "simulation", r.name, "mean_tick_us", r.mean_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p95_tick_us", r.p95_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p99_tick_us", r.p99_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "vehicles_per_sec", r.vehicles_per_sec, "veh/s" });
        csv_rows.push_back({ "simulation", r.name, "ticks_per_sec", r.sim_ticks_per_sec, "ticks/s" });
        csv_rows.push_back({ "simulation", r.name, "rss_mb", r.rss_mb, "MB" });
        csv_rows.push_back({ "simulation", r.name, "total_arrived", static_cast<double>(r.total_arrived), "count" });
    }

    // Workload B: Scaled 10x10 Urban Grid (500 vehicles, Adaptive + TrafficAware)
    {
        Config cfg;
        cfg.scenario = TrafficScenario::Custom;
        cfg.seed = 42;
        cfg.vehicle_spawn_interval_s = 0.1f;
        cfg.max_vehicles = 500;
        cfg.signal_controller_type = SignalControllerType::Adaptive;
        cfg.routing_strategy = RoutingStrategyType::TrafficAware;
        auto r = run_simulation_workload("W2_UrbanGrid_10x10_500v", cfg,
                                         [] { return make_multilane_grid(10, 10, 120.0f, 2); },
                                         400, 500, false);
        std::cout << "  " << std::left << std::setw(32) << r.name
                  << " | tick=" << std::right << std::setw(8) << std::fixed << std::setprecision(2) << r.mean_tick_us << " us"
                  << " (p95=" << std::setw(7) << r.p95_tick_us << ")"
                  << " | " << std::setw(9) << std::setprecision(0) << r.vehicles_per_sec << " veh/s"
                  << " | avgVeh=" << std::setprecision(1) << r.avg_active_vehicles
                  << " | RSS=" << std::setprecision(1) << r.rss_mb << " MB"
                  << " | arrived=" << r.total_arrived << "\n";
        csv_rows.push_back({ "simulation", r.name, "mean_tick_us", r.mean_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p95_tick_us", r.p95_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p99_tick_us", r.p99_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "vehicles_per_sec", r.vehicles_per_sec, "veh/s" });
        csv_rows.push_back({ "simulation", r.name, "ticks_per_sec", r.sim_ticks_per_sec, "ticks/s" });
        csv_rows.push_back({ "simulation", r.name, "rss_mb", r.rss_mb, "MB" });
        csv_rows.push_back({ "simulation", r.name, "total_arrived", static_cast<double>(r.total_arrived), "count" });
    }

    // Workload C: High-Density Stress Test 12x12 Grid (1200 vehicles, Coordinated + TrafficAware + Emergency)
    {
        Config cfg;
        cfg.scenario = TrafficScenario::Custom;
        cfg.seed = 42;
        cfg.vehicle_spawn_interval_s = 0.1f;
        cfg.max_vehicles = 1200;
        cfg.signal_controller_type = SignalControllerType::Coordinated;
        cfg.routing_strategy = RoutingStrategyType::TrafficAware;
        cfg.enable_emergency_priority = true;
        cfg.enable_emergency_rerouting = true;
        auto r = run_simulation_workload("W3_Stress_12x12_1200v", cfg,
                                         [] { return make_multilane_grid(12, 12, 120.0f, 2); },
                                         300, 1200, true);
        std::cout << "  " << std::left << std::setw(32) << r.name
                  << " | tick=" << std::right << std::setw(8) << std::fixed << std::setprecision(2) << r.mean_tick_us << " us"
                  << " (p95=" << std::setw(7) << r.p95_tick_us << ")"
                  << " | " << std::setw(9) << std::setprecision(0) << r.vehicles_per_sec << " veh/s"
                  << " | avgVeh=" << std::setprecision(1) << r.avg_active_vehicles
                  << " | RSS=" << std::setprecision(1) << r.rss_mb << " MB"
                  << " | arrived=" << r.total_arrived << "\n";
        csv_rows.push_back({ "simulation", r.name, "mean_tick_us", r.mean_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p95_tick_us", r.p95_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p99_tick_us", r.p99_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "vehicles_per_sec", r.vehicles_per_sec, "veh/s" });
        csv_rows.push_back({ "simulation", r.name, "ticks_per_sec", r.sim_ticks_per_sec, "ticks/s" });
        csv_rows.push_back({ "simulation", r.name, "rss_mb", r.rss_mb, "MB" });
        csv_rows.push_back({ "simulation", r.name, "total_arrived", static_cast<double>(r.total_arrived), "count" });
    }

    // Workload D: Multi-Intersection RL Control (6x6 Grid, 300 vehicles, live ONNX RLPolicyInference)
    {
        Config cfg;
        cfg.scenario = TrafficScenario::Custom;
        cfg.seed = 42;
        cfg.vehicle_spawn_interval_s = 0.1f;
        cfg.max_vehicles = 300;
        cfg.signal_controller_type = SignalControllerType::ReinforcementLearning;
        cfg.routing_strategy = RoutingStrategyType::Static;
        cfg.enable_rl_model = true;
        cfg.rl_model_path = resolve_model_path("models/traffic_signal_dqn.onnx");
        auto r = run_simulation_workload("W4_RL_ONNX_6x6_300v", cfg,
                                         [] { return make_multilane_grid(6, 6, 120.0f, 2); },
                                         400, 300, false);
        std::cout << "  " << std::left << std::setw(32) << r.name
                  << " | tick=" << std::right << std::setw(8) << std::fixed << std::setprecision(2) << r.mean_tick_us << " us"
                  << " (p95=" << std::setw(7) << r.p95_tick_us << ")"
                  << " | " << std::setw(9) << std::setprecision(0) << r.vehicles_per_sec << " veh/s"
                  << " | avgVeh=" << std::setprecision(1) << r.avg_active_vehicles
                  << " | RSS=" << std::setprecision(1) << r.rss_mb << " MB"
                  << " | arrived=" << r.total_arrived << "\n";
        csv_rows.push_back({ "simulation", r.name, "mean_tick_us", r.mean_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p95_tick_us", r.p95_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "p99_tick_us", r.p99_tick_us, "us" });
        csv_rows.push_back({ "simulation", r.name, "vehicles_per_sec", r.vehicles_per_sec, "veh/s" });
        csv_rows.push_back({ "simulation", r.name, "ticks_per_sec", r.sim_ticks_per_sec, "ticks/s" });
        csv_rows.push_back({ "simulation", r.name, "rss_mb", r.rss_mb, "MB" });
        csv_rows.push_back({ "simulation", r.name, "total_arrived", static_cast<double>(r.total_arrived), "count" });
    }

    run_subsystem_profile(csv_rows);
    run_pathfinding_benchmarks(csv_rows);
    run_ai_inference_benchmarks(csv_rows);
    run_ecs_evaluation_microbenchmark(csv_rows);

    const MemStats final_mem = get_memory_stats();
    std::cout << "\n[6] OVERALL PROCESS MEMORY USAGE\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Current RSS : " << std::fixed << std::setprecision(2) << final_mem.rss_mb << " MB\n";
    std::cout << "  Peak RSS    : " << std::fixed << std::setprecision(2) << final_mem.peak_rss_mb << " MB\n";
    csv_rows.push_back({ "memory", "process_overall", "rss_mb", final_mem.rss_mb, "MB" });
    csv_rows.push_back({ "memory", "process_overall", "peak_rss_mb", final_mem.peak_rss_mb, "MB" });

    if (!csv_path.empty()) {
        std::filesystem::path p(csv_path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream out(csv_path, std::ios::out | std::ios::trunc);
        if (out.is_open()) {
            out << "label,category,benchmark,metric,value,unit\n";
            out << std::fixed << std::setprecision(4);
            for (const auto& row : csv_rows) {
                out << label << "," << row.category << "," << row.benchmark << ","
                    << row.metric << "," << row.value << "," << row.unit << "\n";
            }
            std::cout << "\nExported benchmark metrics to: " << csv_path << "\n";
        }
    }

    std::cout << "================================================================================\n";
    return 0;
}

