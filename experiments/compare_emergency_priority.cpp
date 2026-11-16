//
// experiments/compare_emergency_priority.cpp
//
// Automated benchmark and comparative evaluation of Emergency Vehicle Priority:
// Evaluates 3 distinct operational strategies:
//   1. Baseline (No Priority): Emergency vehicles navigate normal traffic without signal preemption or detour routing.
//   2. Signal Preemption Only: Approaching emergency vehicles preempt traffic signals with green hold corridors.
//   3. Full Priority Corridor: Signal preemption + dynamic detour rerouting of background normal traffic.
//
// Measures:
//   - Emergency vehicle travel time, waiting time, and response speed.
//   - Disruption to normal traffic (normal trip duration, queue waiting times, fleet throughput).
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace syntraq;

namespace {

struct MetricStat {
    double mean{ 0.0 };
    double std_dev{ 0.0 };
    double min_val{ 0.0 };
    double max_val{ 0.0 };
};

MetricStat compute_stat(const std::vector<double>& values) {
    MetricStat stat{};
    if (values.empty()) return stat;

    const size_t n = values.size();
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stat.mean = sum / static_cast<double>(n);

    stat.min_val = *std::min_element(values.begin(), values.end());
    stat.max_val = *std::max_element(values.begin(), values.end());

    if (n > 1) {
        double sq_sum = 0.0;
        for (double v : values) {
            double diff = v - stat.mean;
            sq_sum += diff * diff;
        }
        stat.std_dev = std::sqrt(sq_sum / static_cast<double>(n - 1));
    }
    return stat;
}

RoadNetwork make_emergency_test_network() {
    // 3-intersection linear arterial (0 -> 1 -> 2) with a parallel bypass route (0 -> 3 -> 2)
    RoadNetwork net;
    const auto i0 = net.add_intersection({   0.0f,   0.0f }, "WestOrigin");
    const auto i1 = net.add_intersection({ 250.0f,   0.0f }, "CentralHub");
    const auto i2 = net.add_intersection({ 500.0f,   0.0f }, "EastHospital");
    const auto i3 = net.add_intersection({ 250.0f, 150.0f }, "SouthBypass");
    const auto i4 = net.add_intersection({ 250.0f, -150.0f }, "NorthInflow");

    // Main arterial corridor (multi-lane: 2 lanes)
    net.add_road(i0, i1, 250.0f, 13.89f, 2, "MainWest");
    net.add_road(i1, i2, 250.0f, 13.89f, 2, "MainEast");

    // Cross-traffic intersecting at CentralHub (i1)
    net.add_road(i4, i1, 150.0f, 13.89f, 1, "CrossNorth");
    net.add_road(i1, i3, 150.0f, 13.89f, 1, "CrossSouth");

    // Parallel bypass roads (alternative detour for normal vehicles)
    net.add_road(i0, i3, 260.0f, 13.89f, 1, "BypassIn");
    net.add_road(i3, i2, 260.0f, 13.89f, 1, "BypassOut");

    return net;
}

struct ModeResult {
    std::string mode_name;
    std::vector<double> emergency_travel_times;
    std::vector<double> emergency_wait_times;
    std::vector<double> emergency_speeds_kmh;
    std::vector<double> normal_travel_times;
    std::vector<double> normal_wait_times;
    std::vector<double> normal_speeds_kmh;
    std::vector<double> total_completed;
    std::vector<double> congestion_ratios;
};

ModeResult run_mode_trials(
    const std::string& mode_name,
    bool enable_preemption,
    bool enable_rerouting,
    const std::vector<uint32_t>& seeds,
    float duration_s = 60.0f)
{
    ModeResult mr;
    mr.mode_name = mode_name;

    for (uint32_t seed : seeds) {
        Config cfg;
        cfg.scenario                   = TrafficScenario::EmergencyPriority;
        cfg.sim_duration_s             = duration_s;
        cfg.dt_seconds                 = 0.1f;
        cfg.seed                       = seed;
        cfg.signal_controller_type     = SignalControllerType::Adaptive;
        cfg.routing_strategy           = enable_rerouting ? RoutingStrategyType::TrafficAware : RoutingStrategyType::Static;
        cfg.enable_emergency_priority  = enable_preemption;
        cfg.enable_emergency_rerouting = enable_rerouting;
        cfg.emergency_preemption_distance_m = 120.0f;

        Simulation sim{ cfg, make_emergency_test_network };
        sim.set_allowed_spawn_origins({ static_cast<IntersectionId>(0), static_cast<IntersectionId>(4) });
        sim.set_allowed_spawn_destinations({ static_cast<IntersectionId>(2), static_cast<IntersectionId>(3) });

        // Run simulation
        const uint32_t total_ticks = static_cast<uint32_t>(duration_s / cfg.dt_seconds);
        for (uint32_t tick = 0; tick < total_ticks; ++tick) {
            // Trigger emergency vehicle at tick 50 (5.0s) from WestOrigin to EastHospital
            if (tick == 50 || tick == 350) {
                sim.trigger_emergency_vehicle(static_cast<IntersectionId>(0), static_cast<IntersectionId>(2));
            }
            sim.tick();
        }

        const auto res = sim.get_run_result();

        if (res.emergency_completed > 0) {
            mr.emergency_travel_times.push_back(res.emergency_avg_travel_time_s);
            mr.emergency_wait_times.push_back(res.emergency_avg_waiting_time_s);
            mr.emergency_speeds_kmh.push_back(res.emergency_avg_speed_mps * 3.6);
        } else {
            // Did not arrive in time
            mr.emergency_travel_times.push_back(duration_s);
            mr.emergency_wait_times.push_back(duration_s * 0.5);
            mr.emergency_speeds_kmh.push_back(5.0);
        }

        mr.normal_travel_times.push_back(res.normal_avg_travel_time_s > 0.0f ? res.normal_avg_travel_time_s : res.avg_travel_time_s);
        mr.normal_wait_times.push_back(res.normal_avg_waiting_time_s > 0.0f ? res.normal_avg_waiting_time_s : res.avg_waiting_time_s);
        mr.normal_speeds_kmh.push_back(res.normal_avg_speed_mps > 0.0f ? res.normal_avg_speed_mps * 3.6 : res.avg_speed_kmh);
        mr.total_completed.push_back(res.total_completed);
        mr.congestion_ratios.push_back(res.congestion_ratio * 100.0);
    }

    return mr;
}

} // namespace

int main(int argc, char** argv) {
    std::string csv_out_path = "data/emergency_priority_summary.csv";
    float duration_s = 60.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--csv" || arg == "-o") && i + 1 < argc) {
            csv_out_path = argv[++i];
        } else if ((arg == "--duration" || arg == "-d") && i + 1 < argc) {
            duration_s = std::stof(argv[++i]);
        }
    }

    const std::vector<uint32_t> seeds = { 101, 202, 303, 404, 505 };

    std::cout << "\n========================================================================================\n"
              << "       SyntraQ Emergency Priority Corridor — Comparative Benchmark Evaluation          \n"
              << "========================================================================================\n"
              << " Evaluating 3 Operational Configurations across " << seeds.size() << " repeatable random seeds\n"
              << " Duration: " << duration_s << " s per trial | Dynamic Route Detours & Signal Preemption\n\n";

    std::cout << "[1/3] Benchmarking Baseline (No Priority)...\n";
    const auto res_baseline = run_mode_trials("Baseline (No Priority)", false, false, seeds, duration_s);

    std::cout << "[2/3] Benchmarking Signal Preemption Only...\n";
    const auto res_preemption = run_mode_trials("Signal Preemption Only", true, false, seeds, duration_s);

    std::cout << "[3/3] Benchmarking Full Priority Corridor (Preemption + Rerouting)...\n";
    const auto res_full = run_mode_trials("Full Priority Corridor", true, true, seeds, duration_s);

    const std::vector<ModeResult> all_results = { res_baseline, res_preemption, res_full };

    // Print Emergency Response Performance Table
    std::cout << "\n--- EMERGENCY RESPONSE PERFORMANCE METRICS ---\n";
    std::cout << std::left
              << std::setw(28) << "Strategy Mode"
              << std::setw(22) << "Emerg Travel Time (s)"
              << std::setw(22) << "Emerg Wait Time (s)"
              << std::setw(20) << "Emerg Speed (km/h)"
              << "\n";
    std::cout << std::string(92, '-') << "\n";

    for (const auto& mr : all_results) {
        const auto stat_travel = compute_stat(mr.emergency_travel_times);
        const auto stat_wait   = compute_stat(mr.emergency_wait_times);
        const auto stat_speed  = compute_stat(mr.emergency_speeds_kmh);

        std::ostringstream ss_trv, ss_wt, ss_spd;
        ss_trv << std::fixed << std::setprecision(1) << stat_travel.mean << " +/- " << stat_travel.std_dev;
        ss_wt  << std::fixed << std::setprecision(1) << stat_wait.mean   << " +/- " << stat_wait.std_dev;
        ss_spd << std::fixed << std::setprecision(1) << stat_speed.mean  << " +/- " << stat_speed.std_dev;

        std::cout << std::left
                  << std::setw(28) << mr.mode_name
                  << std::setw(22) << ss_trv.str()
                  << std::setw(22) << ss_wt.str()
                  << std::setw(20) << ss_spd.str()
                  << "\n";
    }

    // Print Disruption to Normal Traffic Table
    std::cout << "\n--- NORMAL TRAFFIC DISRUPTION & NETWORK IMPACT ---\n";
    std::cout << std::left
              << std::setw(28) << "Strategy Mode"
              << std::setw(22) << "Normal Travel Time (s)"
              << std::setw(20) << "Normal Wait (s)"
              << std::setw(16) << "Completed Veh"
              << std::setw(16) << "Congestion (%)"
              << "\n";
    std::cout << std::string(102, '-') << "\n";

    for (const auto& mr : all_results) {
        const auto stat_trv  = compute_stat(mr.normal_travel_times);
        const auto stat_wt   = compute_stat(mr.normal_wait_times);
        const auto stat_comp = compute_stat(mr.total_completed);
        const auto stat_cong = compute_stat(mr.congestion_ratios);

        std::ostringstream ss_trv, ss_wt, ss_comp, ss_cong;
        ss_trv  << std::fixed << std::setprecision(1) << stat_trv.mean  << " +/- " << stat_trv.std_dev;
        ss_wt   << std::fixed << std::setprecision(1) << stat_wt.mean   << " +/- " << stat_wt.std_dev;
        ss_comp << std::fixed << std::setprecision(0) << stat_comp.mean << " veh";
        ss_cong << std::fixed << std::setprecision(1) << stat_cong.mean << "%";

        std::cout << std::left
                  << std::setw(28) << mr.mode_name
                  << std::setw(22) << ss_trv.str()
                  << std::setw(20) << ss_wt.str()
                  << std::setw(16) << ss_comp.str()
                  << std::setw(16) << ss_cong.str()
                  << "\n";
    }

    // Export CSV Summary
    fs::path p(csv_out_path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream out(csv_out_path);
    if (out.is_open()) {
        out << "mode,emergency_travel_time_mean_s,emergency_travel_time_std_s,"
            << "emergency_wait_time_mean_s,emergency_wait_time_std_s,"
            << "emergency_speed_mean_kmh,emergency_speed_std_kmh,"
            << "normal_travel_time_mean_s,normal_travel_time_std_s,"
            << "normal_wait_time_mean_s,normal_wait_time_std_s,"
            << "total_completed_mean,congestion_ratio_mean\n";

        for (const auto& mr : all_results) {
            const auto st_trv  = compute_stat(mr.emergency_travel_times);
            const auto st_wt   = compute_stat(mr.emergency_wait_times);
            const auto st_spd  = compute_stat(mr.emergency_speeds_kmh);
            const auto sn_trv  = compute_stat(mr.normal_travel_times);
            const auto sn_wt   = compute_stat(mr.normal_wait_times);
            const auto sn_comp = compute_stat(mr.total_completed);
            const auto sn_cong = compute_stat(mr.congestion_ratios);

            out << std::fixed << std::setprecision(3)
                << mr.mode_name << ","
                << st_trv.mean << "," << st_trv.std_dev << ","
                << st_wt.mean << "," << st_wt.std_dev << ","
                << st_spd.mean << "," << st_spd.std_dev << ","
                << sn_trv.mean << "," << sn_trv.std_dev << ","
                << sn_wt.mean << "," << sn_wt.std_dev << ","
                << sn_comp.mean << "," << sn_cong.mean << "\n";
        }
        out.close();
        std::cout << "\n[SyntraQ] Summary metrics successfully exported to: " << csv_out_path << "\n\n";
    }

    return 0;
}

