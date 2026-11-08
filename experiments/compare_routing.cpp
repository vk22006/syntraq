//
// experiments/compare_routing.cpp
//
// Automated experiment runner comparing Static Routing (Distance-only)
// vs. Traffic-Aware Routing (Distance + Congestion Penalty + Predicted Delay).
// Evaluates route selection divergence, trip completion, waiting time,
// travel time, and network congestion across multiple random seeds.
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/routing/static_routing.hpp"
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

struct AggregateMetrics {
    MetricStat speed_kmh;
    MetricStat wait_time_s;
    MetricStat travel_time_s;
    MetricStat max_queue;
    MetricStat avg_queue;
    MetricStat throughput_vph;
    MetricStat congestion_ratio;
    MetricStat completed_trips;
};

AggregateMetrics aggregate(const std::vector<RunResult>& runs) {
    std::vector<double> speed, wait, travel, max_q, avg_q, thr, cong, completed;
    for (const auto& r : runs) {
        speed.push_back(r.avg_speed_kmh);
        wait.push_back(r.avg_waiting_time_s);
        travel.push_back(r.avg_travel_time_s);
        max_q.push_back(static_cast<double>(r.max_queue_length));
        avg_q.push_back(r.avg_queue_length);
        thr.push_back(r.throughput_vph);
        cong.push_back(r.congestion_ratio * 100.0);
        completed.push_back(static_cast<double>(r.total_completed));
    }

    AggregateMetrics agg;
    agg.speed_kmh        = compute_stat(speed);
    agg.wait_time_s      = compute_stat(wait);
    agg.travel_time_s    = compute_stat(travel);
    agg.max_queue        = compute_stat(max_q);
    agg.avg_queue        = compute_stat(avg_q);
    agg.throughput_vph   = compute_stat(thr);
    agg.congestion_ratio = compute_stat(cong);
    agg.completed_trips  = compute_stat(completed);
    return agg;
}

/// Bifurcated arterial network with competing paths:
/// Path 1 (Direct Arterial through B):
///   A -> B (120m) -> D (120m) = 240m total distance
/// Path 2 (Bypass Arterial through C):
///   A -> C (160m) -> D (160m) = 320m total distance
RoadNetwork make_bifurcated_experiment_network(IntersectionId& out_a, IntersectionId& out_d) {
    RoadNetwork net;
    out_a = net.add_intersection({   0.0f,   0.0f }, "Origin_A");
    auto b = net.add_intersection({ 120.0f,   0.0f }, "Direct_B");
    auto c = net.add_intersection({ 140.0f, 120.0f }, "Bypass_C");
    out_d = net.add_intersection({ 260.0f,   0.0f }, "Dest_D");

    // Direct path
    net.add_road(out_a, b, 120.0f);
    net.add_road(b, out_d, 120.0f);

    // Bypass path
    net.add_road(out_a, c, 160.0f);
    net.add_road(c, out_d, 160.0f);

    return net;
}

} // namespace

int main(int argc, char** argv) {
    float sim_duration = 60.0f;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" && i + 1 < argc) {
            sim_duration = std::stof(argv[++i]);
        }
    }

    std::cout << "========================================================================\n";
    std::cout << "       SyntraQ Automated Routing Strategy Comparative Benchmark         \n";
    std::cout << "            Static (Distance-Only) vs. Traffic-Aware Routing            \n";
    std::cout << "========================================================================\n";
    std::cout << "Duration per run: " << sim_duration << " s\n\n";

    fs::create_directories("data/experiments");

    const std::vector<uint32_t> seeds = { 42, 101, 202, 303, 404 };
    const std::vector<TrafficScenario> scenarios = {
        TrafficScenario::Medium,
        TrafficScenario::High,
        TrafficScenario::RushHour
    };

    std::vector<RunResult> static_runs;
    std::vector<RunResult> aware_runs;

    for (auto sc : scenarios) {
        auto preset = get_scenario_preset(sc);
        std::cout << "------------------------------------------------------------------------\n";
        std::cout << "Evaluating Scenario: " << preset.name << " (" << seeds.size() << " seeds)\n";
        std::cout << "------------------------------------------------------------------------\n";

        std::vector<RunResult> sc_static_runs;
        std::vector<RunResult> sc_aware_runs;

        for (uint32_t seed : seeds) {
            // 1. Static Routing
            Config cfg_static;
            cfg_static.scenario         = sc;
            cfg_static.seed             = seed;
            cfg_static.sim_duration_s   = sim_duration;
            cfg_static.routing_strategy = RoutingStrategyType::Static;

            IntersectionId origin_a{}, dest_d{};
            Simulation sim_static(cfg_static, [&] {
                return make_bifurcated_experiment_network(origin_a, dest_d);
            });
            sim_static.set_allowed_spawn_origins({ origin_a });
            sim_static.set_allowed_spawn_destinations({ dest_d });
            sim_static.run_for(sim_duration);

            auto r_static = sim_static.get_run_result();
            sc_static_runs.push_back(r_static);
            static_runs.push_back(r_static);

            // 2. Traffic-Aware Routing
            Config cfg_aware = cfg_static;
            cfg_aware.routing_strategy = RoutingStrategyType::TrafficAware;

            Simulation sim_aware(cfg_aware, [&] {
                return make_bifurcated_experiment_network(origin_a, dest_d);
            });
            sim_aware.set_allowed_spawn_origins({ origin_a });
            sim_aware.set_allowed_spawn_destinations({ dest_d });
            sim_aware.run_for(sim_duration);

            auto r_aware = sim_aware.get_run_result();
            sc_aware_runs.push_back(r_aware);
            aware_runs.push_back(r_aware);
        }

        auto agg_s = aggregate(sc_static_runs);
        auto agg_a = aggregate(sc_aware_runs);

        auto diff_pct = [](double baseline, double current, bool lower_is_better) -> std::string {
            if (std::abs(baseline) < 1e-6) return "0.0%";
            double pct = ((current - baseline) / baseline) * 100.0;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1);
            if (pct > 0) ss << "+";
            ss << pct << "%";
            bool positive = lower_is_better ? (pct < 0) : (pct > 0);
            if (positive) ss << " [improved]";
            return ss.str();
        };

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Metric                         Static Routing         Traffic-Aware          Delta (%)\n";
        std::cout << "----------------------------------------------------------------------------------------\n";
        std::cout << "Avg Travel Time (s)         "
                  << std::setw(10) << agg_s.travel_time_s.mean << " +- " << std::setw(5) << agg_s.travel_time_s.std_dev << "   "
                  << std::setw(10) << agg_a.travel_time_s.mean << " +- " << std::setw(5) << agg_a.travel_time_s.std_dev << "   "
                  << diff_pct(agg_s.travel_time_s.mean, agg_a.travel_time_s.mean, true) << "\n";

        std::cout << "Avg Waiting Time (s)        "
                  << std::setw(10) << agg_s.wait_time_s.mean << " +- " << std::setw(5) << agg_s.wait_time_s.std_dev << "   "
                  << std::setw(10) << agg_a.wait_time_s.mean << " +- " << std::setw(5) << agg_a.wait_time_s.std_dev << "   "
                  << diff_pct(agg_s.wait_time_s.mean, agg_a.wait_time_s.mean, true) << "\n";

        std::cout << "Avg Queue Depth (veh)       "
                  << std::setw(10) << agg_s.avg_queue.mean << " +- " << std::setw(5) << agg_s.avg_queue.std_dev << "   "
                  << std::setw(10) << agg_a.avg_queue.mean << " +- " << std::setw(5) << agg_a.avg_queue.std_dev << "   "
                  << diff_pct(agg_s.avg_queue.mean, agg_a.avg_queue.mean, true) << "\n";

        std::cout << "Peak Queue Length (veh)     "
                  << std::setw(10) << agg_s.max_queue.mean << " +- " << std::setw(5) << agg_s.max_queue.std_dev << "   "
                  << std::setw(10) << agg_a.max_queue.mean << " +- " << std::setw(5) << agg_a.max_queue.std_dev << "   "
                  << diff_pct(agg_s.max_queue.mean, agg_a.max_queue.mean, true) << "\n";

        std::cout << "Completed Trips (veh)       "
                  << std::setw(10) << agg_s.completed_trips.mean << " +- " << std::setw(5) << agg_s.completed_trips.std_dev << "   "
                  << std::setw(10) << agg_a.completed_trips.mean << " +- " << std::setw(5) << agg_a.completed_trips.std_dev << "   "
                  << diff_pct(agg_s.completed_trips.mean, agg_a.completed_trips.mean, false) << "\n";

        std::cout << "Congestion Ratio (%)        "
                  << std::setw(10) << agg_s.congestion_ratio.mean << " +- " << std::setw(5) << agg_s.congestion_ratio.std_dev << "   "
                  << std::setw(10) << agg_a.congestion_ratio.mean << " +- " << std::setw(5) << agg_a.congestion_ratio.std_dev << "   "
                  << diff_pct(agg_s.congestion_ratio.mean, agg_a.congestion_ratio.mean, true) << "\n\n";
    }

    // Export detailed runs CSV
    std::ofstream csv_out("data/experiments/routing_comparison_runs.csv");
    if (csv_out.is_open()) {
        csv_out << "strategy," << RunResult::csv_header() << "\n";
        for (const auto& r : static_runs) {
            csv_out << "StaticRouting," << r.to_csv_row() << "\n";
        }
        for (const auto& r : aware_runs) {
            csv_out << "TrafficAwareRouting," << r.to_csv_row() << "\n";
        }
        std::cout << "[SyntraQ] Detailed runs CSV saved to: data/experiments/routing_comparison_runs.csv\n";
    }

    // Export JSON summary
    std::ofstream json_out("data/experiments/routing_comparison.json");
    if (json_out.is_open()) {
        auto agg_all_s = aggregate(static_runs);
        auto agg_all_a = aggregate(aware_runs);
        json_out << "{\n";
        json_out << "  \"duration_s\": " << sim_duration << ",\n";
        json_out << "  \"num_runs\": " << static_runs.size() << ",\n";
        json_out << "  \"static_routing\": {\n";
        json_out << "    \"travel_time_s\": " << agg_all_s.travel_time_s.mean << ",\n";
        json_out << "    \"wait_time_s\": " << agg_all_s.wait_time_s.mean << ",\n";
        json_out << "    \"avg_queue\": " << agg_all_s.avg_queue.mean << ",\n";
        json_out << "    \"max_queue\": " << agg_all_s.max_queue.mean << ",\n";
        json_out << "    \"completed_trips\": " << agg_all_s.completed_trips.mean << ",\n";
        json_out << "    \"congestion_ratio\": " << agg_all_s.congestion_ratio.mean << "\n";
        json_out << "  },\n";
        json_out << "  \"traffic_aware_routing\": {\n";
        json_out << "    \"travel_time_s\": " << agg_all_a.travel_time_s.mean << ",\n";
        json_out << "    \"wait_time_s\": " << agg_all_a.wait_time_s.mean << ",\n";
        json_out << "    \"avg_queue\": " << agg_all_a.avg_queue.mean << ",\n";
        json_out << "    \"max_queue\": " << agg_all_a.max_queue.mean << ",\n";
        json_out << "    \"completed_trips\": " << agg_all_a.completed_trips.mean << ",\n";
        json_out << "    \"congestion_ratio\": " << agg_all_a.congestion_ratio.mean << "\n";
        json_out << "  }\n";
        json_out << "}\n";
        std::cout << "[SyntraQ] Benchmark JSON metrics saved to: data/experiments/routing_comparison.json\n";
    }

    return 0;
}
