//
// experiments/compare_network_coordination.cpp
//
// Multi-intersection network coordination benchmark.
// Compares:
//   1. Fixed-Time (Uncoordinated baseline)
//   2. Independent Adaptive (Actuated local queues, uncoordinated)
//   3. Coordinated Control (Upstream Green-Wave Progression + Downstream Spillback Gating)
//
// Evaluates performance on a multi-intersection arterial corridor across multiple random seeds
// and traffic demand levels (Medium, High, Rush Hour).
// Measures:
//   - Queue propagation (peak queue depth)
//   - Average waiting time
//   - Average travel time
//   - Throughput
//   - Congestion ratio
//
// Exports raw per-run results to data/experiments/network_coordination_runs.csv
// and statistical summaries to data/experiments/network_coordination_summary.csv.
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/signals/network_coordination_manager.hpp"
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
    MetricStat wait_time_s;
    MetricStat travel_time_s;
    MetricStat max_queue;
    MetricStat avg_queue;
    MetricStat throughput_vph;
    MetricStat congestion_ratio;
    MetricStat completed_trips;
};

AggregateMetrics aggregate(const std::vector<RunResult>& runs) {
    std::vector<double> waits, travels, max_qs, avg_qs, thr_vphs, cong_ratios, completed;
    waits.reserve(runs.size());
    travels.reserve(runs.size());
    max_qs.reserve(runs.size());
    avg_qs.reserve(runs.size());
    thr_vphs.reserve(runs.size());
    cong_ratios.reserve(runs.size());
    completed.reserve(runs.size());

    for (const auto& r : runs) {
        waits.push_back(static_cast<double>(r.avg_waiting_time_s));
        travels.push_back(static_cast<double>(r.avg_travel_time_s));
        max_qs.push_back(static_cast<double>(r.max_queue_length));
        avg_qs.push_back(static_cast<double>(r.avg_queue_length));
        thr_vphs.push_back(static_cast<double>(r.throughput_vph));
        cong_ratios.push_back(static_cast<double>(r.congestion_ratio * 100.0f));
        completed.push_back(static_cast<double>(r.total_completed));
    }

    AggregateMetrics m;
    m.wait_time_s      = compute_stat(waits);
    m.travel_time_s    = compute_stat(travels);
    m.max_queue        = compute_stat(max_qs);
    m.avg_queue        = compute_stat(avg_qs);
    m.throughput_vph   = compute_stat(thr_vphs);
    m.congestion_ratio = compute_stat(cong_ratios);
    m.completed_trips  = compute_stat(completed);
    return m;
}

// 3-Intersection Arterial Corridor Generator
// 3 controlled signal junctions spaced 250m apart along a main avenue,
// crossed by 3 secondary streets.
RoadNetwork make_corridor_network(IntersectionId& west_in, IntersectionId& east_out) {
    RoadNetwork net;

    // Signalized corridor intersections
    auto i0 = net.add_intersection({ 150.0f, 0.0f });
    auto i1 = net.add_intersection({ 400.0f, 0.0f });
    auto i2 = net.add_intersection({ 650.0f, 0.0f });

    // Outer boundary endpoints
    west_in  = net.add_intersection({   0.0f, 0.0f });
    east_out = net.add_intersection({ 800.0f, 0.0f });

    // Cross-street boundary endpoints
    auto n0 = net.add_intersection({ 150.0f,  150.0f });
    auto s0 = net.add_intersection({ 150.0f, -150.0f });
    auto n1 = net.add_intersection({ 400.0f,  150.0f });
    auto s1 = net.add_intersection({ 400.0f, -150.0f });
    auto n2 = net.add_intersection({ 650.0f,  150.0f });
    auto s2 = net.add_intersection({ 650.0f, -150.0f });

    // Main arterial corridor (2 lanes each direction, 50 km/h)
    net.add_road(west_in, i0, 13.89f, 2);
    net.add_road(i0, i1, 13.89f, 2);
    net.add_road(i1, i2, 13.89f, 2);
    net.add_road(i2, east_out, 13.89f, 2);

    net.add_road(i1, i0, 13.89f, 2);
    net.add_road(i2, i1, 13.89f, 2);

    // Cross streets (1 lane, 40 km/h)
    net.add_road(n0, i0, 11.11f, 1);
    net.add_road(s0, i0, 11.11f, 1);
    net.add_road(i0, n0, 11.11f, 1);
    net.add_road(i0, s0, 11.11f, 1);

    net.add_road(n1, i1, 11.11f, 1);
    net.add_road(s1, i1, 11.11f, 1);
    net.add_road(i1, n1, 11.11f, 1);
    net.add_road(i1, s1, 11.11f, 1);

    net.add_road(n2, i2, 11.11f, 1);
    net.add_road(s2, i2, 11.11f, 1);
    net.add_road(i2, n2, 11.11f, 1);
    net.add_road(i2, s2, 11.11f, 1);

    return net;
}

} // namespace

int main(int argc, char* argv[]) {
    float sim_duration = 60.0f;
    if (argc > 1) {
        sim_duration = std::stof(argv[1]);
    }

    std::cout << "========================================================================\n";
    std::cout << "     NETWORK-LEVEL COORDINATED TRAFFIC SIGNAL CONTROL BENCHMARK         \n";
    std::cout << "========================================================================\n";
    std::cout << "Comparing:\n";
    std::cout << "  1. Fixed-Time (Uncoordinated baseline)\n";
    std::cout << "  2. Adaptive (Independent local queue actuation)\n";
    std::cout << "  3. Coordinated Control (Green-Wave Progression + Spillback Gating)\n";
    std::cout << "Simulation duration per run: " << sim_duration << " s\n\n";

    fs::create_directories("data/experiments");

    const std::vector<TrafficScenario> scenarios = {
        TrafficScenario::Medium,
        TrafficScenario::High,
        TrafficScenario::RushHour
    };

    const std::vector<uint32_t> seeds = { 42, 101, 202, 303, 404 };

    struct StrategyResult {
        std::string strategy_name;
        SignalControllerType type;
        std::vector<RunResult> runs;
    };

    std::vector<RunResult> all_fixed_runs;
    std::vector<RunResult> all_adapt_runs;
    std::vector<RunResult> all_coord_runs;

    std::ofstream summary_csv("data/experiments/network_coordination_summary.csv");
    summary_csv << "scenario,strategy,wait_time_mean,wait_time_std,travel_time_mean,travel_time_std,"
                << "max_queue_mean,max_queue_std,avg_queue_mean,avg_queue_std,"
                << "throughput_vph_mean,throughput_vph_std,congestion_pct_mean,congestion_pct_std,"
                << "completed_mean,completed_std\n";

    std::ofstream runs_csv("data/experiments/network_coordination_runs.csv");
    runs_csv << "strategy," << RunResult::csv_header() << "\n";

    for (auto sc : scenarios) {
        auto preset = get_scenario_preset(sc);
        std::cout << "------------------------------------------------------------------------\n";
        std::cout << "Evaluating Scenario: " << preset.name << " (" << seeds.size() << " seeds)\n";
        std::cout << "------------------------------------------------------------------------\n";

        std::vector<RunResult> sc_fixed_runs;
        std::vector<RunResult> sc_adapt_runs;
        std::vector<RunResult> sc_coord_runs;

        for (uint32_t seed : seeds) {
            // Helper lambda to run a controller mode on the corridor
            auto run_corridor = [&](SignalControllerType ctrl_type) -> RunResult {
                Config cfg;
                cfg.scenario               = sc;
                cfg.seed                   = seed;
                cfg.sim_duration_s         = sim_duration;
                cfg.dt_seconds             = 0.1f;
                cfg.signal_controller_type = ctrl_type;
                cfg.default_green_duration_s = 10.0f;

                IntersectionId west_in{}, east_out{};
                Simulation sim(cfg, [&] {
                    return make_corridor_network(west_in, east_out);
                });
                sim.run_for(sim_duration);
                return sim.metrics().get_run_result();
            };

            auto r_fixed = run_corridor(SignalControllerType::FixedTime);
            sc_fixed_runs.push_back(r_fixed);
            all_fixed_runs.push_back(r_fixed);
            runs_csv << "FixedTime," << r_fixed.to_csv_row() << "\n";

            auto r_adapt = run_corridor(SignalControllerType::Adaptive);
            sc_adapt_runs.push_back(r_adapt);
            all_adapt_runs.push_back(r_adapt);
            runs_csv << "Adaptive," << r_adapt.to_csv_row() << "\n";

            auto r_coord = run_corridor(SignalControllerType::Coordinated);
            sc_coord_runs.push_back(r_coord);
            all_coord_runs.push_back(r_coord);
            runs_csv << "Coordinated," << r_coord.to_csv_row() << "\n";
        }

        auto agg_f = aggregate(sc_fixed_runs);
        auto agg_a = aggregate(sc_adapt_runs);
        auto agg_c = aggregate(sc_coord_runs);

        auto write_summary_row = [&](const std::string& strat_name, const AggregateMetrics& m) {
            summary_csv << preset.name << "," << strat_name << ","
                        << m.wait_time_s.mean << "," << m.wait_time_s.std_dev << ","
                        << m.travel_time_s.mean << "," << m.travel_time_s.std_dev << ","
                        << m.max_queue.mean << "," << m.max_queue.std_dev << ","
                        << m.avg_queue.mean << "," << m.avg_queue.std_dev << ","
                        << m.throughput_vph.mean << "," << m.throughput_vph.std_dev << ","
                        << m.congestion_ratio.mean << "," << m.congestion_ratio.std_dev << ","
                        << m.completed_trips.mean << "," << m.completed_trips.std_dev << "\n";
        };

        write_summary_row("FixedTime", agg_f);
        write_summary_row("Adaptive", agg_a);
        write_summary_row("Coordinated", agg_c);

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
        std::cout << "Metric                    Fixed-Time         Adaptive           Coordinated        Coord vs Fixed\n";
        std::cout << "------------------------------------------------------------------------------------------------\n";
        std::cout << "Avg Waiting Time (s)   "
                  << std::setw(8) << agg_f.wait_time_s.mean << " +- " << std::setw(4) << agg_f.wait_time_s.std_dev << "   "
                  << std::setw(8) << agg_a.wait_time_s.mean << " +- " << std::setw(4) << agg_a.wait_time_s.std_dev << "   "
                  << std::setw(8) << agg_c.wait_time_s.mean << " +- " << std::setw(4) << agg_c.wait_time_s.std_dev << "   "
                  << diff_pct(agg_f.wait_time_s.mean, agg_c.wait_time_s.mean, true) << "\n";

        std::cout << "Avg Travel Time (s)    "
                  << std::setw(8) << agg_f.travel_time_s.mean << " +- " << std::setw(4) << agg_f.travel_time_s.std_dev << "   "
                  << std::setw(8) << agg_a.travel_time_s.mean << " +- " << std::setw(4) << agg_a.travel_time_s.std_dev << "   "
                  << std::setw(8) << agg_c.travel_time_s.mean << " +- " << std::setw(4) << agg_c.travel_time_s.std_dev << "   "
                  << diff_pct(agg_f.travel_time_s.mean, agg_c.travel_time_s.mean, true) << "\n";

        std::cout << "Peak Queue Length (veh)"
                  << std::setw(8) << agg_f.max_queue.mean << " +- " << std::setw(4) << agg_f.max_queue.std_dev << "   "
                  << std::setw(8) << agg_a.max_queue.mean << " +- " << std::setw(4) << agg_a.max_queue.std_dev << "   "
                  << std::setw(8) << agg_c.max_queue.mean << " +- " << std::setw(4) << agg_c.max_queue.std_dev << "   "
                  << diff_pct(agg_f.max_queue.mean, agg_c.max_queue.mean, true) << "\n";

        std::cout << "Avg Queue Depth (veh)  "
                  << std::setw(8) << agg_f.avg_queue.mean << " +- " << std::setw(4) << agg_f.avg_queue.std_dev << "   "
                  << std::setw(8) << agg_a.avg_queue.mean << " +- " << std::setw(4) << agg_a.avg_queue.std_dev << "   "
                  << std::setw(8) << agg_c.avg_queue.mean << " +- " << std::setw(4) << agg_c.avg_queue.std_dev << "   "
                  << diff_pct(agg_f.avg_queue.mean, agg_c.avg_queue.mean, true) << "\n";

        std::cout << "Throughput (veh/h)     "
                  << std::setw(8) << agg_f.throughput_vph.mean << " +- " << std::setw(4) << agg_f.throughput_vph.std_dev << "   "
                  << std::setw(8) << agg_a.throughput_vph.mean << " +- " << std::setw(4) << agg_a.throughput_vph.std_dev << "   "
                  << std::setw(8) << agg_c.throughput_vph.mean << " +- " << std::setw(4) << agg_c.throughput_vph.std_dev << "   "
                  << diff_pct(agg_f.throughput_vph.mean, agg_c.throughput_vph.mean, false) << "\n";

        std::cout << "Congestion Ratio (%)   "
                  << std::setw(8) << agg_f.congestion_ratio.mean << " +- " << std::setw(4) << agg_f.congestion_ratio.std_dev << "   "
                  << std::setw(8) << agg_a.congestion_ratio.mean << " +- " << std::setw(4) << agg_a.congestion_ratio.std_dev << "   "
                  << std::setw(8) << agg_c.congestion_ratio.mean << " +- " << std::setw(4) << agg_c.congestion_ratio.std_dev << "   "
                  << diff_pct(agg_f.congestion_ratio.mean, agg_c.congestion_ratio.mean, true) << "\n\n";
    }

    std::cout << "Saved detailed runs to data/experiments/network_coordination_runs.csv\n";
    std::cout << "Saved summary statistics to data/experiments/network_coordination_summary.csv\n";
    return 0;
}
