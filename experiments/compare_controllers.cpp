//
// experiments/compare_controllers.cpp
//
// Automated experiment runner comparing Fixed-Time vs. Adaptive Traffic Signal Controllers.
// Runs the same scenarios (Low, Medium, High, Rush Hour) across multiple identical seeds
// and benchmarks:
//   - Waiting Time (s)
//   - Throughput (veh/h)
//   - Travel Time (s)
//   - Congestion Ratio (%)
//
// Exports raw per-run results to data/experiments/controller_comparison_runs.csv
// and aggregated statistical comparisons to data/experiments/controller_comparison_summary.csv.
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_metrics.hpp"
#include "syntraq/simulation/simulation.hpp"

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
    MetricStat congestion_pct;
};

AggregateMetrics aggregate(const std::vector<RunResult>& runs) {
    std::vector<double> speeds, waits, travels, max_qs, avg_qs, thr_vphs, cong_pcts;
    speeds.reserve(runs.size());
    waits.reserve(runs.size());
    travels.reserve(runs.size());
    max_qs.reserve(runs.size());
    avg_qs.reserve(runs.size());
    thr_vphs.reserve(runs.size());
    cong_pcts.reserve(runs.size());

    for (const auto& r : runs) {
        speeds.push_back(static_cast<double>(r.avg_speed_kmh));
        waits.push_back(static_cast<double>(r.avg_waiting_time_s));
        travels.push_back(static_cast<double>(r.avg_travel_time_s));
        max_qs.push_back(static_cast<double>(r.max_queue_length));
        avg_qs.push_back(static_cast<double>(r.avg_queue_length));
        thr_vphs.push_back(static_cast<double>(r.throughput_vph));
        cong_pcts.push_back(static_cast<double>(r.congestion_ratio * 100.0f));
    }

    AggregateMetrics m;
    m.speed_kmh      = compute_stat(speeds);
    m.wait_time_s    = compute_stat(waits);
    m.travel_time_s  = compute_stat(travels);
    m.max_queue      = compute_stat(max_qs);
    m.avg_queue      = compute_stat(avg_qs);
    m.throughput_vph = compute_stat(thr_vphs);
    m.congestion_pct = compute_stat(cong_pcts);
    return m;
}

} // namespace

int main(int argc, char* argv[]) {
    float sim_duration_s = 180.0f; // 3-minute evaluation per run
    std::vector<uint32_t> seeds = { 42u, 101u, 202u, 303u, 404u };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--duration" || arg == "-d") && i + 1 < argc) {
            sim_duration_s = std::stof(argv[++i]);
        }
    }

    const fs::path exp_dir   = "data/experiments";
    const fs::path runs_path = exp_dir / "controller_comparison_runs.csv";
    const fs::path summary_path = exp_dir / "controller_comparison_summary.csv";

    fs::create_directories(exp_dir);

    struct ScenarioSpec {
        TrafficScenario scenario;
        std::string     name;
    } scenarios[] = {
        { TrafficScenario::Low,      "Low Traffic" },
        { TrafficScenario::Medium,   "Medium Traffic" },
        { TrafficScenario::High,     "High Traffic" },
        { TrafficScenario::RushHour, "Rush Hour" }
    };

    struct ControllerSpec {
        SignalControllerType type;
        std::string          name;
    } controllers[] = {
        { SignalControllerType::FixedTime, "FixedTime" },
        { SignalControllerType::Adaptive,  "Adaptive"  }
    };

    std::cout << "\n======================================================================\n";
    std::cout << " [SyntraQ] Comparative Traffic Signal Controller Evaluation Harness\n";
    std::cout << " Comparing Fixed-Time vs. Adaptive across " << sizeof(scenarios)/sizeof(scenarios[0])
              << " scenarios (" << seeds.size() << " seeds each, " << sim_duration_s << "s/run)\n";
    std::cout << "======================================================================\n\n";

    std::ofstream runs_file(runs_path);
    if (!runs_file.is_open()) {
        std::cerr << "Error: Failed to open " << runs_path << "\n";
        return 1;
    }

    runs_file << "controller,scenario,seed,duration_s,total_spawned,total_completed,"
              << "active_remaining,avg_speed_kmh,avg_waiting_time_s,avg_travel_time_s,"
              << "max_queue_length,avg_queue_length,throughput_vph,congestion_ratio\n";

    struct ComparisonRecord {
        std::string      scenario_name;
        AggregateMetrics fixed_metrics;
        AggregateMetrics adaptive_metrics;
    };
    std::vector<ComparisonRecord> comparison_records;

    for (const auto& sc : scenarios) {
        std::cout << ">>> Scenario: " << sc.name << "\n";
        ComparisonRecord rec;
        rec.scenario_name = sc.name;

        for (const auto& ctrl : controllers) {
            std::vector<RunResult> runs;
            runs.reserve(seeds.size());

            std::cout << "  Controller: " << std::setw(10) << ctrl.name << " | ";

            for (uint32_t seed : seeds) {
                Config cfg;
                cfg.scenario               = sc.scenario;
                cfg.seed                   = seed;
                cfg.sim_duration_s         = sim_duration_s;
                cfg.headless               = true;
                cfg.signal_controller_type = ctrl.type;

                Simulation sim{ cfg };
                sim.run_for(sim_duration_s);

                RunResult res = sim.get_run_result();
                runs.push_back(res);

                runs_file << ctrl.name << ","
                          << sc.name << ","
                          << seed << ","
                          << sim_duration_s << ","
                          << res.total_spawned << ","
                          << res.total_completed << ","
                          << res.active_remaining << ","
                          << res.avg_speed_kmh << ","
                          << res.avg_waiting_time_s << ","
                          << res.avg_travel_time_s << ","
                          << res.max_queue_length << ","
                          << res.avg_queue_length << ","
                          << res.throughput_vph << ","
                          << res.congestion_ratio << "\n";
                runs_file.flush();
                std::cout << ".";
            }

            AggregateMetrics agg = aggregate(runs);
            if (ctrl.type == SignalControllerType::FixedTime) {
                rec.fixed_metrics = agg;
            } else {
                rec.adaptive_metrics = agg;
            }

            std::cout << " Done. (AvgWait: " << std::fixed << std::setprecision(1) << agg.wait_time_s.mean
                      << "s, Throughput: " << std::setprecision(0) << agg.throughput_vph.mean << " vph)\n";
        }

        comparison_records.push_back(rec);
        std::cout << "\n";
    }
    runs_file.close();

    // ── Export Summary Statistics & Relative Improvement CSV ───────────────────
    std::ofstream summary_file(summary_path);
    if (!summary_file.is_open()) {
        std::cerr << "Error: Failed to open " << summary_path << "\n";
        return 1;
    }

    summary_file << "scenario,metric,fixed_mean,fixed_std,adaptive_mean,adaptive_std,improvement_pct\n";

    auto write_row = [&](const std::string& sc_name,
                         const std::string& metric_name,
                         const MetricStat&  fixed,
                         const MetricStat&  adapt,
                         bool               higher_is_better) {
        double imp = 0.0;
        if (higher_is_better) {
            imp = (fixed.mean > 1e-4) ? ((adapt.mean - fixed.mean) / fixed.mean * 100.0) : 0.0;
        } else {
            imp = (fixed.mean > 1e-4) ? ((fixed.mean - adapt.mean) / fixed.mean * 100.0) : 0.0;
        }

        summary_file << std::fixed << std::setprecision(3)
                     << sc_name << ","
                     << metric_name << ","
                     << fixed.mean << ","
                     << fixed.std_dev << ","
                     << adapt.mean << ","
                     << adapt.std_dev << ","
                     << imp << "\n";
    };

    std::cout << "==========================================================================================\n";
    std::cout << " PERFORMANCE COMPARISON: FIXED-TIME vs. ADAPTIVE CONTROLLER\n";
    std::cout << "==========================================================================================\n";
    std::cout << std::left << std::setw(15) << "Scenario"
              << std::setw(18) << "Metric"
              << std::setw(16) << "Fixed-Time"
              << std::setw(16) << "Adaptive"
              << std::setw(16) << "Change (%)"
              << "\n------------------------------------------------------------------------------------------\n";

    for (const auto& rec : comparison_records) {
        auto print_and_write = [&](const std::string& metric,
                                   const MetricStat&  f,
                                   const MetricStat&  a,
                                   bool               higher_is_better,
                                   const std::string& unit) {
            write_row(rec.scenario_name, metric, f, a, higher_is_better);

            double imp = higher_is_better
                ? ((f.mean > 1e-4) ? ((a.mean - f.mean) / f.mean * 100.0) : 0.0)
                : ((f.mean > 1e-4) ? ((f.mean - a.mean) / f.mean * 100.0) : 0.0);

            std::ostringstream f_str, a_str, imp_str;
            f_str << std::fixed << std::setprecision(1) << f.mean << " " << unit;
            a_str << std::fixed << std::setprecision(1) << a.mean << " " << unit;

            if (higher_is_better) {
                imp_str << (imp >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << imp << "%";
            } else {
                imp_str << (imp >= 0 ? "-" : "+") << std::fixed << std::setprecision(1) << std::abs(imp) << "% (better)";
            }

            std::cout << std::left << std::setw(15) << rec.scenario_name
                      << std::setw(18) << metric
                      << std::setw(16) << f_str.str()
                      << std::setw(16) << a_str.str()
                      << std::setw(16) << imp_str.str()
                      << "\n";
        };

        print_and_write("Waiting Time", rec.fixed_metrics.wait_time_s,    rec.adaptive_metrics.wait_time_s,    false, "s");
        print_and_write("Throughput",   rec.fixed_metrics.throughput_vph, rec.adaptive_metrics.throughput_vph, true,  "vph");
        print_and_write("Travel Time",  rec.fixed_metrics.travel_time_s,  rec.adaptive_metrics.travel_time_s,  false, "s");
        print_and_write("Congestion",   rec.fixed_metrics.congestion_pct, rec.adaptive_metrics.congestion_pct, false, "%");
        std::cout << "------------------------------------------------------------------------------------------\n";
    }

    summary_file.close();
    std::cout << "\n[SyntraQ] Raw runs exported to: " << runs_path << "\n";
    std::cout << "[SyntraQ] Summary stats exported to: " << summary_path << "\n";

    return 0;
}
