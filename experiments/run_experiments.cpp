//
// experiments/run_experiments.cpp
//
// Automated experiment runner for SyntraQ baseline evaluations.
// Executes the baseline fixed-time traffic signal controller across multiple
// traffic scenarios (Low, Medium, High, Rush Hour) using multiple random seeds.
// Records per-run results to CSV and calculates mean, standard deviation,
// minimum, and maximum statistics across runs.
//

#include "syntraq/core/config.h"
#include "syntraq/metrics/traffic_metrics.h"
#include "syntraq/simulation/simulation.h"

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

struct ScenarioStats {
    std::string scenario_name;
    size_t      sample_size{ 0 };

    struct MetricStat {
        double mean{ 0.0 };
        double std_dev{ 0.0 };
        double min_val{ 0.0 };
        double max_val{ 0.0 };
    };

    MetricStat speed_kmh;
    MetricStat wait_time_s;
    MetricStat travel_time_s;
    MetricStat max_queue;
    MetricStat avg_queue;
    MetricStat throughput_vph;
    MetricStat congestion_pct;
};

ScenarioStats::MetricStat compute_metric_stat(const std::vector<double>& values) {
    ScenarioStats::MetricStat stat{};
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
    } else {
        stat.std_dev = 0.0;
    }
    return stat;
}

ScenarioStats aggregate_results(const std::string& name, const std::vector<RunResult>& runs) {
    ScenarioStats stats;
    stats.scenario_name = name;
    stats.sample_size   = runs.size();

    std::vector<double> speeds;
    std::vector<double> waits;
    std::vector<double> travels;
    std::vector<double> max_qs;
    std::vector<double> avg_qs;
    std::vector<double> thr_vphs;
    std::vector<double> cong_pcts;

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

    stats.speed_kmh      = compute_metric_stat(speeds);
    stats.wait_time_s    = compute_metric_stat(waits);
    stats.travel_time_s  = compute_metric_stat(travels);
    stats.max_queue      = compute_metric_stat(max_qs);
    stats.avg_queue      = compute_metric_stat(avg_qs);
    stats.throughput_vph = compute_metric_stat(thr_vphs);
    stats.congestion_pct = compute_metric_stat(cong_pcts);

    return stats;
}

std::vector<uint32_t> parse_seeds(const std::string& str) {
    std::vector<uint32_t> seeds;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            seeds.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
    }
    return seeds;
}

} // namespace

int main(int argc, char* argv[]) {
    float                 sim_duration_s = 180.0f;
    std::vector<uint32_t> seeds          = { 42, 101, 2024, 777, 999, 1234, 5678, 9012, 31415, 65432 };
    std::string           output_dir     = "data/experiments";
    std::string           runs_filename  = "baseline_runs.csv";
    std::string           stats_filename = "baseline_summary_stats.csv";

    // ── Parse CLI arguments ──────────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--duration" || arg == "-d") && i + 1 < argc) {
            sim_duration_s = std::stof(argv[++i]);
        } else if (arg == "--seeds" && i + 1 < argc) {
            seeds = parse_seeds(argv[++i]);
        } else if (arg == "--num-seeds" && i + 1 < argc) {
            int n = std::stoi(argv[++i]);
            seeds.clear();
            for (int s = 0; s < n; ++s) {
                seeds.push_back(static_cast<uint32_t>(42 + s * 107));
            }
        } else if ((arg == "--out-dir" || arg == "-o") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "SyntraQ Baseline Experiment Runner\n"
                      << "Usage: syntraq_experiments [options]\n"
                      << "Options:\n"
                      << "  -d, --duration <sec>   Duration per simulation run (default: 180.0)\n"
                      << "  --seeds <s1,s2,...>    Comma-separated list of random seeds\n"
                      << "  --num-seeds <N>        Auto-generate N deterministic seeds\n"
                      << "  -o, --out-dir <path>   Output directory for CSV files (default: data/experiments)\n"
                      << "  -h, --help             Show this help message\n";
            return 0;
        }
    }

    fs::create_directories(output_dir);
    const fs::path runs_path  = fs::path(output_dir) / runs_filename;
    const fs::path stats_path = fs::path(output_dir) / stats_filename;

    std::cout << "===============================================================\n"
              << "       SyntraQ Baseline Traffic Experiment Runner              \n"
              << "===============================================================\n"
              << " Simulation Duration: " << sim_duration_s << " s per run\n"
              << " Seeds Count        : " << seeds.size() << " seeds per scenario\n"
              << " Output Directory   : " << output_dir << "\n"
              << "===============================================================\n\n";

    const struct ScenarioSpec {
        TrafficScenario scenario;
        std::string     name;
    } scenarios[] = {
        { TrafficScenario::Low,      "Low Traffic" },
        { TrafficScenario::Medium,   "Medium Traffic" },
        { TrafficScenario::High,     "High Traffic" },
        { TrafficScenario::RushHour, "Rush Hour" }
    };

    std::ofstream runs_file(runs_path);
    if (!runs_file.is_open()) {
        std::cerr << "Error: Failed to open runs CSV file at " << runs_path << "\n";
        return 1;
    }
    runs_file << RunResult::csv_header() << "\n";

    std::vector<ScenarioStats> all_stats;
    size_t total_runs_done = 0;
    const size_t total_runs = sizeof(scenarios)/sizeof(scenarios[0]) * seeds.size();

    for (const auto& spec : scenarios) {
        std::cout << ">>> Running Scenario: " << spec.name << " (" << seeds.size() << " seeds)...\n";
        std::vector<RunResult> scenario_runs;
        scenario_runs.reserve(seeds.size());

        for (uint32_t seed : seeds) {
            Config cfg;
            cfg.scenario         = spec.scenario;
            cfg.seed             = seed;
            cfg.sim_duration_s   = sim_duration_s;
            cfg.headless         = true;

            Simulation sim{ cfg };
            sim.run_for(sim_duration_s);

            RunResult res = sim.get_run_result();
            scenario_runs.push_back(res);

            runs_file << res.to_csv_row() << "\n";
            runs_file.flush();

            total_runs_done++;
            std::cout << "  [" << std::setw(2) << total_runs_done << "/" << total_runs << "] "
                      << "Seed " << std::setw(5) << seed << " -> "
                      << "Spawned: " << std::setw(3) << res.total_spawned << ", "
                      << "Arrived: " << std::setw(3) << res.total_completed << ", "
                      << "AvgSpeed: " << std::fixed << std::setprecision(1) << std::setw(4) << res.avg_speed_kmh << " km/h, "
                      << "AvgWait: " << std::setprecision(1) << std::setw(4) << res.avg_waiting_time_s << " s, "
                      << "Throughput: " << std::setprecision(0) << std::setw(4) << res.throughput_vph << " veh/h\n";
        }

        all_stats.push_back(aggregate_results(spec.name, scenario_runs));
        std::cout << "\n";
    }
    runs_file.close();

    // ── Export Summary Statistics CSV ─────────────────────────────────────────
    std::ofstream stats_file(stats_path);
    if (!stats_file.is_open()) {
        std::cerr << "Error: Failed to open stats CSV file at " << stats_path << "\n";
        return 1;
    }

    stats_file << "scenario,metric,mean,std_dev,min,max\n";
    auto write_metric_row = [&](const std::string& sc, const std::string& metric, const ScenarioStats::MetricStat& m) {
        stats_file << std::fixed << std::setprecision(3)
                   << sc << ","
                   << metric << ","
                   << m.mean << ","
                   << m.std_dev << ","
                   << m.min_val << ","
                   << m.max_val << "\n";
    };

    for (const auto& s : all_stats) {
        write_metric_row(s.scenario_name, "avg_speed_kmh", s.speed_kmh);
        write_metric_row(s.scenario_name, "avg_waiting_time_s", s.wait_time_s);
        write_metric_row(s.scenario_name, "avg_travel_time_s", s.travel_time_s);
        write_metric_row(s.scenario_name, "max_queue_length", s.max_queue);
        write_metric_row(s.scenario_name, "avg_queue_length", s.avg_queue);
        write_metric_row(s.scenario_name, "throughput_vph", s.throughput_vph);
        write_metric_row(s.scenario_name, "congestion_pct", s.congestion_pct);
    }
    stats_file.close();

    // ── Print Formatted Summary Tables to Console ─────────────────────────────
    std::cout << "========================================================================================\n"
              << "                           BASELINE EXPERIMENTAL RESULTS SUMMARY                        \n"
              << "========================================================================================\n"
              << std::left
              << std::setw(18) << "Scenario"
              << std::setw(18) << "Avg Speed (km/h)"
              << std::setw(18) << "Wait Time (s)"
              << std::setw(18) << "Throughput (veh/h)"
              << std::setw(16) << "Max Queue"
              << std::setw(16) << "Congestion (%)"
              << "\n"
              << "----------------------------------------------------------------------------------------\n";

    for (const auto& s : all_stats) {
        std::ostringstream spd, wt, thr, mq, cg;
        spd << std::fixed << std::setprecision(1) << s.speed_kmh.mean << " +/- " << s.speed_kmh.std_dev;
        wt  << std::fixed << std::setprecision(1) << s.wait_time_s.mean << " +/- " << s.wait_time_s.std_dev;
        thr << std::fixed << std::setprecision(0) << s.throughput_vph.mean << " +/- " << s.throughput_vph.std_dev;
        mq  << std::fixed << std::setprecision(1) << s.max_queue.mean << " [" << (int)s.max_queue.min_val << "-" << (int)s.max_queue.max_val << "]";
        cg  << std::fixed << std::setprecision(1) << s.congestion_pct.mean << "%";

        std::cout << std::left
                  << std::setw(18) << s.scenario_name
                  << std::setw(18) << spd.str()
                  << std::setw(18) << wt.str()
                  << std::setw(18) << thr.str()
                  << std::setw(16) << mq.str()
                  << std::setw(16) << cg.str()
                  << "\n";
    }

    std::cout << "========================================================================================\n"
              << " Raw Runs CSV : " << runs_path << "\n"
              << " Stats CSV    : " << stats_path << "\n"
              << "========================================================================================\n\n";

    return 0;
}
