//
// experiments/generate_dataset.cpp
//
// Automated dataset generator for traffic prediction model development.
// Executes multiple simulation runs across varied traffic scenarios (Low, Medium,
// High, Rush Hour) and random seeds. Samples multi-dimensional state features at
// fixed intervals and extracts forward lookahead ground-truth prediction targets.
// Exports clean, structured training data to CSV.
//

#include "syntraq/core/config.hpp"
#include "syntraq/metrics/traffic_data_collector.hpp"
#include "syntraq/simulation/simulation.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace syntraq;

int main(int argc, char* argv[]) {
    uint32_t runs_per_scenario = 5;       // 5 seeds x 4 scenarios = 20 runs
    float    duration_s        = 120.0f;  // 120 simulation seconds per run
    float    sample_interval_s = 1.0f;   // 1 sample per second
    std::string output_path    = "data/dataset/raw_traffic_data.csv";

    // CLI argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--runs" && i + 1 < argc) {
            runs_per_scenario = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_s = std::stof(argv[++i]);
        } else if (arg == "--interval" && i + 1 < argc) {
            sample_interval_s = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: syntraq_generate_dataset [options]\n"
                      << "  --runs <N>        Runs per scenario preset (default: 5)\n"
                      << "  --duration <sec>  Simulation duration per run in seconds (default: 120.0)\n"
                      << "  --interval <sec>  Sample interval in seconds (default: 1.0)\n"
                      << "  --output <path>   Output CSV filepath (default: data/dataset/raw_traffic_data.csv)\n";
            return 0;
        }
    }

    // Ensure output directory exists
    const fs::path p(output_path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    // Clear old file if exists
    if (fs::exists(output_path)) {
        fs::remove(output_path);
    }

    const std::vector<TrafficScenario> scenarios = {
        TrafficScenario::Low,
        TrafficScenario::Medium,
        TrafficScenario::High,
        TrafficScenario::RushHour
    };

    std::cout << "========================================================================================\n"
              << "                      SYNTRAQ TRAINING DATASET GENERATION HARNESS                       \n"
              << "========================================================================================\n"
              << " Target Output File   : " << output_path << "\n"
              << " Scenarios            : Low, Medium, High, Rush Hour (4 total)\n"
              << " Runs per Scenario    : " << runs_per_scenario << " (Total: " << (runs_per_scenario * scenarios.size()) << " runs)\n"
              << " Duration per Run     : " << duration_s << " s\n"
              << " Sample Interval      : " << sample_interval_s << " s\n"
              << " Prediction Horizons  : 15.0 s (Short) & 30.0 s (Medium)\n"
              << "========================================================================================\n\n";

    const auto start_wall = std::chrono::steady_clock::now();
    uint32_t global_run_idx = 0;
    uint32_t total_samples_exported = 0;

    struct ScenarioSummary {
        std::string name;
        uint32_t    runs{ 0 };
        uint32_t    samples{ 0 };
    };
    std::vector<ScenarioSummary> summary_stats;

    for (const auto& scenario : scenarios) {
        ScenarioSummary sc_summary;

        for (uint32_t r = 0; r < runs_per_scenario; ++r) {
            ++global_run_idx;
            const uint32_t seed = 1000u + r * 137u + static_cast<uint32_t>(scenario) * 31u;

            std::ostringstream run_id_ss;
            run_id_ss << "run_" << std::setfill('0') << std::setw(3) << global_run_idx;
            const std::string run_id = run_id_ss.str();

            Config cfg;
            cfg.scenario               = scenario;
            cfg.seed                   = seed;
            cfg.enable_traffic_signals = true;
            cfg.headless               = true;

            Simulation sim{ cfg };
            sim.set_run_id(run_id);
            sim.enable_data_collection(true, sample_interval_s);
            sim.set_scenario(scenario, seed);

            if (r == 0) {
                sc_summary.name = sim.scenario_config().name;
            }

            // Execute headless simulation
            sim.run_for(duration_s);
            sim.data_collector().finalize();

            // Export to CSV
            const auto valid = sim.data_collector().valid_samples(/* require_30s = */ true);
            const bool append = (global_run_idx > 1);
            static_cast<void>(sim.export_training_data_csv(output_path, append, /* require_30s = */ true));

            sc_summary.runs++;
            sc_summary.samples += static_cast<uint32_t>(valid.size());
            total_samples_exported += static_cast<uint32_t>(valid.size());

            std::cout << " [Run " << std::setw(3) << global_run_idx << "] "
                      << std::left << std::setw(16) << sim.scenario_config().name
                      << " Seed: " << std::setw(6) << seed
                      << " Duration: " << std::setw(4) << duration_s << "s"
                      << " -> Captured: " << std::setw(4) << valid.size() << " valid training samples\n";
        }
        summary_stats.push_back(sc_summary);
    }

    const auto end_wall = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(end_wall - start_wall).count();

    std::cout << "\n========================================================================================\n"
              << "                             DATASET GENERATION SUMMARY                                 \n"
              << "========================================================================================\n"
              << std::left
              << std::setw(25) << "Scenario"
              << std::setw(15) << "Completed Runs"
              << std::setw(20) << "Training Samples"
              << "\n"
              << "----------------------------------------------------------------------------------------\n";

    for (const auto& s : summary_stats) {
        std::cout << std::left
                  << std::setw(25) << s.name
                  << std::setw(15) << s.runs
                  << std::setw(20) << s.samples
                  << "\n";
    }

    std::cout << "----------------------------------------------------------------------------------------\n"
              << " Total Runs Executed  : " << global_run_idx << "\n"
              << " Total Valid Samples  : " << total_samples_exported << "\n"
              << " Wallclock Time       : " << std::fixed << std::setprecision(2) << elapsed_sec << " s\n"
              << " Output CSV Location  : " << output_path << "\n"
              << "========================================================================================\n\n";

    return 0;
}
