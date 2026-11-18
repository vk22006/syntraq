//
// experiments/run_reproducible_experiments.cpp
//
// Reproducible Experiment CLI & Batch Runner for SyntraQ.
// Supports running parametric experiment sweeps comparing:
//  - Controllers: FixedTime, Adaptive, ReinforcementLearning, Coordinated
//  - Scenarios: Low, Medium, High, RushHour, EmergencyPriority
//  - Routing Modes: Static, TrafficAware
//  - Densities: Multiple multipliers
//  - Seeds: Multiple repeatable RNG seeds
//
// CLI flags:
//  --parallel / -p          : Run batch experiments concurrently across CPU threads
//  --threads <N>            : Worker thread count (default: auto)
//  --duration <s=180>       : Simulation duration per run in seconds
//  --seeds <s1,s2,...>      : Comma-separated list of random seeds (default: 42,101,202,303,404)
//  --controllers <c1,c2,...>: Filter controllers (fixed,adaptive,rl,coordinated)
//  --scenarios <s1,s2,...>  : Filter scenarios (low,medium,high,rush,emergency)
//  --routing <r1,r2,...>    : Filter routing strategies (static,traffic_aware)
//  --densities <d1,d2,...>  : Density multipliers (e.g. 0.8,1.0,1.2)
//  --model-path <path>      : ONNX policy path for RL controller
//  --model-version <tag>    : Policy version tag
//  --out-dir <path>         : Output directory for CSV / JSON artifacts
//  --quick                  : Run quick test suite (30s duration, 2 seeds)
//

#include "syntraq/experiments/experiment_runner.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace syntraq;

namespace {

std::vector<std::string> split_comma(const std::string& str) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) tokens.push_back(item);
    }
    return tokens;
}

} // namespace

int main(int argc, char* argv[]) {
    ExperimentRunner::RunnerOptions runner_opts;
    runner_opts.parallel    = false;
    runner_opts.num_threads = 0;
    runner_opts.verbose     = true;

    float duration_s = 180.0f;
    std::vector<uint32_t> seeds = { 42u, 101u, 202u, 303u, 404u };
    std::vector<TrafficScenario> scenarios = {
        TrafficScenario::Low,
        TrafficScenario::Medium,
        TrafficScenario::High,
        TrafficScenario::RushHour
    };
    std::vector<SignalControllerType> controllers = {
        SignalControllerType::FixedTime,
        SignalControllerType::Adaptive,
        SignalControllerType::ReinforcementLearning,
        SignalControllerType::Coordinated
    };
    std::vector<RoutingStrategyType> routing_modes = {
        RoutingStrategyType::Static,
        RoutingStrategyType::TrafficAware
    };
    std::vector<float> densities = { 1.0f };

    std::string model_version = "dqn_v1";
    std::string model_path    = "models/traffic_signal_dqn.onnx";
    std::string out_dir       = "data/experiments/reproducible";
    std::string batch_name    = "Full_Controller_Routing_Matrix";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--parallel" || arg == "-p") {
            runner_opts.parallel = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            runner_opts.num_threads = static_cast<size_t>(std::stoul(argv[++i]));
            runner_opts.parallel = true;
        } else if (arg == "--duration" || arg == "-d") {
            if (i + 1 < argc) duration_s = std::stof(argv[++i]);
        } else if (arg == "--seeds" && i + 1 < argc) {
            seeds.clear();
            for (const auto& tok : split_comma(argv[++i])) {
                seeds.push_back(static_cast<uint32_t>(std::stoul(tok)));
            }
        } else if (arg == "--quick") {
            duration_s = 30.0f;
            seeds = { 42u, 101u };
            densities = { 1.0f };
            scenarios = { TrafficScenario::Medium, TrafficScenario::RushHour };
            controllers = {
                SignalControllerType::FixedTime,
                SignalControllerType::Adaptive,
                SignalControllerType::ReinforcementLearning,
                SignalControllerType::Coordinated
            };
            routing_modes = { RoutingStrategyType::Static, RoutingStrategyType::TrafficAware };
            batch_name = "Quick_Validation_Sweep";
        } else if (arg == "--densities" && i + 1 < argc) {
            densities.clear();
            for (const auto& tok : split_comma(argv[++i])) {
                densities.push_back(std::stof(tok));
            }
        } else if (arg == "--controllers" && i + 1 < argc) {
            controllers.clear();
            for (const auto& tok : split_comma(argv[++i])) {
                if (tok == "fixed" || tok == "fixed_time" || tok == "FixedTime") {
                    controllers.push_back(SignalControllerType::FixedTime);
                } else if (tok == "adaptive" || tok == "Adaptive") {
                    controllers.push_back(SignalControllerType::Adaptive);
                } else if (tok == "rl" || tok == "RL" || tok == "ReinforcementLearning") {
                    controllers.push_back(SignalControllerType::ReinforcementLearning);
                } else if (tok == "coordinated" || tok == "Coordinated") {
                    controllers.push_back(SignalControllerType::Coordinated);
                }
            }
        } else if (arg == "--scenarios" && i + 1 < argc) {
            scenarios.clear();
            for (const auto& tok : split_comma(argv[++i])) {
                if (tok == "low" || tok == "Low") scenarios.push_back(TrafficScenario::Low);
                else if (tok == "medium" || tok == "med" || tok == "Medium") scenarios.push_back(TrafficScenario::Medium);
                else if (tok == "high" || tok == "High") scenarios.push_back(TrafficScenario::High);
                else if (tok == "rush" || tok == "RushHour") scenarios.push_back(TrafficScenario::RushHour);
                else if (tok == "emergency" || tok == "Emergency") scenarios.push_back(TrafficScenario::EmergencyPriority);
            }
        } else if (arg == "--routing" && i + 1 < argc) {
            routing_modes.clear();
            for (const auto& tok : split_comma(argv[++i])) {
                if (tok == "static" || tok == "Static") routing_modes.push_back(RoutingStrategyType::Static);
                else if (tok == "traffic_aware" || tok == "trafficaware" || tok == "TrafficAware") {
                    routing_modes.push_back(RoutingStrategyType::TrafficAware);
                }
            }
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--model-version" && i + 1 < argc) {
            model_version = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--batch-name" && i + 1 < argc) {
            batch_name = argv[++i];
        }
    }

    std::cout << "\n======================================================================\n";
    std::cout << " [SyntraQ] Reproducible Experiment Framework Runner\n";
    std::cout << " Batch Name: " << batch_name << "\n";
    std::cout << " Controllers: " << controllers.size() << " | Scenarios: " << scenarios.size()
              << " | Routing Modes: " << routing_modes.size() << " | Densities: " << densities.size()
              << " | Seeds: " << seeds.size() << "\n";
    std::cout << " Run Duration: " << duration_s << "s | Model: " << model_version << " (" << model_path << ")\n";
    std::cout << " Output Directory: " << out_dir << "\n";
    std::cout << "======================================================================\n\n";

    ExperimentBatch batch(batch_name);
    batch.add_grid(scenarios,
                   controllers,
                   routing_modes,
                   seeds,
                   duration_s,
                   densities,
                   model_version,
                   model_path);

    std::cout << "[SyntraQ] Configured " << batch.size() << " reproducible experiment runs.\n";

    ExperimentRunner runner(runner_opts);
    const auto results = runner.run_batch(batch);

    // Aggregate grouped results
    const auto summaries = ExperimentRunner::aggregate_results(results);

    // Export artifacts
    const std::string runs_csv    = out_dir + "/experiment_runs.csv";
    const std::string summary_csv = out_dir + "/experiment_summary.csv";
    const std::string batch_json  = out_dir + "/experiment_results.json";

    if (ExperimentRunner::export_runs_csv(results, runs_csv)) {
        std::cout << "[SyntraQ] Exported detailed runs to: " << runs_csv << "\n";
    } else {
        std::cerr << "[SyntraQ] Failed to export runs CSV to: " << runs_csv << "\n";
    }

    if (ExperimentRunner::export_summary_csv(summaries, summary_csv)) {
        std::cout << "[SyntraQ] Exported aggregated summary to: " << summary_csv << "\n";
    } else {
        std::cerr << "[SyntraQ] Failed to export summary CSV to: " << summary_csv << "\n";
    }

    if (ExperimentRunner::export_json(batch_name, results, summaries, batch_json)) {
        std::cout << "[SyntraQ] Exported comprehensive JSON to: " << batch_json << "\n";
    } else {
        std::cerr << "[SyntraQ] Failed to export JSON to: " << batch_json << "\n";
    }

    std::cout << "\n[SyntraQ] Experiment sweep complete! Total runs executed: " << results.size() << "\n\n";
    return 0;
}

