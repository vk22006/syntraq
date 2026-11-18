//
// src/experiments/experiment_runner.cpp
//
// Implementation of the SyntraQ Reproducible Experiment Framework.
//

#include "syntraq/experiments/experiment_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace syntraq {

namespace {

std::string controller_type_to_string(SignalControllerType type) {
    switch (type) {
        case SignalControllerType::FixedTime:             return "FixedTime";
        case SignalControllerType::Adaptive:              return "Adaptive";
        case SignalControllerType::ReinforcementLearning: return "ReinforcementLearning";
        case SignalControllerType::Coordinated:           return "Coordinated";
    }
    return "UnknownController";
}

std::string scenario_type_to_string(TrafficScenario scenario) {
    switch (scenario) {
        case TrafficScenario::Low:               return "Low Traffic";
        case TrafficScenario::Medium:            return "Medium Traffic";
        case TrafficScenario::High:              return "High Traffic";
        case TrafficScenario::RushHour:          return "Rush Hour";
        case TrafficScenario::EmergencyPriority: return "Emergency Priority";
        case TrafficScenario::Custom:            return "Custom";
    }
    return "UnknownScenario";
}

std::string routing_type_to_string(RoutingStrategyType routing) {
    switch (routing) {
        case RoutingStrategyType::Static:       return "Static";
        case RoutingStrategyType::TrafficAware: return "TrafficAware";
    }
    return "UnknownRouting";
}

MetricStatistics compute_series_stats(const std::vector<double>& values) {
    MetricStatistics s{};
    if (values.empty()) return s;

    const size_t n = values.size();
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    s.mean = sum / static_cast<double>(n);

    s.min_val = *std::min_element(values.begin(), values.end());
    s.max_val = *std::max_element(values.begin(), values.end());

    if (n > 1) {
        double sq_sum = 0.0;
        for (double v : values) {
            const double diff = v - s.mean;
            sq_sum += diff * diff;
        }
        s.std_dev = std::sqrt(sq_sum / static_cast<double>(n - 1));

        // Critical value for 95% confidence interval:
        // For small sample sizes (N=3..30), use standard Student-t approx or normal approx 1.96
        // For common N=5 seeds, Student-t t_0.025, df=4 is 2.776.
        double t_crit = 1.96;
        if (n == 2) t_crit = 12.706;
        else if (n == 3) t_crit = 4.303;
        else if (n == 4) t_crit = 3.182;
        else if (n == 5) t_crit = 2.776;
        else if (n == 10) t_crit = 2.262;
        else if (n <= 30) t_crit = 2.045;

        const double sem = s.std_dev / std::sqrt(static_cast<double>(n));
        s.ci95_lower = s.mean - t_crit * sem;
        s.ci95_upper = s.mean + t_crit * sem;
    } else {
        s.std_dev = 0.0;
        s.ci95_lower = s.mean;
        s.ci95_upper = s.mean;
    }

    return s;
}

std::string escape_json_str(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

} // namespace

// ── ExperimentConfig ────────────────────────────────────────────────────────

Config ExperimentConfig::to_simulation_config() const {
    Config cfg;
    cfg.seed                  = seed;
    cfg.sim_duration_s        = duration_s;
    cfg.dt_seconds            = dt_seconds;
    cfg.headless              = true;
    cfg.scenario              = scenario;
    cfg.vehicle_spawn_interval_s = spawn_interval_s;
    cfg.max_vehicles          = max_vehicles;
    cfg.signal_controller_type= controller;
    cfg.routing_strategy      = routing_mode;
    cfg.enable_emergency_priority = enable_emergency;
    cfg.enable_emergency_rerouting = enable_rerouting;

    if (!model_path.empty()) {
        cfg.rl_model_path   = model_path;
        cfg.enable_rl_model = true;
    } else if (model_version == "none" || model_version.empty()) {
        cfg.enable_rl_model = false;
    }

    return cfg;
}

// ── ExperimentResult ────────────────────────────────────────────────────────

std::string ExperimentResult::csv_header() {
    return "experiment_id,scenario,traffic_density,spawn_interval_s,max_vehicles,"
           "controller,routing_mode,model_version,seed,duration_s,"
           "total_spawned,total_completed,active_remaining,"
           "avg_speed_kmh,avg_waiting_time_s,avg_travel_time_s,"
           "max_queue_length,avg_queue_length,throughput_vph,congestion_ratio,"
           "emergency_completed,emergency_avg_travel_time_s,emergency_avg_waiting_time_s,emergency_avg_speed_mps,"
           "normal_avg_travel_time_s,normal_avg_waiting_time_s,normal_avg_speed_mps,"
           "wall_time_s,sim_fps,success\n";
}

std::string ExperimentResult::to_csv_row() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << config.experiment_id << ","
       << config.scenario_name << ","
       << config.traffic_density << ","
       << config.spawn_interval_s << ","
       << config.max_vehicles << ","
       << config.controller_name << ","
       << config.routing_name << ","
       << config.model_version << ","
       << config.seed << ","
       << config.duration_s << ","
       << metrics.total_spawned << ","
       << metrics.total_completed << ","
       << metrics.active_remaining << ","
       << metrics.avg_speed_kmh << ","
       << metrics.avg_waiting_time_s << ","
       << metrics.avg_travel_time_s << ","
       << metrics.max_queue_length << ","
       << metrics.avg_queue_length << ","
       << metrics.throughput_vph << ","
       << metrics.congestion_ratio << ","
       << metrics.emergency_completed << ","
       << metrics.emergency_avg_travel_time_s << ","
       << metrics.emergency_avg_waiting_time_s << ","
       << metrics.emergency_avg_speed_mps << ","
       << metrics.normal_avg_travel_time_s << ","
       << metrics.normal_avg_waiting_time_s << ","
       << metrics.normal_avg_speed_mps << ","
       << execution_wall_time_s << ","
       << sim_fps << ","
       << (success ? "1" : "0") << "\n";
    return ss.str();
}

std::string ExperimentResult::to_json_fragment(int indent) const {
    const std::string pad(indent, ' ');
    const std::string pad2(indent + 2, ' ');
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << pad << "{\n";
    ss << pad2 << "\"experiment_id\": \"" << escape_json_str(config.experiment_id) << "\",\n";
    ss << pad2 << "\"scenario\": \"" << escape_json_str(config.scenario_name) << "\",\n";
    ss << pad2 << "\"traffic_density\": " << config.traffic_density << ",\n";
    ss << pad2 << "\"spawn_interval_s\": " << config.spawn_interval_s << ",\n";
    ss << pad2 << "\"max_vehicles\": " << config.max_vehicles << ",\n";
    ss << pad2 << "\"controller\": \"" << escape_json_str(config.controller_name) << "\",\n";
    ss << pad2 << "\"routing_mode\": \"" << escape_json_str(config.routing_name) << "\",\n";
    ss << pad2 << "\"model_version\": \"" << escape_json_str(config.model_version) << "\",\n";
    ss << pad2 << "\"seed\": " << config.seed << ",\n";
    ss << pad2 << "\"duration_s\": " << config.duration_s << ",\n";
    ss << pad2 << "\"total_spawned\": " << metrics.total_spawned << ",\n";
    ss << pad2 << "\"total_completed\": " << metrics.total_completed << ",\n";
    ss << pad2 << "\"active_remaining\": " << metrics.active_remaining << ",\n";
    ss << pad2 << "\"avg_speed_kmh\": " << metrics.avg_speed_kmh << ",\n";
    ss << pad2 << "\"avg_waiting_time_s\": " << metrics.avg_waiting_time_s << ",\n";
    ss << pad2 << "\"avg_travel_time_s\": " << metrics.avg_travel_time_s << ",\n";
    ss << pad2 << "\"max_queue_length\": " << metrics.max_queue_length << ",\n";
    ss << pad2 << "\"avg_queue_length\": " << metrics.avg_queue_length << ",\n";
    ss << pad2 << "\"throughput_vph\": " << metrics.throughput_vph << ",\n";
    ss << pad2 << "\"congestion_ratio\": " << metrics.congestion_ratio << ",\n";
    ss << pad2 << "\"emergency_completed\": " << metrics.emergency_completed << ",\n";
    ss << pad2 << "\"emergency_avg_travel_time_s\": " << metrics.emergency_avg_travel_time_s << ",\n";
    ss << pad2 << "\"normal_avg_travel_time_s\": " << metrics.normal_avg_travel_time_s << ",\n";
    ss << pad2 << "\"execution_wall_time_s\": " << execution_wall_time_s << ",\n";
    ss << pad2 << "\"sim_fps\": " << sim_fps << ",\n";
    ss << pad2 << "\"success\": " << (success ? "true" : "false") << "\n";
    ss << pad << "}";
    return ss.str();
}

// ── ExperimentGroupSummary ──────────────────────────────────────────────────

std::string ExperimentGroupSummary::csv_header() {
    return "group_key,controller,scenario,routing_mode,model_version,traffic_density,"
           "sample_count,duration_s,"
           "speed_kmh_mean,speed_kmh_std,speed_kmh_ci95_low,speed_kmh_ci95_high,"
           "wait_time_s_mean,wait_time_s_std,wait_time_s_ci95_low,wait_time_s_ci95_high,"
           "travel_time_s_mean,travel_time_s_std,travel_time_s_ci95_low,travel_time_s_ci95_high,"
           "throughput_vph_mean,throughput_vph_std,throughput_vph_ci95_low,throughput_vph_ci95_high,"
           "congestion_pct_mean,congestion_pct_std,congestion_pct_ci95_low,congestion_pct_ci95_high,"
           "avg_queue_mean,avg_queue_std,max_queue_mean,max_queue_max\n";
}

std::string ExperimentGroupSummary::to_csv_row() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << group_key << ","
       << controller_name << ","
       << scenario_name << ","
       << routing_name << ","
       << model_version << ","
       << traffic_density << ","
       << sample_count << ","
       << duration_s << ","
       << speed_kmh.mean << "," << speed_kmh.std_dev << "," << speed_kmh.ci95_lower << "," << speed_kmh.ci95_upper << ","
       << wait_time_s.mean << "," << wait_time_s.std_dev << "," << wait_time_s.ci95_lower << "," << wait_time_s.ci95_upper << ","
       << travel_time_s.mean << "," << travel_time_s.std_dev << "," << travel_time_s.ci95_lower << "," << travel_time_s.ci95_upper << ","
       << throughput_vph.mean << "," << throughput_vph.std_dev << "," << throughput_vph.ci95_lower << "," << throughput_vph.ci95_upper << ","
       << congestion_pct.mean << "," << congestion_pct.std_dev << "," << congestion_pct.ci95_lower << "," << congestion_pct.ci95_upper << ","
       << avg_queue.mean << "," << avg_queue.std_dev << ","
       << max_queue.mean << "," << max_queue.max_val << "\n";
    return ss.str();
}

std::string ExperimentGroupSummary::to_json_fragment(int indent) const {
    const std::string pad(indent, ' ');
    const std::string pad2(indent + 2, ' ');
    const std::string pad3(indent + 4, ' ');
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);

    auto emit_metric = [&](const char* name, const MetricStatistics& m, bool last = false) {
        ss << pad2 << "\"" << name << "\": {\n";
        ss << pad3 << "\"mean\": " << m.mean << ",\n";
        ss << pad3 << "\"std_dev\": " << m.std_dev << ",\n";
        ss << pad3 << "\"min\": " << m.min_val << ",\n";
        ss << pad3 << "\"max\": " << m.max_val << ",\n";
        ss << pad3 << "\"ci95_lower\": " << m.ci95_lower << ",\n";
        ss << pad3 << "\"ci95_upper\": " << m.ci95_upper << "\n";
        ss << pad2 << "}" << (last ? "" : ",") << "\n";
    };

    ss << pad << "{\n";
    ss << pad2 << "\"group_key\": \"" << escape_json_str(group_key) << "\",\n";
    ss << pad2 << "\"controller\": \"" << escape_json_str(controller_name) << "\",\n";
    ss << pad2 << "\"scenario\": \"" << escape_json_str(scenario_name) << "\",\n";
    ss << pad2 << "\"routing_mode\": \"" << escape_json_str(routing_name) << "\",\n";
    ss << pad2 << "\"model_version\": \"" << escape_json_str(model_version) << "\",\n";
    ss << pad2 << "\"traffic_density\": " << traffic_density << ",\n";
    ss << pad2 << "\"sample_count\": " << sample_count << ",\n";
    ss << pad2 << "\"duration_s\": " << duration_s << ",\n";
    emit_metric("speed_kmh", speed_kmh);
    emit_metric("wait_time_s", wait_time_s);
    emit_metric("travel_time_s", travel_time_s);
    emit_metric("throughput_vph", throughput_vph);
    emit_metric("congestion_pct", congestion_pct);
    emit_metric("avg_queue", avg_queue);
    emit_metric("max_queue", max_queue, true);
    ss << pad << "}";
    return ss.str();
}

// ── ExperimentBatch ─────────────────────────────────────────────────────────

ExperimentBatch::ExperimentBatch(std::string batch_name)
    : name_(std::move(batch_name)) {}

void ExperimentBatch::add_experiment(ExperimentConfig config) {
    if (config.controller_name.empty()) {
        config.controller_name = controller_type_to_string(config.controller);
    }
    if (config.scenario_name.empty()) {
        config.scenario_name = scenario_type_to_string(config.scenario);
    }
    if (config.routing_name.empty()) {
        config.routing_name = routing_type_to_string(config.routing_mode);
    }
    experiments_.push_back(std::move(config));
}

void ExperimentBatch::add_grid(const std::vector<TrafficScenario>&     scenarios,
                              const std::vector<SignalControllerType>& controllers,
                              const std::vector<RoutingStrategyType>&  routing_modes,
                              const std::vector<uint32_t>&             seeds,
                              float                                    duration_s,
                              const std::vector<float>&                density_multipliers,
                              const std::string&                       model_version,
                              const std::string&                       model_path) {
    for (auto scenario : scenarios) {
        const auto base_preset = get_scenario_preset(scenario);
        const std::string sc_name = scenario_type_to_string(scenario);

        for (float density : density_multipliers) {
            const float spawn_interval = std::max(0.2f, base_preset.spawn_interval_s / density);
            const uint32_t max_veh = static_cast<uint32_t>(std::round(static_cast<float>(base_preset.max_vehicles) * density));

            for (auto ctrl : controllers) {
                const std::string ctrl_name = controller_type_to_string(ctrl);

                for (auto route : routing_modes) {
                    const std::string route_name = routing_type_to_string(route);

                    for (uint32_t seed : seeds) {
                        ExperimentConfig cfg;
                        cfg.scenario           = scenario;
                        cfg.scenario_name      = sc_name;
                        cfg.traffic_density    = density;
                        cfg.spawn_interval_s   = spawn_interval;
                        cfg.max_vehicles       = max_veh;
                        cfg.controller         = ctrl;
                        cfg.controller_name    = ctrl_name;
                        cfg.routing_mode       = route;
                        cfg.routing_name       = route_name;
                        cfg.seed               = seed;
                        cfg.duration_s         = duration_s;
                        cfg.model_version      = (ctrl == SignalControllerType::ReinforcementLearning) ? model_version : "none";
                        cfg.model_path         = (ctrl == SignalControllerType::ReinforcementLearning) ? model_path : "";
                        cfg.enable_emergency   = (scenario == TrafficScenario::EmergencyPriority);
                        cfg.enable_rerouting   = (scenario == TrafficScenario::EmergencyPriority);

                        std::ostringstream id_ss;
                        id_ss << ctrl_name << "__" << sc_name << "__" << route_name
                              << "__d" << std::fixed << std::setprecision(1) << density
                              << "__s" << seed;
                        cfg.experiment_id      = id_ss.str();

                        add_experiment(cfg);
                    }
                }
            }
        }
    }
}

// ── ExperimentRunner ────────────────────────────────────────────────────────

ExperimentRunner::ExperimentRunner(RunnerOptions options)
    : options_(std::move(options)) {
    if (options_.parallel && options_.num_threads == 0) {
        options_.num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
}

ExperimentResult ExperimentRunner::run_single(const ExperimentConfig& config) const {
    ExperimentResult res;
    res.config = config;

    const auto t_start = std::chrono::steady_clock::now();

    try {
        Config sim_cfg = config.to_simulation_config();

        std::unique_ptr<Simulation> sim;
        if (options_.custom_network_factory) {
            sim = std::make_unique<Simulation>(sim_cfg, options_.custom_network_factory);
        } else {
            sim = std::make_unique<Simulation>(sim_cfg);
        }

        // Apply controller type explicitly to ensure all intersection controllers are constructed
        sim->set_signal_controller_type(config.controller);
        sim->set_routing_strategy_type(config.routing_mode);
        sim->set_spawn_interval(config.spawn_interval_s);
        sim->set_max_vehicles(config.max_vehicles);
        sim->set_seed(config.seed);

        if (config.enable_emergency) {
            sim->set_emergency_priority_enabled(true);
            sim->set_emergency_rerouting_enabled(config.enable_rerouting);
        }

        // Execute discrete simulation loop headless
        sim->run_for(config.duration_s);

        res.metrics = sim->get_run_result();
        res.success = true;
    } catch (const std::exception& ex) {
        res.success = false;
        res.error_message = ex.what();
    } catch (...) {
        res.success = false;
        res.error_message = "Unknown exception during simulation execution.";
    }

    const auto t_end = std::chrono::steady_clock::now();
    const std::chrono::duration<float> elapsed = t_end - t_start;
    res.execution_wall_time_s = elapsed.count();

    if (res.execution_wall_time_s > 1e-6f) {
        const float total_ticks = config.duration_s / config.dt_seconds;
        res.sim_fps = total_ticks / res.execution_wall_time_s;
    }

    return res;
}

std::vector<ExperimentResult> ExperimentRunner::run_batch(const ExperimentBatch& batch) const {
    const auto& experiments = batch.experiments();
    const size_t total = experiments.size();
    std::vector<ExperimentResult> results(total);

    if (options_.verbose) {
        std::cout << "[ExperimentRunner] Starting batch '" << batch.name() << "' with "
                  << total << " experiments (Mode: "
                  << (options_.parallel ? ("Parallel, " + std::to_string(options_.num_threads) + " threads") : "Sequential")
                  << ")\n";
    }

    if (!options_.parallel || options_.num_threads <= 1 || total <= 1) {
        // Sequential execution
        for (size_t i = 0; i < total; ++i) {
            if (options_.verbose) {
                std::cout << "\r[ExperimentRunner] Executing [" << (i + 1) << "/" << total << "] "
                          << experiments[i].experiment_id << "..." << std::flush;
            }
            results[i] = run_single(experiments[i]);
        }
        if (options_.verbose) std::cout << "\n";
    } else {
        // Parallel execution across worker threads
        const size_t num_threads = std::min(options_.num_threads, total);
        std::atomic<size_t> next_idx{ 0 };
        std::atomic<size_t> completed{ 0 };
        std::vector<std::thread> workers;
        workers.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            workers.emplace_back([&]() {
                while (true) {
                    const size_t idx = next_idx.fetch_add(1);
                    if (idx >= total) break;

                    results[idx] = run_single(experiments[idx]);

                    const size_t c = completed.fetch_add(1) + 1;
                    if (options_.verbose) {
                        // Periodic console update
                        if (c % 5 == 0 || c == total) {
                            std::cout << "\r[ExperimentRunner] Progress: " << c << "/" << total
                                      << " (" << static_cast<int>(c * 100 / total) << "%)" << std::flush;
                        }
                    }
                }
            });
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
        if (options_.verbose) std::cout << "\n";
    }

    if (options_.verbose) {
        std::cout << "[ExperimentRunner] Finished batch '" << batch.name() << "'. Total runs: " << total << "\n";
    }

    return results;
}

std::vector<ExperimentGroupSummary>
ExperimentRunner::aggregate_results(const std::vector<ExperimentResult>& results) {
    // Group results by (controller, scenario, routing_mode, model_version, density)
    std::map<std::string, std::vector<const ExperimentResult*>> groups;

    for (const auto& r : results) {
        if (!r.success) continue;
        std::ostringstream ss;
        ss << r.config.controller_name << "__"
           << r.config.scenario_name << "__"
           << r.config.routing_name << "__"
           << r.config.model_version << "__d"
           << std::fixed << std::setprecision(1) << r.config.traffic_density;
        groups[ss.str()].push_back(&r);
    }

    std::vector<ExperimentGroupSummary> summaries;
    summaries.reserve(groups.size());

    for (const auto& [key, list] : groups) {
        if (list.empty()) continue;

        const auto& first_cfg = list.front()->config;
        ExperimentGroupSummary sum;
        sum.group_key       = key;
        sum.controller_name = first_cfg.controller_name;
        sum.scenario_name   = first_cfg.scenario_name;
        sum.routing_name    = first_cfg.routing_name;
        sum.model_version   = first_cfg.model_version;
        sum.traffic_density = first_cfg.traffic_density;
        sum.duration_s      = first_cfg.duration_s;
        sum.sample_count    = list.size();

        std::vector<double> speeds;
        std::vector<double> waits;
        std::vector<double> travels;
        std::vector<double> max_qs;
        std::vector<double> avg_qs;
        std::vector<double> thrs;
        std::vector<double> congs;
        std::vector<double> emg_travels;
        std::vector<double> norm_travels;

        for (const auto* r : list) {
            sum.seeds.push_back(r->config.seed);
            speeds.push_back(r->metrics.avg_speed_kmh);
            waits.push_back(r->metrics.avg_waiting_time_s);
            travels.push_back(r->metrics.avg_travel_time_s);
            max_qs.push_back(r->metrics.max_queue_length);
            avg_qs.push_back(r->metrics.avg_queue_length);
            thrs.push_back(r->metrics.throughput_vph);
            congs.push_back(r->metrics.congestion_ratio * 100.0);

            if (r->metrics.emergency_completed > 0) {
                emg_travels.push_back(r->metrics.emergency_avg_travel_time_s);
            }
            if (r->metrics.normal_avg_travel_time_s > 0.0f) {
                norm_travels.push_back(r->metrics.normal_avg_travel_time_s);
            }
        }

        sum.speed_kmh      = compute_series_stats(speeds);
        sum.wait_time_s    = compute_series_stats(waits);
        sum.travel_time_s  = compute_series_stats(travels);
        sum.max_queue      = compute_series_stats(max_qs);
        sum.avg_queue      = compute_series_stats(avg_qs);
        sum.throughput_vph = compute_series_stats(thrs);
        sum.congestion_pct = compute_series_stats(congs);
        sum.emergency_travel_time_s = compute_series_stats(emg_travels);
        sum.normal_travel_time_s    = compute_series_stats(norm_travels);

        summaries.push_back(std::move(sum));
    }

    return summaries;
}

bool ExperimentRunner::export_runs_csv(const std::vector<ExperimentResult>& results,
                                      const std::string&                  filepath) {
    const fs::path p(filepath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << ExperimentResult::csv_header();
    for (const auto& r : results) {
        out << r.to_csv_row();
    }
    return true;
}

bool ExperimentRunner::export_summary_csv(const std::vector<ExperimentGroupSummary>& summaries,
                                         const std::string&                        filepath) {
    const fs::path p(filepath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << ExperimentGroupSummary::csv_header();
    for (const auto& s : summaries) {
        out << s.to_csv_row();
    }
    return true;
}

bool ExperimentRunner::export_json(const std::string&                         batch_name,
                                  const std::vector<ExperimentResult>&       results,
                                  const std::vector<ExperimentGroupSummary>& summaries,
                                  const std::string&                         filepath) {
    const fs::path p(filepath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "{\n";
    out << "  \"batch_name\": \"" << escape_json_str(batch_name) << "\",\n";
    out << "  \"total_runs\": " << results.size() << ",\n";
    out << "  \"total_groups\": " << summaries.size() << ",\n";

    // Summary groups
    out << "  \"summaries\": [\n";
    for (size_t i = 0; i < summaries.size(); ++i) {
        out << summaries[i].to_json_fragment(4);
        if (i + 1 < summaries.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    // Raw runs
    out << "  \"runs\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        out << results[i].to_json_fragment(4);
        if (i + 1 < results.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    return true;
}

} // namespace syntraq

