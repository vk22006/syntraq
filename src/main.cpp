//
// src/main.cpp
//
// Application entry point.
// Responsibilities: parse argv, build Config, wire Simulation + Renderer,
// run the fixed-timestep simulation + rendering loop, and clean up.
//

#include "syntraq/core/config.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/rendering/renderer.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // ── Load configuration ───────────────────────────────────────────────
    std::string cfgPath = "configs/default_scenario.json";
    std::string screenshot_path;
    std::string csv_summary_path;
    std::string csv_timeseries_path;
    bool cli_headless = false;
    bool enable_debug_overlays = false;
    int  cli_width  = 0;
    int  cli_height = 0;
    std::optional<float> cli_duration;
    std::optional<uint32_t> cli_seed;
    std::optional<syntraq::TrafficScenario> cli_scenario;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless" || arg == "-h") {
            cli_headless = true;
        } else if (arg == "--debug-overlays") {
            enable_debug_overlays = true;
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            cli_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            cli_height = std::stoi(argv[++i]);
        } else if ((arg == "--duration" || arg == "-d") && i + 1 < argc) {
            cli_duration = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cli_seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if ((arg == "--scenario" || arg == "-s") && i + 1 < argc) {
            std::string sc = argv[++i];
            if (sc == "low" || sc == "Low") {
                cli_scenario = syntraq::TrafficScenario::Low;
            } else if (sc == "med" || sc == "medium" || sc == "Medium") {
                cli_scenario = syntraq::TrafficScenario::Medium;
            } else if (sc == "high" || sc == "High") {
                cli_scenario = syntraq::TrafficScenario::High;
            } else if (sc == "rush" || sc == "rushhour" || sc == "RushHour") {
                cli_scenario = syntraq::TrafficScenario::RushHour;
            }
        } else if ((arg == "--csv" || arg == "--csv-summary") && i + 1 < argc) {
            csv_summary_path = argv[++i];
        } else if (arg == "--csv-timeseries" && i + 1 < argc) {
            csv_timeseries_path = argv[++i];
        } else if (arg.rfind(".json") != std::string::npos) {
            cfgPath = arg;
        }
    }

    syntraq::Config cfg = syntraq::Config::from_file(cfgPath);
    if (cli_headless) {
        cfg.headless = true;
    }
    if (cli_width > 0) {
        cfg.window_width = cli_width;
    }
    if (cli_height > 0) {
        cfg.window_height = cli_height;
    }
    if (cli_duration.has_value()) {
        cfg.sim_duration_s = *cli_duration;
    }
    if (cli_seed.has_value()) {
        cfg.seed = *cli_seed;
    }
    if (cli_scenario.has_value()) {
        cfg.scenario = *cli_scenario;
    }

    // ── Headless mode ────────────────────────────────────────────────────
    if (cfg.headless) {
        syntraq::Simulation sim{ cfg };
        sim.run_for(cfg.sim_duration_s);
        const auto res = sim.get_run_result();

        std::cout << "\n======================================================\n"
                  << " [SyntraQ] Simulation Run Complete\n"
                  << "======================================================\n"
                  << " Scenario         : " << res.scenario_name << "\n"
                  << " Random Seed      : " << res.seed << "\n"
                  << " Duration         : " << res.duration_s << " s (" << res.total_ticks << " ticks)\n"
                  << " Total Spawned    : " << res.total_spawned << "\n"
                  << " Total Completed  : " << res.total_completed << "\n"
                  << " Active Remaining : " << res.active_remaining << "\n"
                  << " Avg Speed        : " << res.avg_speed_mps << " m/s (" << res.avg_speed_kmh << " km/h)\n"
                  << " Avg Waiting Time : " << res.avg_waiting_time_s << " s\n"
                  << " Avg Travel Time  : " << res.avg_travel_time_s << " s\n"
                  << " Max Queue Length : " << res.max_queue_length << " vehicles\n"
                  << " Avg Queue Length : " << res.avg_queue_length << " vehicles\n"
                  << " Throughput       : " << res.throughput_vph << " veh/h (" << res.throughput_vps << " veh/s)\n"
                  << " Congestion Ratio : " << (res.congestion_ratio * 100.0f) << "%\n"
                  << "======================================================\n\n";

        if (!csv_summary_path.empty()) {
            if (sim.export_summary_csv(csv_summary_path)) {
                std::cout << "[SyntraQ] Summary metrics exported to: " << csv_summary_path << "\n";
            } else {
                std::cerr << "[SyntraQ] Failed to export summary CSV to: " << csv_summary_path << "\n";
            }
        }

        if (!csv_timeseries_path.empty()) {
            if (sim.export_timeseries_csv(csv_timeseries_path)) {
                std::cout << "[SyntraQ] Time-series metrics exported to: " << csv_timeseries_path << "\n";
            } else {
                std::cerr << "[SyntraQ] Failed to export time-series CSV to: " << csv_timeseries_path << "\n";
            }
        }

        return 0;
    }

    // ── Windowed mode ────────────────────────────────────────────────────
    syntraq::Simulation sim{ cfg };
    syntraq::Renderer   renderer{ cfg };

    if (enable_debug_overlays) {
        renderer.debug_flags().show_intersection_ids = true;
        renderer.debug_flags().show_road_ids         = true;
        renderer.debug_flags().show_lane_boundaries  = true;
        renderer.debug_flags().show_vehicle_vectors  = true;
    }

    int frame_count = 0;
    while (!renderer.should_close()) {
        const float dt_real = renderer.frame_time();
        sim.update(dt_real);

        const auto action = renderer.render_frame(sim.state(),
                                                  sim.network(),
                                                  sim.vehicles(),
                                                  sim.time_scale(),
                                                  &sim.signal_controllers());

        if (action.request_pause_toggle) {
            sim.set_paused(!sim.is_paused());
        }
        if (action.request_reset) {
            sim.reset();
        }
        if (action.request_speed_scale.has_value()) {
            sim.set_time_scale(*action.request_speed_scale);
        }
        if (action.request_scenario.has_value()) {
            sim.set_scenario(*action.request_scenario);
        }

        frame_count++;
        if (!screenshot_path.empty() && (sim.state().active_vehicles >= 10 || frame_count >= 600)) {
            renderer.take_screenshot(screenshot_path);
            break;
        }
    }

    if (!csv_summary_path.empty()) {
        static_cast<void>(sim.export_summary_csv(csv_summary_path));
    }
    if (!csv_timeseries_path.empty()) {
        static_cast<void>(sim.export_timeseries_csv(csv_timeseries_path));
    }

    return 0;
}
