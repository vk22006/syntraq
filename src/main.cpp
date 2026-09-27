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
    bool cli_headless = false;
    bool enable_debug_overlays = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless" || arg == "-h") {
            cli_headless = true;
        } else if (arg == "--debug-overlays") {
            enable_debug_overlays = true;
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (arg.rfind(".json") != std::string::npos) {
            cfgPath = arg;
        }
    }

    syntraq::Config cfg = syntraq::Config::from_file(cfgPath);
    if (cli_headless) {
        cfg.headless = true;
    }

    // ── Headless mode ────────────────────────────────────────────────────
    if (cfg.headless) {
        syntraq::Simulation sim{ cfg };
        sim.run_for(cfg.sim_duration_s);
        const auto& s = sim.state();
        std::cout << "[SyntraQ] Headless run complete.\n"
                  << "  Ticks  : " << s.tick          << "\n"
                  << "  Time   : " << s.elapsed_s     << " s\n"
                  << "  Spawned: " << s.total_spawned << "\n"
                  << "  Arrived: " << s.total_arrived << "\n";
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
                                                  sim.time_scale());

        if (action.request_pause_toggle) {
            sim.set_paused(!sim.is_paused());
        }
        if (action.request_reset) {
            sim.reset();
        }
        if (action.request_speed_scale.has_value()) {
            sim.set_time_scale(*action.request_speed_scale);
        }

        frame_count++;
        if (!screenshot_path.empty() && (sim.state().active_vehicles >= 10 || frame_count >= 600)) {
            renderer.take_screenshot(screenshot_path);
            break;
        }
    }

    return 0;
}
