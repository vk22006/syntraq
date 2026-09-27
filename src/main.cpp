//
// src/main.cpp
//
// Application entry point.
// Responsibilities: parse argv, build Config, wire Simulation + Renderer,
// run the main loop, and clean up. Nothing else.
//

#include "syntraq/core/config.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/rendering/renderer.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // ── Load configuration ───────────────────────────────────────────────
    std::string cfgPath = "configs/default_scenario.json";
    if (argc > 1) {
        cfgPath = argv[1];
    }

    syntraq::Config cfg = syntraq::Config::from_file(cfgPath);

    // ── Headless mode ────────────────────────────────────────────────────
    if (cfg.headless) {
        syntraq::Simulation sim{ cfg };
        sim.run_for(cfg.sim_duration_s);
        const auto& s = sim.state();
        std::cout << "[SyntraQ] Headless run complete.\n"
                  << "  Ticks  : " << s.tick       << "\n"
                  << "  Time   : " << s.elapsed_s  << " s\n"
                  << "  Spawned: " << s.total_spawned << "\n"
                  << "  Arrived: " << s.total_arrived << "\n";
        return 0;
    }

    // ── Windowed mode ────────────────────────────────────────────────────
    syntraq::Simulation sim{ cfg };
    syntraq::Renderer   renderer{ cfg };

    while (!renderer.should_close()) {
        sim.tick();
        renderer.render_frame(sim.state(), sim.network(), sim.vehicles());
    }

    return 0;
}
