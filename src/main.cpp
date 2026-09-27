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
        std::cout << "[SyntraQ] Headless run complete. "
                  << "Ticks: "   << sim.state().tick
                  << "  Time: "  << sim.state().elapsed_s << "s\n"
                  << "  Intersections: " << sim.network().intersection_count()
                  << "  Roads: "         << sim.network().road_count() << "\n";
        return 0;
    }

    // ── Windowed mode ────────────────────────────────────────────────────
    syntraq::Simulation sim{ cfg };
    syntraq::Renderer   renderer{ cfg };

    while (!renderer.should_close()) {
        sim.tick();
        renderer.render_frame(sim.state(), sim.network());
    }

    return 0;
}
