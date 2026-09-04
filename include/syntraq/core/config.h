#pragma once

//
// syntraq/core/config.h
//
// Lightweight simulation configuration.
// Loaded once at startup; all subsystems read from this struct.
// Deliberately simple — not a large framework.
//

#include <cstdint>
#include <string>

namespace syntraq {

/// All tunable simulation parameters in one plain-data struct.
/// Populated by load_config() from a JSON file, or default-constructed
/// for headless / test usage.
struct Config {
    // ── Window ──────────────────────────────────────────────────
    std::string window_title{ "SyntraQ" };
    int         window_width { 1280 };
    int         window_height{ 720 };
    int         target_fps   { 60 };

    // ── Simulation ───────────────────────────────────────────────
    float    dt_seconds       { 0.1f };   ///< Fixed simulation timestep
    uint32_t seed             { 42u };    ///< RNG seed (deterministic when fixed)
    float    sim_duration_s   { 300.0f }; ///< Headless run duration
    bool     headless         { false };  ///< Skip window / rendering when true

    // ── Scenario (future use) ────────────────────────────────────
    std::string scenario_file{ "configs/default_scenario.json" };

    /// Load from a JSON file; returns default config on failure.
    static Config from_file(const std::string& path);
};

} // namespace syntraq
