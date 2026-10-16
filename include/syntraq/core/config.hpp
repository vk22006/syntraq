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

// ── Traffic Scenarios ─────────────────────────────────────────

enum class TrafficScenario : uint8_t {
    Low,
    Medium,
    High,
    RushHour,
    Custom,
};

struct TrafficScenarioConfig {
    TrafficScenario scenario{ TrafficScenario::Medium };
    std::string     name{ "Medium Traffic" };
    float           spawn_interval_s{ 2.0f };
    uint32_t        max_vehicles{ 40 };
    uint32_t        seed{ 42u };
};

/// Returns preset configuration for a given repeatable traffic scenario.
inline TrafficScenarioConfig get_scenario_preset(TrafficScenario scenario, uint32_t seed = 42u) {
    TrafficScenarioConfig cfg;
    cfg.scenario = scenario;
    cfg.seed     = seed;
    switch (scenario) {
        case TrafficScenario::Low:
            cfg.name             = "Low Traffic";
            cfg.spawn_interval_s = 4.0f;
            cfg.max_vehicles     = 15;
            break;
        case TrafficScenario::Medium:
            cfg.name             = "Medium Traffic";
            cfg.spawn_interval_s = 2.0f;
            cfg.max_vehicles     = 35;
            break;
        case TrafficScenario::High:
            cfg.name             = "High Traffic";
            cfg.spawn_interval_s = 1.0f;
            cfg.max_vehicles     = 60;
            break;
        case TrafficScenario::RushHour:
            cfg.name             = "Rush Hour";
            cfg.spawn_interval_s = 0.5f;
            cfg.max_vehicles     = 100;
            break;
        case TrafficScenario::Custom:
            cfg.name             = "Custom";
            cfg.spawn_interval_s = 2.0f;
            cfg.max_vehicles     = 40;
            break;
    }
    return cfg;
}

/// All tunable simulation parameters in one plain-data struct.
/// Populated by load_config() from a JSON file, or default-constructed
/// for headless / test usage.
struct Config {
    // ── Window ──────────────────────────────────────────────────
    std::string window_title{ "SyntraQ" };
    int         window_width { 1600 };
    int         window_height{ 900 };
    int         target_fps   { 60 };

    // ── Simulation ───────────────────────────────────────────────
    float    dt_seconds       { 0.1f };   ///< Fixed simulation timestep
    uint32_t seed             { 42u };    ///< RNG seed (deterministic when fixed)
    float    sim_duration_s   { 300.0f }; ///< Headless run duration
    bool     headless         { false };  ///< Skip window / rendering when true

    // ── Traffic Behavior & Scenario ──────────────────────────────
    TrafficScenario scenario                { TrafficScenario::Medium };
    float           vehicle_spawn_interval_s{ 2.0f };
    uint32_t        max_vehicles            { 40 };

    // ── Scenario File (future use) ───────────────────────────────
    std::string scenario_file{ "configs/default_scenario.json" };

    // ── Traffic Signals ──────────────────────────────────────────
    bool  enable_traffic_signals   { true };   ///< Enable baseline traffic signals
    float default_green_duration_s { 10.0f };  ///< Fixed-time green stage duration
    float default_yellow_duration_s{  3.0f };  ///< Fixed-time yellow stage duration
    float default_all_red_duration_s{ 2.0f };  ///< Interphase all-red clearance duration

    /// Load from a JSON file; returns default config on failure.
    static Config from_file(const std::string& path);
};

} // namespace syntraq
