//
// tests/test_config.cpp
//
// Verify the Config struct and Config::from_file() behave correctly.
//

#include "syntraq/core/config.h"
#include <gtest/gtest.h>

namespace syntraq {

// ── Default construction ─────────────────────────────────────────────────

TEST(ConfigTest, DefaultValuesAreReasonable) {
    Config cfg;
    EXPECT_EQ(cfg.window_width,  1600);
    EXPECT_EQ(cfg.window_height, 900);
    EXPECT_EQ(cfg.target_fps,    60);
    EXPECT_FLOAT_EQ(cfg.dt_seconds, 0.1f);
    EXPECT_EQ(cfg.seed, 42u);
    EXPECT_FALSE(cfg.headless);
    EXPECT_FALSE(cfg.window_title.empty());
}

TEST(ConfigTest, DefaultDtIsPositive) {
    Config cfg;
    EXPECT_GT(cfg.dt_seconds, 0.0f);
}

TEST(ConfigTest, DefaultSimDurationIsPositive) {
    Config cfg;
    EXPECT_GT(cfg.sim_duration_s, 0.0f);
}

// ── from_file with missing path returns defaults ─────────────────────────

TEST(ConfigTest, FromFileMissingPathReturnsDefaults) {
    // Should not throw; returns a usable default config.
    Config cfg = Config::from_file("nonexistent_path/no_such_file.json");
    EXPECT_GT(cfg.dt_seconds, 0.0f);
    EXPECT_GT(cfg.window_width, 0);
}

// ── Field mutation sanity ────────────────────────────────────────────────

TEST(ConfigTest, FieldsAreMutable) {
    Config cfg;
    cfg.seed = 12345u;
    cfg.headless = true;
    EXPECT_EQ(cfg.seed, 12345u);
    EXPECT_TRUE(cfg.headless);
}

} // namespace syntraq
