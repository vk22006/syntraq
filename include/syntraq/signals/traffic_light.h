#pragma once

//
// syntraq/signals/traffic_light.h
//
// Traffic light representation for intersection approach control.
// Supports three signal states: Red, Yellow, Green.
//

#include "syntraq/world/ids.h"
#include <string_view>

namespace syntraq {

// ── SignalColor ──────────────────────────────────────────────────────────────

enum class SignalColor : uint8_t {
    Red,
    Yellow,
    Green
};

[[nodiscard]] constexpr std::string_view to_string(SignalColor color) noexcept {
    switch (color) {
        case SignalColor::Red:    return "Red";
        case SignalColor::Yellow: return "Yellow";
        case SignalColor::Green:  return "Green";
    }
    return "Unknown";
}

// ── TrafficLight ─────────────────────────────────────────────────────────────

/// Represents a single traffic signal head controlling a specific incoming road.
struct TrafficLight {
    RoadId         road_id        { kInvalidRoadId };
    IntersectionId intersection_id{ kInvalidIntersectionId };
    SignalColor    color          { SignalColor::Red };

    [[nodiscard]] bool is_red()    const noexcept { return color == SignalColor::Red; }
    [[nodiscard]] bool is_yellow() const noexcept { return color == SignalColor::Yellow; }
    [[nodiscard]] bool is_green()  const noexcept { return color == SignalColor::Green; }

    /// Right-of-way permission: only Green permits entering the intersection.
    [[nodiscard]] bool can_proceed() const noexcept { return color == SignalColor::Green; }

    [[nodiscard]] bool is_valid() const noexcept {
        return road_id != kInvalidRoadId && intersection_id != kInvalidIntersectionId;
    }
};

} // namespace syntraq
