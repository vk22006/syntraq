#pragma once

//
// syntraq/world/ids.h
//
// Strongly-typed ID types for world entities.
// Using enum class over a uint32_t base gives us zero-cost type safety:
// an IntersectionId cannot accidentally be passed where a RoadId is expected.
//

#include <cstdint>
#include <functional>   // std::hash
#include <limits>

namespace syntraq {

// ── Intersection ID ──────────────────────────────────────────────────────────

enum class IntersectionId : uint32_t {};

inline constexpr IntersectionId kInvalidIntersectionId{
    std::numeric_limits<uint32_t>::max()
};

// ── Road ID ──────────────────────────────────────────────────────────────────

enum class RoadId : uint32_t {};

inline constexpr RoadId kInvalidRoadId{
    std::numeric_limits<uint32_t>::max()
};

} // namespace syntraq

// ── std::hash specialisations ────────────────────────────────────────────────
// Required so IDs can be used as unordered_map keys.

namespace std {

template <>
struct hash<syntraq::IntersectionId> {
    size_t operator()(syntraq::IntersectionId id) const noexcept {
        return hash<uint32_t>{}(static_cast<uint32_t>(id));
    }
};

template <>
struct hash<syntraq::RoadId> {
    size_t operator()(syntraq::RoadId id) const noexcept {
        return hash<uint32_t>{}(static_cast<uint32_t>(id));
    }
};

} // namespace std
