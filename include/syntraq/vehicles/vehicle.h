#pragma once

//
// syntraq/vehicles/vehicle.h
//
// Vehicle — an autonomous agent travelling along the road network.
//
// Design choices:
//   - Position is represented as (current_road, lane_index, progress_m):
//     progress_m is metres along the road from the "from" intersection.
//     This is clean for movement math and avoids redundant 2D coordinate
//     storage that would need to stay in sync with road topology.
//   - The Renderer interpolates 2D pixel position from progress_m at draw time.
//   - Route is a pre-computed ordered list of RoadIds; route_index points to
//     the road the vehicle is currently on.
//   - VehicleState is a separate enum for readability in logs and tests.
//

#include "syntraq/world/ids.h"
#include <cstdint>
#include <vector>

namespace syntraq {

// ── VehicleState ─────────────────────────────────────────────────────────────

enum class VehicleState : uint8_t {
  Moving,  ///< Accelerating / cruising / decelerating along a road
  Stopped, ///< Voluntarily stopped (future: red light, congestion)
  Arrived, ///< Reached destination — ready for despawn
};

// ── Vehicle ──────────────────────────────────────────────────────────────────

struct Vehicle {
  // ── Identity ───────────────────────────────────────────────────────────
  VehicleId id{kInvalidVehicleId};

  // ── Route ──────────────────────────────────────────────────────────────
  std::vector<RoadId> route; ///< Ordered roads from spawn to destination
  uint32_t route_index{0};   ///< Index of the road currently on

  IntersectionId destination{kInvalidIntersectionId}; ///< Final intersection

  // ── Position ───────────────────────────────────────────────────────────
  RoadId current_road{kInvalidRoadId}; ///< Road the vehicle is on
  uint32_t lane_index{0};              ///< Lane within that road
  float progress_m{0.f};               ///< Metres travelled along current road

  // ── Kinematics ─────────────────────────────────────────────────────────
  float speed_mps{0.f}; ///< Current speed (m/s)
  float max_speed_mps{
      13.89f};            ///< Speed cap (matches road speed limit, 50 km/h)
  float accel_mps2{3.0f}; ///< Acceleration magnitude (m/s²)
  float decel_mps2{5.0f}; ///< Deceleration magnitude (m/s²)

  // ── Lifetime stats ─────────────────────────────────────────────────────
  float travel_time_s{0.f}; ///< Cumulative time since spawn
  float distance_m{0.f};    ///< Cumulative distance travelled

  // ── State ──────────────────────────────────────────────────────────────
  VehicleState state{VehicleState::Moving};

  // ── Helpers ────────────────────────────────────────────────────────────

  [[nodiscard]] bool is_valid() const noexcept {
    return id != kInvalidVehicleId && !route.empty();
  }

  [[nodiscard]] bool has_arrived() const noexcept {
    return state == VehicleState::Arrived;
  }

  /// Remaining route length including the current road.
  [[nodiscard]] uint32_t roads_remaining() const noexcept {
    if (route_index >= route.size())
      return 0;
    return static_cast<uint32_t>(route.size() - route_index);
  }
};

} // namespace syntraq
