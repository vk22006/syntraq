#pragma once

//
// syntraq/rendering/renderer.h
//
// Renderer abstraction — wraps Raylib window lifecycle, 2D camera,
// and Dear ImGui simulation control / debug overlays.
//
// NOTE: This header deliberately does NOT include raylib.h or imgui.h so
// that other modules (Simulation, tests) don't pull in those heavy headers
// transitively. The renderer is always compiled as a separate translation unit.
//

#include "syntraq/core/config.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/world/road_network.h"
#include "syntraq/vehicles/vehicle.h"

#include <memory>
#include <optional>
#include <vector>

namespace syntraq {

/// Visual and debug toggles for rendering.
struct DebugRenderFlags {
    bool show_intersection_ids { false };
    bool show_road_ids         { false };
    bool show_lane_boundaries  { false };
    bool show_vehicle_vectors  { false };
    bool show_debug_statistics { true };
};

/// Simulation commands triggered from UI buttons or keyboard shortcuts.
struct SimControlAction {
    bool                 request_pause_toggle{ false };
    bool                 request_reset       { false };
    std::optional<float> request_speed_scale {};
};

/// Forward declaration for internal camera storage (avoids raylib.h in header).
struct RenderCamera;

/// Manages the Raylib window + Dear ImGui context + 2D navigation camera.
class Renderer {
public:
    explicit Renderer(const Config& cfg);
    ~Renderer();

    // Non-copyable, non-movable — owns OS window resources.
    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&)                 = delete;
    Renderer& operator=(Renderer&&)      = delete;

    /// Returns true when the user closes the window.
    [[nodiscard]] bool should_close() const;

    /// Elapsed real time since last frame in seconds.
    [[nodiscard]] float frame_time() const;

    /// One frame: camera update -> begin -> draw world -> ImGui -> end.
    /// Returns control actions requested by the user.
    SimControlAction render_frame(const SimState&             state,
                                  const RoadNetwork&          network,
                                  const std::vector<Vehicle>& vehicles,
                                  float                       current_speed_scale = 1.0f);

    /// Reset camera view to center and frame the road network with comfortable margins.
    void reset_camera(const RoadNetwork& network);

    /// Save current frame screenshot to file.
    void take_screenshot(const std::string& path) const;

    [[nodiscard]] const DebugRenderFlags& debug_flags() const noexcept { return debug_flags_; }
    [[nodiscard]]       DebugRenderFlags& debug_flags()       noexcept { return debug_flags_; }

private:
    void update_camera(const RoadNetwork& network);
    void begin_frame();
    void draw_world_grid();
    void draw_network(const RoadNetwork& network);
    void draw_vehicles(const RoadNetwork& network,
                       const std::vector<Vehicle>& vehicles);
    SimControlAction draw_imgui(const SimState&    state,
                                const RoadNetwork& network,
                                float              current_speed_scale);
    void draw_status_bar(const SimState& state, float current_speed_scale);
    void end_frame();

    const Config&                 cfg_;
    DebugRenderFlags              debug_flags_{};
    std::unique_ptr<RenderCamera> camera_;
    bool                          camera_initialized_{ false };
};

} // namespace syntraq
