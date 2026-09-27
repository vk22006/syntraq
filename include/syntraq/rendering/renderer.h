#pragma once

//
// syntraq/rendering/renderer.h
//
// Renderer abstraction — wraps Raylib window lifecycle and drawing.
// The Renderer reads SimState, RoadNetwork, and Vehicles; it never writes.
// Dear ImGui is set up via rlImGui for debug overlays.
//
// NOTE: This header deliberately does NOT include raylib.h or imgui.h so
// that other modules (Simulation, tests) don't pull in those heavy headers
// transitively. The renderer is always compiled as a separate translation unit.
//

#include "syntraq/core/config.h"
#include "syntraq/simulation/simulation.h"
#include "syntraq/world/road_network.h"
#include "syntraq/vehicles/vehicle.h"

#include <vector>

namespace syntraq {

/// Manages the Raylib window + Dear ImGui context.
/// Constructed once in main(); destroyed when the program exits.
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

    /// One frame: begin → draw network → draw vehicles → ImGui → end.
    void render_frame(const SimState&           state,
                      const RoadNetwork&         network,
                      const std::vector<Vehicle>& vehicles);

private:
    void begin_frame();
    void draw_network(const RoadNetwork& network);
    void draw_vehicles(const RoadNetwork& network,
                       const std::vector<Vehicle>& vehicles);
    void draw_imgui(const SimState& state, const RoadNetwork& network);
    void end_frame();

    const Config& cfg_;
};

} // namespace syntraq
