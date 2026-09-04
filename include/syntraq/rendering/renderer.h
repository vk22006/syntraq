#pragma once

//
// syntraq/rendering/renderer.h
//
// Renderer abstraction — wraps Raylib window lifecycle and drawing.
// The Renderer reads SimState; it never modifies Simulation internals.
// Dear ImGui is set up here via rlImGui so debug overlays can be added
// incrementally in future milestones.
//
// NOTE: This header deliberately does NOT include raylib.h or imgui.h so
// that other modules (Simulation, tests) don't pull in those heavy headers
// transitively. The renderer is always compiled as a separate translation unit.
//

#include "syntraq/core/config.h"
#include "syntraq/simulation/simulation.h"

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

    /// Returns false when the user closes the window.
    [[nodiscard]] bool should_close() const;

    /// One frame: begin-frame → draw → imgui → end-frame.
    void render_frame(const SimState& state);

private:
    void begin_frame();
    void draw_scene(const SimState& state);
    void draw_imgui(const SimState& state);
    void end_frame();

    const Config& cfg_;
};

} // namespace syntraq
