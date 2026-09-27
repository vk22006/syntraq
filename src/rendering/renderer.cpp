//
// src/rendering/renderer.cpp
//
// Renderer implementation.
// All Raylib and ImGui includes are confined to this translation unit.
//

#include "syntraq/rendering/renderer.h"

// Third-party headers — isolated to this .cpp
#include "raylib.h"
#include "rlImGui.h"
#include "imgui.h"

#include <cmath>
#include <string>

namespace syntraq {

// ── Palette ───────────────────────────────────────────────────────────────────

namespace pal {
    // Background
    constexpr Color bg           = {  12,  14,  20, 255 };
    // Road surface
    constexpr Color road_fill    = {  40,  44,  58, 255 };
    // Lane divider (dashed centre line)
    constexpr Color lane_divider = {  90, 100, 120, 180 };
    // Kerb / road edge
    constexpr Color road_edge    = {  65,  72,  90, 255 };
    // Intersection node fill
    constexpr Color node_fill    = {  28,  32,  45, 255 };
    // Intersection node border
    constexpr Color node_border  = {  80, 200, 255, 255 };
    // Intersection label
    constexpr Color node_label   = { 140, 180, 220, 200 };
    // Status bar text
    constexpr Color status_text  = { 100, 180, 100, 255 };
}

// ── Construction / destruction ────────────────────────────────────────────────

Renderer::Renderer(const Config& cfg)
    : cfg_(cfg)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(cfg_.window_width, cfg_.window_height,
               cfg_.window_title.c_str());
    SetTargetFPS(cfg_.target_fps);

    // Initialise rlImGui (sets up ImGui context + raylib backend)
    rlImGuiSetup(true);  // true = use default font
}

Renderer::~Renderer() {
    rlImGuiShutdown();
    CloseWindow();
}

// ── Public interface ──────────────────────────────────────────────────────────

bool Renderer::should_close() const {
    return WindowShouldClose();
}

void Renderer::render_frame(const SimState& state, const RoadNetwork& network) {
    begin_frame();
    draw_network(network);
    draw_imgui(state, network);
    end_frame();
}

// ── Private helpers ───────────────────────────────────────────────────────────

void Renderer::begin_frame() {
    BeginDrawing();
    ClearBackground(pal::bg);
    rlImGuiBegin();
}

// Helper: draw a thick line between two points with a given half-width.
// We approximate it with DrawLineEx.
static void draw_road_segment(Vector2 a, Vector2 b, float thickness, Color c) {
    DrawLineEx(a, b, thickness, c);
}

void Renderer::draw_network(const RoadNetwork& network) {
    // ── Draw roads ────────────────────────────────────────────────────────
    // We want: filled road body, edge lines, and dashed centre-lane dividers.
    // Raylib has no thick polygon primitive, so we use DrawLineEx for the body
    // (thick enough to look like a road) and DrawLine for the kerb lines.

    for (const auto& [rid, road] : network.roads()) {
        const Intersection* src = network.intersection(road.from);
        const Intersection* dst = network.intersection(road.to);
        if (!src || !dst) continue;

        const Vector2 a{ src->position.x, src->position.y };
        const Vector2 b{ dst->position.x, dst->position.y };

        // Each lane is 3.5 m → scaled to pixels; we cap visual width
        const float lane_px    = 14.f;  // visual lane width in pixels
        const float road_width = static_cast<float>(road.lane_count()) * lane_px;

        // Road surface
        draw_road_segment(a, b, road_width, pal::road_fill);

        // Kerb lines (thin, slightly offset — approximated along the segment)
        draw_road_segment(a, b, road_width + 2.f, pal::road_edge);
        // Re-draw road fill on top to get an edge effect
        draw_road_segment(a, b, road_width - 2.f, pal::road_fill);

        // Lane dividers — drawn as dots along the centre-line
        // Only draw if this is the "forward" direction (from < to) to avoid
        // drawing twice for bi-directional pairs.
        if (static_cast<uint32_t>(road.from) < static_cast<uint32_t>(road.to)) {
            const float len = std::hypot(b.x - a.x, b.y - a.y);
            if (len > 0.f && road.lane_count() > 1) {
                const float dx = (b.x - a.x) / len;
                const float dy = (b.y - a.y) / len;
                // Dash every 18 px, 9 px on, 9 px off
                constexpr float dash = 9.f;
                constexpr float gap  = 9.f;
                float t = gap;
                while (t + dash < len) {
                    Vector2 p0{ a.x + dx * t,        a.y + dy * t };
                    Vector2 p1{ a.x + dx * (t + dash), a.y + dy * (t + dash) };
                    DrawLineEx(p0, p1, 1.5f, pal::lane_divider);
                    t += dash + gap;
                }
            }
        }
    }

    // ── Draw intersections ────────────────────────────────────────────────
    constexpr float NODE_RADIUS = 10.f;
    for (const auto& [iid, node] : network.intersections()) {
        const float cx = node.position.x;
        const float cy = node.position.y;

        // Filled circle + border ring
        DrawCircle(static_cast<int>(cx), static_cast<int>(cy),
                   NODE_RADIUS,     pal::node_fill);
        DrawCircleLines(static_cast<int>(cx), static_cast<int>(cy),
                        NODE_RADIUS, pal::node_border);

        // Label
        if (!node.name.empty()) {
            const int fs = 10;
            const int tw = MeasureText(node.name.c_str(), fs);
            DrawText(node.name.c_str(),
                     static_cast<int>(cx) - tw / 2,
                     static_cast<int>(cy) + static_cast<int>(NODE_RADIUS) + 3,
                     fs, pal::node_label);
        }
    }

    // ── Status bar ────────────────────────────────────────────────────────
    const int H = GetScreenHeight();
    std::string status =
        "Intersections: " + std::to_string(network.intersection_count()) +
        "   Roads: "      + std::to_string(network.road_count());
    DrawText(status.c_str(), 16, H - 36, 18, pal::status_text);
}

void Renderer::draw_imgui(const SimState& state, const RoadNetwork& network) {
    ImGui::SetNextWindowPos ({ 10.f,  10.f }, ImGuiCond_Once);
    ImGui::SetNextWindowSize({ 300.f, 160.f }, ImGuiCond_Once);
    ImGui::Begin("SyntraQ Debug", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);

    ImGui::Text("Milestone 2 — Road Network");
    ImGui::Separator();
    ImGui::Text("Tick          : %llu",
                static_cast<unsigned long long>(state.tick));
    ImGui::Text("Elapsed       : %.2f s", state.elapsed_s);
    ImGui::Text("FPS           : %.1f",   GetFPS());
    ImGui::Separator();
    ImGui::Text("Intersections : %u", network.intersection_count());
    ImGui::Text("Roads         : %u", network.road_count());

    ImGui::End();
}

void Renderer::end_frame() {
    rlImGuiEnd();
    EndDrawing();
}

} // namespace syntraq
