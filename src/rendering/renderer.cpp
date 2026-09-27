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
    constexpr Color bg           = {  12,  14,  20, 255 };
    constexpr Color road_fill    = {  40,  44,  58, 255 };
    constexpr Color lane_divider = {  90, 100, 120, 180 };
    constexpr Color road_edge    = {  65,  72,  90, 255 };
    constexpr Color node_fill    = {  28,  32,  45, 255 };
    constexpr Color node_border  = {  80, 200, 255, 255 };
    constexpr Color node_label   = { 140, 180, 220, 200 };
    constexpr Color status_text  = { 100, 180, 100, 255 };

    // Vehicle colours — cycling palette
    constexpr Color vehicle_colors[] = {
        {  80, 220, 140, 255 },  // teal-green
        { 255, 180,  50, 255 },  // amber
        { 100, 160, 255, 255 },  // sky-blue
        { 255,  90,  90, 255 },  // coral
        { 200, 120, 255, 255 },  // violet
        {  60, 220, 220, 255 },  // cyan
        { 255, 200,  80, 255 },  // yellow
    };
    constexpr int kNumVehicleColors =
        static_cast<int>(sizeof(vehicle_colors) / sizeof(vehicle_colors[0]));
}

// ── Construction / destruction ────────────────────────────────────────────────

Renderer::Renderer(const Config& cfg)
    : cfg_(cfg)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(cfg_.window_width, cfg_.window_height,
               cfg_.window_title.c_str());
    SetTargetFPS(cfg_.target_fps);
    rlImGuiSetup(true);
}

Renderer::~Renderer() {
    rlImGuiShutdown();
    CloseWindow();
}

// ── Public ────────────────────────────────────────────────────────────────────

bool Renderer::should_close() const {
    return WindowShouldClose();
}

void Renderer::render_frame(const SimState&             state,
                             const RoadNetwork&           network,
                             const std::vector<Vehicle>&  vehicles) {
    begin_frame();
    draw_network(network);
    draw_vehicles(network, vehicles);
    draw_imgui(state, network);
    end_frame();
}

// ── Private ───────────────────────────────────────────────────────────────────

void Renderer::begin_frame() {
    BeginDrawing();
    ClearBackground(pal::bg);
    rlImGuiBegin();
}

static void draw_thick_line(Vector2 a, Vector2 b, float thickness, Color c) {
    DrawLineEx(a, b, thickness, c);
}

void Renderer::draw_network(const RoadNetwork& network) {
    // ── Roads ─────────────────────────────────────────────────────────────
    for (const auto& [rid, road] : network.roads()) {
        const Intersection* src = network.intersection(road.from);
        const Intersection* dst = network.intersection(road.to);
        if (!src || !dst) continue;

        const Vector2 a{ src->position.x, src->position.y };
        const Vector2 b{ dst->position.x, dst->position.y };

        const float lane_px  = 14.f;
        const float road_w   = static_cast<float>(road.lane_count()) * lane_px;

        draw_thick_line(a, b, road_w + 2.f, pal::road_edge);
        draw_thick_line(a, b, road_w - 2.f, pal::road_fill);

        // Dashed centre divider (only for bi-directional forward direction)
        if (static_cast<uint32_t>(road.from) < static_cast<uint32_t>(road.to)) {
            const float len = std::hypot(b.x - a.x, b.y - a.y);
            if (len > 0.f && road.lane_count() > 1) {
                const float dx = (b.x - a.x) / len;
                const float dy = (b.y - a.y) / len;
                constexpr float dash = 9.f, gap = 9.f;
                float t = gap;
                while (t + dash < len) {
                    DrawLineEx({ a.x + dx * t,          a.y + dy * t },
                               { a.x + dx * (t + dash), a.y + dy * (t + dash) },
                               1.5f, pal::lane_divider);
                    t += dash + gap;
                }
            }
        }
    }

    // ── Intersections ─────────────────────────────────────────────────────
    constexpr float R = 10.f;
    for (const auto& [iid, node] : network.intersections()) {
        const int cx = static_cast<int>(node.position.x);
        const int cy = static_cast<int>(node.position.y);
        DrawCircle(cx, cy, R, pal::node_fill);
        DrawCircleLines(cx, cy, R, pal::node_border);
        if (!node.name.empty()) {
            const int fs = 10;
            const int tw = MeasureText(node.name.c_str(), fs);
            DrawText(node.name.c_str(), cx - tw / 2,
                     cy + static_cast<int>(R) + 3, fs, pal::node_label);
        }
    }

    // Status bar
    const int H = GetScreenHeight();
    std::string status =
        "Intersections: " + std::to_string(network.intersection_count()) +
        "   Roads: "      + std::to_string(network.road_count());
    DrawText(status.c_str(), 16, H - 36, 18, pal::status_text);
}

void Renderer::draw_vehicles(const RoadNetwork&           network,
                              const std::vector<Vehicle>&  vehicles) {
    constexpr float VEHICLE_RADIUS = 6.f;
    constexpr float LANE_OFFSET    = 5.f;  // pixels perpendicular to road axis

    for (const Vehicle& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;

        const Road* road = network.road(v.current_road);
        if (!road) continue;

        const Intersection* src = network.intersection(road->from);
        const Intersection* dst = network.intersection(road->to);
        if (!src || !dst) continue;

        // ── Interpolate pixel position along road ─────────────────────────
        const float t = (road->length_m > 0.f)
                      ? v.progress_m / road->length_m
                      : 0.f;
        const float clamp_t = (t < 0.f) ? 0.f : (t > 1.f) ? 1.f : t;

        const float rx = src->position.x + (dst->position.x - src->position.x) * clamp_t;
        const float ry = src->position.y + (dst->position.y - src->position.y) * clamp_t;

        // ── Perpendicular offset per lane ─────────────────────────────────
        const float rdx = dst->position.x - src->position.x;
        const float rdy = dst->position.y - src->position.y;
        const float rlen = std::hypot(rdx, rdy);
        float px = 0.f, py = 0.f;
        if (rlen > 0.f) {
            const float perp_x = -rdy / rlen;
            const float perp_y =  rdx / rlen;
            const float sign   = (v.lane_index % 2 == 0) ? 1.f : -1.f;
            px = perp_x * LANE_OFFSET * sign;
            py = perp_y * LANE_OFFSET * sign;
        }

        const Vector2 pos{ rx + px, ry + py };

        // ── Pick colour from cycling palette ──────────────────────────────
        const Color col = pal::vehicle_colors[
            static_cast<uint32_t>(v.id) % pal::kNumVehicleColors];

        // Draw: outline for contrast, then filled circle
        DrawCircleV(pos, VEHICLE_RADIUS + 1.5f, { 10, 12, 20, 200 });
        DrawCircleV(pos, VEHICLE_RADIUS, col);

        // Speed indicator: tiny dot dimmer when slow
        const float speed_frac = (v.max_speed_mps > 0.f)
                                ? v.speed_mps / v.max_speed_mps : 0.f;
        const uint8_t brightness = static_cast<uint8_t>(55 + speed_frac * 200);
        DrawCircleV(pos, VEHICLE_RADIUS * 0.35f,
                    { brightness, brightness, brightness, 200 });
    }
}

void Renderer::draw_imgui(const SimState& state, const RoadNetwork& network) {
    ImGui::SetNextWindowPos ({ 10.f,  10.f }, ImGuiCond_Once);
    ImGui::SetNextWindowSize({ 310.f, 210.f }, ImGuiCond_Once);
    ImGui::Begin("SyntraQ Debug", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);

    ImGui::Text("Milestone 3 \xe2\x80\x94 Vehicle System");
    ImGui::Separator();
    ImGui::Text("Tick          : %llu",
                static_cast<unsigned long long>(state.tick));
    ImGui::Text("Elapsed       : %.2f s", state.elapsed_s);
    ImGui::Text("FPS           : %.1f",   GetFPS());
    ImGui::Separator();
    ImGui::Text("Intersections : %u", network.intersection_count());
    ImGui::Text("Roads         : %u", network.road_count());
    ImGui::Separator();
    ImGui::Text("Active        : %u", state.active_vehicles);
    ImGui::Text("Spawned       : %u", state.total_spawned);
    ImGui::Text("Arrived       : %u", state.total_arrived);

    ImGui::End();
}

void Renderer::end_frame() {
    rlImGuiEnd();
    EndDrawing();
}

} // namespace syntraq
