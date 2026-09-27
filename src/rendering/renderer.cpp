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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace syntraq {

// ── Internal Camera Wrapper ──────────────────────────────────────────────────

struct RenderCamera {
    Camera2D cam{};
    float    min_zoom{ 0.25f };
    float    max_zoom{ 4.00f };
};

// ── Palette ───────────────────────────────────────────────────────────────────

namespace pal {
    constexpr Color bg              = {  14,  16,  22, 255 };
    constexpr Color grid_line       = {  24,  28,  38, 160 };
    constexpr Color road_edge       = {  54,  62,  80, 255 };
    constexpr Color road_fill       = {  32,  36,  48, 255 };
    constexpr Color lane_divider    = { 120, 135, 160, 130 };
    constexpr Color lane_guide      = {  60, 180, 220, 140 };

    constexpr Color node_apron      = {  32,  36,  48, 255 };
    constexpr Color node_core       = {  22,  26,  36, 255 };
    constexpr Color node_border     = {  70, 140, 200, 200 };
    constexpr Color node_label_bg   = {  16,  20,  28, 220 };
    constexpr Color node_label_txt  = { 180, 210, 240, 240 };

    constexpr Color status_bg       = {  12,  14,  20, 200 };
    constexpr Color status_border   = {  36,  42,  56, 255 };
    constexpr Color status_text     = { 140, 170, 195, 230 };

    // Vehicle colours — cycling palette for distinct visibility
    constexpr Color vehicle_colors[] = {
        {  75, 215, 145, 255 },  // emerald green
        { 255, 180,  55, 255 },  // warm amber
        {  90, 165, 255, 255 },  // vivid sky blue
        { 255,  95,  95, 255 },  // soft coral
        { 195, 125, 255, 255 },  // lavender violet
        {  55, 215, 215, 255 },  // cyan
        { 250, 210,  70, 255 },  // golden yellow
    };
    constexpr int kNumVehicleColors =
        static_cast<int>(sizeof(vehicle_colors) / sizeof(vehicle_colors[0]));
}

// ── Construction / destruction ────────────────────────────────────────────────

Renderer::Renderer(const Config& cfg)
    : cfg_(cfg)
    , camera_(std::make_unique<RenderCamera>())
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(cfg_.window_width, cfg_.window_height,
               cfg_.window_title.c_str());
    SetTargetFPS(cfg_.target_fps);

    camera_->cam.target   = { 0.0f, 0.0f };
    camera_->cam.offset   = { static_cast<float>(cfg_.window_width) * 0.42f,
                              static_cast<float>(cfg_.window_height) * 0.50f };
    camera_->cam.rotation = 0.0f;
    camera_->cam.zoom     = 1.0f;

    rlImGuiSetup(true);

    // Apply dark technical styling to Dear ImGui
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 6.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.GrabRounding      = 4.0f;
    style.WindowBorderSize  = 1.0f;
    style.FrameBorderSize   = 1.0f;
    style.ItemSpacing       = ImVec2(8.0f, 5.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]             = ImVec4(0.08f, 0.10f, 0.14f, 0.94f);
    colors[ImGuiCol_Border]               = ImVec4(0.20f, 0.25f, 0.35f, 0.65f);
    colors[ImGuiCol_FrameBg]              = ImVec4(0.12f, 0.15f, 0.22f, 0.85f);
    colors[ImGuiCol_FrameBgHovered]       = ImVec4(0.18f, 0.24f, 0.34f, 0.85f);
    colors[ImGuiCol_FrameBgActive]        = ImVec4(0.22f, 0.30f, 0.42f, 0.85f);
    colors[ImGuiCol_TitleBg]              = ImVec4(0.10f, 0.13f, 0.18f, 1.00f);
    colors[ImGuiCol_TitleBgActive]        = ImVec4(0.14f, 0.18f, 0.26f, 1.00f);
    colors[ImGuiCol_Button]               = ImVec4(0.16f, 0.22f, 0.32f, 0.90f);
    colors[ImGuiCol_ButtonHovered]        = ImVec4(0.24f, 0.32f, 0.46f, 1.00f);
    colors[ImGuiCol_ButtonActive]         = ImVec4(0.30f, 0.40f, 0.58f, 1.00f);
    colors[ImGuiCol_Header]               = ImVec4(0.16f, 0.22f, 0.32f, 0.70f);
    colors[ImGuiCol_HeaderHovered]        = ImVec4(0.22f, 0.30f, 0.44f, 0.80f);
    colors[ImGuiCol_HeaderActive]         = ImVec4(0.26f, 0.36f, 0.52f, 1.00f);
    colors[ImGuiCol_CheckMark]            = ImVec4(0.35f, 0.75f, 0.95f, 1.00f);
}

Renderer::~Renderer() {
    rlImGuiShutdown();
    CloseWindow();
}

// ── Public ────────────────────────────────────────────────────────────────────

bool Renderer::should_close() const {
    return WindowShouldClose();
}

float Renderer::frame_time() const {
    return GetFrameTime();
}

void Renderer::reset_camera(const RoadNetwork& network) {
    if (network.intersection_count() == 0) return;

    float min_x =  1e9f, min_y =  1e9f;
    float max_x = -1e9f, max_y = -1e9f;

    for (const auto& [id, node] : network.intersections()) {
        min_x = std::min(min_x, node.position.x);
        min_y = std::min(min_y, node.position.y);
        max_x = std::max(max_x, node.position.x);
        max_y = std::max(max_y, node.position.y);
    }

    const float center_x = (min_x + max_x) * 0.5f;
    const float center_y = (min_y + max_y) * 0.5f;
    const float net_w    = (max_x - min_x) + 100.0f;
    const float net_h    = (max_y - min_y) + 100.0f;

    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    camera_->cam.target   = { center_x, center_y };
    // Bias offset slightly left to balance the right-hand ImGui panel
    camera_->cam.offset   = { screen_w * 0.42f, screen_h * 0.50f };
    camera_->cam.rotation = 0.0f;

    const float zoom_x = (screen_w * 0.65f) / std::max(net_w, 1.0f);
    const float zoom_y = (screen_h * 0.78f) / std::max(net_h, 1.0f);
    camera_->cam.zoom  = std::clamp(std::min(zoom_x, zoom_y),
                                    camera_->min_zoom,
                                    camera_->max_zoom);
}

void Renderer::take_screenshot(const std::string& path) const {
    TakeScreenshot(path.c_str());
}

SimControlAction Renderer::render_frame(const SimState&             state,
                                        const RoadNetwork&          network,
                                        const std::vector<Vehicle>& vehicles,
                                        float                       current_speed_scale) {
    update_camera(network);

    begin_frame();

    // ── 2D World Space Rendering ───────────────────────────────────────────
    BeginMode2D(camera_->cam);
    draw_world_grid();
    draw_network(network);
    draw_vehicles(network, vehicles);
    EndMode2D();

    // ── Screen Space HUD & UI Overlays ─────────────────────────────────────
    draw_status_bar(state, current_speed_scale);
    SimControlAction action = draw_imgui(state, network, current_speed_scale);

    end_frame();

    // Keyboard shortcuts (only when not interacting with text input)
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        if (IsKeyPressed(KEY_SPACE)) {
            action.request_pause_toggle = true;
        }
        if (IsKeyPressed(KEY_R)) {
            reset_camera(network);
        }
    }

    return action;
}

// ── Private ───────────────────────────────────────────────────────────────────

void Renderer::update_camera(const RoadNetwork& network) {
    if (!camera_initialized_ && network.intersection_count() > 0) {
        reset_camera(network);
        camera_initialized_ = true;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    // 1. Mouse wheel zoom towards mouse cursor (when mouse is not over ImGui)
    if (!io.WantCaptureMouse) {
        const float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            const Vector2 mouse_screen = GetMousePosition();
            const Vector2 mouse_world  = GetScreenToWorld2D(mouse_screen, camera_->cam);

            camera_->cam.offset = mouse_screen;
            camera_->cam.target = mouse_world;

            float zoom_factor = 1.0f + (0.15f * std::abs(wheel));
            if (wheel < 0.0f) zoom_factor = 1.0f / zoom_factor;

            camera_->cam.zoom = std::clamp(camera_->cam.zoom * zoom_factor,
                                           camera_->min_zoom,
                                           camera_->max_zoom);
        }

        // 2. Pan with middle or right mouse drag
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            const Vector2 delta = GetMouseDelta();
            camera_->cam.target.x -= delta.x / camera_->cam.zoom;
            camera_->cam.target.y -= delta.y / camera_->cam.zoom;
        }
    }

    // 3. Keep viewport center aligned if window resizes while keeping relative target
    if (IsWindowResized()) {
        camera_->cam.offset = { screen_w * 0.42f, screen_h * 0.50f };
    }
}

void Renderer::begin_frame() {
    BeginDrawing();
    ClearBackground(pal::bg);
    rlImGuiBegin();
}

void Renderer::draw_world_grid() {
    // Subtle background coordinate grid in world space
    constexpr float kGridSpacing = 100.0f;
    constexpr float kExtentMin   = -300.0f;
    constexpr float kExtentMax   = 1200.0f;

    for (float x = kExtentMin; x <= kExtentMax; x += kGridSpacing) {
        DrawLineV({ x, kExtentMin }, { x, kExtentMax }, pal::grid_line);
    }
    for (float y = kExtentMin; y <= kExtentMax; y += kGridSpacing) {
        DrawLineV({ kExtentMin, y }, { kExtentMax, y }, pal::grid_line);
    }
}

void Renderer::draw_network(const RoadNetwork& network) {
    constexpr float kLaneWidthPx = 14.0f;

    // ── 1. Roads ───────────────────────────────────────────────────────────
    for (const auto& [rid, road] : network.roads()) {
        const Intersection* src = network.intersection(road.from);
        const Intersection* dst = network.intersection(road.to);
        if (!src || !dst) continue;

        const Vector2 a{ src->position.x, src->position.y };
        const Vector2 b{ dst->position.x, dst->position.y };

        const float road_w = static_cast<float>(road.lane_count()) * kLaneWidthPx;

        // Outer curb line
        DrawLineEx(a, b, road_w + 3.0f, pal::road_edge);
        // Main asphalt surface
        DrawLineEx(a, b, road_w - 1.0f, pal::road_fill);

        // Dashed centre divider (draw once per bidirectional pair)
        if (static_cast<uint32_t>(road.from) < static_cast<uint32_t>(road.to)) {
            const float len = std::hypot(b.x - a.x, b.y - a.y);
            if (len > 0.0f) {
                const float dx = (b.x - a.x) / len;
                const float dy = (b.y - a.y) / len;
                constexpr float dash = 8.0f;
                constexpr float gap  = 8.0f;
                float t = gap;
                while (t + dash < len) {
                    DrawLineEx({ a.x + dx * t,          a.y + dy * t },
                               { a.x + dx * (t + dash), a.y + dy * (t + dash) },
                               1.5f, pal::lane_divider);
                    t += dash + gap;
                }
            }
        }

        // Debug: Show lane boundaries
        if (debug_flags_.show_lane_boundaries) {
            const float len = std::hypot(b.x - a.x, b.y - a.y);
            if (len > 0.0f) {
                const float perp_x = -(b.y - a.y) / len;
                const float perp_y =  (b.x - a.x) / len;
                const float half_w = road_w * 0.5f;
                DrawLineEx({ a.x + perp_x * half_w, a.y + perp_y * half_w },
                           { b.x + perp_x * half_w, b.y + perp_y * half_w },
                           1.0f, pal::lane_guide);
                DrawLineEx({ a.x - perp_x * half_w, a.y - perp_y * half_w },
                           { b.x - perp_x * half_w, b.y - perp_y * half_w },
                           1.0f, pal::lane_guide);
            }
        }

        // Debug: Show road IDs (disabled by default)
        if (debug_flags_.show_road_ids) {
            const float mid_x = (a.x + b.x) * 0.5f;
            const float mid_y = (a.y + b.y) * 0.5f;
            std::string road_label = "R" + std::to_string(static_cast<uint32_t>(rid));
            const int fs = 10;
            const int tw = MeasureText(road_label.c_str(), fs);
            DrawRectangle(static_cast<int>(mid_x - tw * 0.5f - 2),
                          static_cast<int>(mid_y - 6),
                          tw + 4, 12, { 18, 22, 30, 220 });
            DrawText(road_label.c_str(),
                     static_cast<int>(mid_x - tw * 0.5f),
                     static_cast<int>(mid_y - 5),
                     fs, { 200, 215, 140, 240 });
        }
    }

    // ── 2. Intersections ───────────────────────────────────────────────────
    constexpr float kIntersectionRadius = 15.0f; // Blends seamlessly with road half-width (14px)
    for (const auto& [iid, node] : network.intersections()) {
        const float cx = node.position.x;
        const float cy = node.position.y;

        // Junction apron (blends with road asphalt, preventing protruding edges)
        DrawCircleV({ cx, cy }, kIntersectionRadius, pal::node_apron);
        DrawCircleLinesV({ cx, cy }, kIntersectionRadius, pal::road_edge);

        // Technical junction core disc
        DrawCircleV({ cx, cy }, 5.5f, pal::node_core);
        DrawCircleLinesV({ cx, cy }, 5.5f, pal::node_border);

        // Debug: Show intersection IDs (OFF by default)
        if (debug_flags_.show_intersection_ids) {
            const std::string label = node.name.empty()
                ? "I" + std::to_string(static_cast<uint32_t>(iid))
                : node.name;
            const int fs = 10;
            const int tw = MeasureText(label.c_str(), fs);
            const int bx = static_cast<int>(cx - tw * 0.5f - 3);
            const int by = static_cast<int>(cy + kIntersectionRadius + 3);
            DrawRectangle(bx, by, tw + 6, 13, pal::node_label_bg);
            DrawRectangleLines(bx, by, tw + 6, 13, { 60, 140, 200, 180 });
            DrawText(label.c_str(), bx + 3, by + 1, fs, pal::node_label_txt);
        }
    }
}

void Renderer::draw_vehicles(const RoadNetwork&          network,
                            const std::vector<Vehicle>& vehicles) {
    constexpr float kCarLength  = 13.0f;
    constexpr float kCarWidth   = 6.6f;
    constexpr float kLaneOffset = 5.0f;

    for (const Vehicle& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;

        const Road* road = network.road(v.current_road);
        if (!road) continue;

        const Intersection* src = network.intersection(road->from);
        const Intersection* dst = network.intersection(road->to);
        if (!src || !dst) continue;

        const float t = (road->length_m > 0.0f) ? (v.progress_m / road->length_m) : 0.0f;
        const float clamp_t = std::clamp(t, 0.0f, 1.0f);

        const float rdx = dst->position.x - src->position.x;
        const float rdy = dst->position.y - src->position.y;
        const float rlen = std::hypot(rdx, rdy);
        if (rlen <= 0.0001f) continue;

        const float udx = rdx / rlen;
        const float udy = rdy / rlen;

        // Normal vector to the right of travel direction
        const float perp_x = -udy;
        const float perp_y =  udx;

        // Lane offset (lane 0 to the right, lane 1 to the left)
        const float sign = (v.lane_index % 2 == 0) ? 1.0f : -1.0f;
        const float px = perp_x * kLaneOffset * sign;
        const float py = perp_y * kLaneOffset * sign;

        const float rx = src->position.x + rdx * clamp_t;
        const float ry = src->position.y + rdy * clamp_t;
        const Vector2 pos{ rx + px, ry + py };

        const float angle_deg = std::atan2(udy, udx) * (180.0f / 3.1415926535f);

        const Color car_col = pal::vehicle_colors[
            static_cast<uint32_t>(v.id) % pal::kNumVehicleColors];

        // 1. Drop shadow / dark outline
        const Rectangle shadow_rec{ pos.x, pos.y, kCarLength + 1.6f, kCarWidth + 1.6f };
        DrawRectanglePro(shadow_rec,
                         { (kCarLength + 1.6f) * 0.5f, (kCarWidth + 1.6f) * 0.5f },
                         angle_deg,
                         { 8, 10, 14, 180 });

        // 2. Main chassis body
        const Rectangle body_rec{ pos.x, pos.y, kCarLength, kCarWidth };
        DrawRectanglePro(body_rec,
                         { kCarLength * 0.5f, kCarWidth * 0.5f },
                         angle_deg,
                         car_col);

        // 3. Cabin / Roof (dark tinted glass, slightly set back from center)
        const float cabin_len = kCarLength * 0.46f;
        const float cabin_wid = kCarWidth * 0.72f;
        const Vector2 cabin_pos{ pos.x - udx * 0.6f, pos.y - udy * 0.6f };
        const Rectangle cabin_rec{ cabin_pos.x, cabin_pos.y, cabin_len, cabin_wid };
        DrawRectanglePro(cabin_rec,
                         { cabin_len * 0.5f, cabin_wid * 0.5f },
                         angle_deg,
                         { 18, 22, 32, 230 });

        // 4. Windshield subtle glint (front edge of cabin)
        const Vector2 glint_pos{ cabin_pos.x + udx * (cabin_len * 0.38f),
                                cabin_pos.y + udy * (cabin_len * 0.38f) };
        const Rectangle glint_rec{ glint_pos.x, glint_pos.y, 1.2f, cabin_wid * 0.75f };
        DrawRectanglePro(glint_rec,
                         { 0.6f, cabin_wid * 0.375f },
                         angle_deg,
                         { 180, 215, 245, 180 });

        // 5. Front headlights (warm white dots at front bumper)
        const float fwd_dist  = kCarLength * 0.5f - 0.5f;
        const float side_dist = kCarWidth * 0.5f - 1.2f;
        const Vector2 nose{ pos.x + udx * fwd_dist, pos.y + udy * fwd_dist };
        DrawCircleV({ nose.x - perp_x * side_dist, nose.y - perp_y * side_dist }, 0.9f, { 255, 255, 210, 240 });
        DrawCircleV({ nose.x + perp_x * side_dist, nose.y + perp_y * side_dist }, 0.9f, { 255, 255, 210, 240 });

        // 6. Taillights (red accents at rear bumper; bright red if stopped)
        const float rear_dist = kCarLength * 0.5f - 0.5f;
        const Vector2 tail{ pos.x - udx * rear_dist, pos.y - udy * rear_dist };
        const Color tail_col = (v.speed_mps < 0.5f || v.state == VehicleState::Stopped)
                             ? Color{ 255, 40, 40, 255 }
                             : Color{ 200, 30, 30, 210 };
        DrawCircleV({ tail.x - perp_x * side_dist, tail.y - perp_y * side_dist }, 0.85f, tail_col);
        DrawCircleV({ tail.x + perp_x * side_dist, tail.y + perp_y * side_dist }, 0.85f, tail_col);

        // 7. Debug: Vehicle direction vectors
        if (debug_flags_.show_vehicle_vectors) {
            const Vector2 tip{ nose.x + udx * 12.0f, nose.y + udy * 12.0f };
            DrawLineEx(nose, tip, 1.5f, { 80, 220, 255, 220 });
            DrawCircleV(tip, 1.8f, { 80, 220, 255, 255 });
        }
    }
}

void Renderer::draw_status_bar(const SimState& state, float current_speed_scale) {
    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    // Bottom translucent status bar
    DrawRectangle(0, static_cast<int>(screen_h - 26.0f), static_cast<int>(screen_w), 26, pal::status_bg);
    DrawLineEx({ 0.0f, screen_h - 26.0f }, { screen_w, screen_h - 26.0f }, 1.0f, pal::status_border);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "SyntraQ | State: %s (%.2fx) | Sim: %.1f s (Tick %llu) | Zoom: %.2fx | Pan: Wheel=Zoom, M/R-Drag=Pan, R=Reset, Space=Pause",
                  state.running ? "RUNNING" : "PAUSED",
                  current_speed_scale,
                  state.elapsed_s,
                  static_cast<unsigned long long>(state.tick),
                  camera_->cam.zoom);

    DrawText(buf, 12, static_cast<int>(screen_h - 19.0f), 12, pal::status_text);
}

SimControlAction Renderer::draw_imgui(const SimState&    state,
                                      const RoadNetwork& network,
                                      float              current_speed_scale) {
    SimControlAction action;

    const float screen_w = static_cast<float>(GetScreenWidth());
    ImGui::SetNextWindowPos({ screen_w - 330.0f, 16.0f }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({ 314.0f, 0.0f }, ImGuiCond_FirstUseEver);

    ImGui::Begin("SyntraQ Control Center", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);

    // ── 1. Simulation Controls ─────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "SIMULATION CONTROLS");
    ImGui::Separator();

    // Pause / Resume toggle button
    if (state.running) {
        if (ImGui::Button("Pause (Space)", ImVec2(100.0f, 26.0f))) {
            action.request_pause_toggle = true;
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.50f, 0.30f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.60f, 0.35f, 1.00f));
        if (ImGui::Button("Resume (Space)", ImVec2(100.0f, 26.0f))) {
            action.request_pause_toggle = true;
        }
        ImGui::PopStyleColor(2);
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset Sim", ImVec2(90.0f, 26.0f))) {
        action.request_reset = true;
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset View", ImVec2(90.0f, 26.0f))) {
        reset_camera(network);
    }

    // Speed multiplier buttons
    ImGui::Spacing();
    ImGui::Text("Speed Multiplier:");
    const float speed_presets[] = { 0.25f, 0.50f, 1.00f, 2.00f, 5.00f };
    const char* speed_labels[]  = { "0.25x", "0.5x", "1x", "2x", "5x" };

    for (int i = 0; i < 5; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (std::abs(current_speed_scale - speed_presets[i]) < 0.01f);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.80f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.65f, 0.90f, 1.00f));
        }

        if (ImGui::Button(speed_labels[i], ImVec2(52.0f, 22.0f))) {
            action.request_speed_scale = speed_presets[i];
        }

        if (is_active) {
            ImGui::PopStyleColor(2);
        }
    }

    // Status indicator line
    ImGui::Spacing();
    if (state.running) {
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.45f, 1.00f), "Status: RUNNING (%.2fx)", current_speed_scale);
    } else {
        ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.00f), "Status: PAUSED");
    }

    // ── 2. Debug Statistics ────────────────────────────────────────────────
    if (debug_flags_.show_debug_statistics) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "SIMULATION METRICS");
        ImGui::Separator();

        if (ImGui::BeginTable("SimStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Tick");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%llu", static_cast<unsigned long long>(state.tick));

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Sim Elapsed");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f s", state.elapsed_s);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Timestep (dt)");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f s", cfg_.dt_seconds);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Render FPS");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%d", GetFPS());

            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "WORLD TOPOLOGY");
        ImGui::Separator();

        if (ImGui::BeginTable("WorldStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Intersections");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%u", network.intersection_count());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Road Segments");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%u", network.road_count());

            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "VEHICLE SYSTEM");
        ImGui::Separator();

        if (ImGui::BeginTable("VehStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Active");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.active_vehicles);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Total Spawned");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.total_spawned);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Total Arrived");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.total_arrived);

            ImGui::EndTable();
        }
    }

    // ── 3. View & Debug Toggles ────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "VIEW & DEBUG TOGGLES");
    ImGui::Separator();

    ImGui::Checkbox("Intersection IDs",     &debug_flags_.show_intersection_ids);
    ImGui::Checkbox("Road IDs",             &debug_flags_.show_road_ids);
    ImGui::Checkbox("Lane Boundaries",      &debug_flags_.show_lane_boundaries);
    ImGui::Checkbox("Vehicle Vectors",      &debug_flags_.show_vehicle_vectors);
    ImGui::Checkbox("Show Debug Statistics", &debug_flags_.show_debug_statistics);

    ImGui::End();

    return action;
}

void Renderer::end_frame() {
    rlImGuiEnd();
    EndDrawing();
}

} // namespace syntraq
