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
    camera_->cam.offset   = { static_cast<float>(cfg_.window_width) * 0.50f,
                              static_cast<float>(cfg_.window_height - 26) * 0.50f };
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
    camera_->cam.offset   = { screen_w * 0.50f, (screen_h - 26.0f) * 0.50f };
    camera_->cam.rotation = 0.0f;

    // Viewport width between left and right side panels (~56% of screen)
    const float zoom_x = (screen_w * 0.56f) / std::max(net_w, 1.0f);
    const float zoom_y = ((screen_h - 26.0f) * 0.82f) / std::max(net_h, 1.0f);
    camera_->cam.zoom  = std::clamp(std::min(zoom_x, zoom_y),
                                    camera_->min_zoom,
                                    camera_->max_zoom);
}

void Renderer::take_screenshot(const std::string& path) const {
    TakeScreenshot(path.c_str());
}

SimControlAction Renderer::render_frame(
    const SimState&                                                    state,
    const RoadNetwork&                                                 network,
    const std::vector<Vehicle>&                                        vehicles,
    float                                                              current_speed_scale,
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers) {
    update_camera(network);

    begin_frame();

    // ── 2D World Space Rendering ───────────────────────────────────────────
    BeginMode2D(camera_->cam);
    draw_world_grid();
    draw_network(network);
    if (debug_flags_.show_traffic_signals && signal_controllers) {
        draw_traffic_signals(network, *signal_controllers);
    }
    draw_vehicles(network, vehicles);
    EndMode2D();

    // ── Screen Space HUD & UI Overlays ─────────────────────────────────────
    draw_status_bar(state, current_speed_scale);
    SimControlAction action = draw_imgui(state, network, current_speed_scale, signal_controllers);

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
        camera_->cam.offset = { screen_w * 0.50f, (screen_h - 26.0f) * 0.50f };
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

void Renderer::draw_traffic_signals(
    const RoadNetwork&                                                 network,
    const std::unordered_map<IntersectionId, TrafficSignalController>& controllers) {

    constexpr float kLaneWidthPx        = 14.0f;
    constexpr float kStopDistFromCenter = 26.0f;

    for (const auto& [inter_id, controller] : controllers) {
        const Intersection* center = network.intersection(inter_id);
        if (!center) continue;

        const Vector2 center_pos{ center->position.x, center->position.y };

        for (const auto& [road_id, light] : controller.lights()) {
            const Road* road = network.road(road_id);
            if (!road) continue;

            const Intersection* from_node = network.intersection(road->from);
            if (!from_node) continue;

            const float dx = center_pos.x - from_node->position.x;
            const float dy = center_pos.y - from_node->position.y;
            const float len = std::hypot(dx, dy);
            if (len <= 0.001f) continue;

            const float udx = dx / len;
            const float udy = dy / len;
            const float perp_x = -udy;
            const float perp_y =  udx;

            const float road_w = static_cast<float>(road->lane_count()) * kLaneWidthPx;

            // 1. Draw stop line across incoming traffic lane(s)
            const Vector2 stop_bar_mid{
                center_pos.x - udx * kStopDistFromCenter + perp_x * (road_w * 0.25f),
                center_pos.y - udy * kStopDistFromCenter + perp_y * (road_w * 0.25f)
            };
            const Vector2 stop_bar_p1{
                stop_bar_mid.x - perp_x * (road_w * 0.45f),
                stop_bar_mid.y - perp_y * (road_w * 0.45f)
            };
            const Vector2 stop_bar_p2{
                stop_bar_mid.x + perp_x * (road_w * 0.45f),
                stop_bar_mid.y + perp_y * (road_w * 0.45f)
            };
            DrawLineEx(stop_bar_p1, stop_bar_p2, 2.5f, { 240, 242, 248, 220 });

            // 2. Traffic light housing box mounted on the right curb
            const float housing_side_offset = road_w * 0.5f + 7.5f;
            const Vector2 box_pos{
                center_pos.x - udx * (kStopDistFromCenter - 2.0f) + perp_x * housing_side_offset,
                center_pos.y - udy * (kStopDistFromCenter - 2.0f) + perp_y * housing_side_offset
            };

            constexpr float kBoxW = 8.0f;
            constexpr float kBoxH = 20.0f;
            const Rectangle housing_rec{
                box_pos.x - kBoxW * 0.5f,
                box_pos.y - kBoxH * 0.5f,
                kBoxW,
                kBoxH
            };

            DrawRectangleRounded(housing_rec, 0.35f, 4, { 18, 22, 28, 245 });
            DrawRectangleRoundedLines(housing_rec, 0.35f, 4, { 55, 65, 80, 255 });

            // 3. Three lamp lenses: Red, Yellow, Green
            const Vector2 red_pos   { box_pos.x, box_pos.y - 5.5f };
            const Vector2 yellow_pos{ box_pos.x, box_pos.y };
            const Vector2 green_pos { box_pos.x, box_pos.y + 5.5f };

            constexpr float kLensR = 2.2f;

            // Red lens
            if (light.color == SignalColor::Red) {
                DrawCircleV(red_pos, 4.8f, { 255, 50, 50, 75 });
                DrawCircleV(red_pos, kLensR, { 255, 45, 45, 255 });
                DrawCircleV({ red_pos.x - 0.6f, red_pos.y - 0.6f }, 0.7f, { 255, 200, 200, 220 });
            } else {
                DrawCircleV(red_pos, kLensR, { 65, 20, 20, 240 });
            }

            // Yellow lens
            if (light.color == SignalColor::Yellow) {
                DrawCircleV(yellow_pos, 4.8f, { 255, 200, 30, 80 });
                DrawCircleV(yellow_pos, kLensR, { 255, 205, 35, 255 });
                DrawCircleV({ yellow_pos.x - 0.6f, yellow_pos.y - 0.6f }, 0.7f, { 255, 250, 200, 220 });
            } else {
                DrawCircleV(yellow_pos, kLensR, { 65, 52, 15, 240 });
            }

            // Green lens
            if (light.color == SignalColor::Green) {
                DrawCircleV(green_pos, 4.8f, { 45, 245, 105, 80 });
                DrawCircleV(green_pos, kLensR, { 45, 245, 105, 255 });
                DrawCircleV({ green_pos.x - 0.6f, green_pos.y - 0.6f }, 0.7f, { 210, 255, 225, 220 });
            } else {
                DrawCircleV(green_pos, kLensR, { 15, 58, 25, 240 });
            }
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

void Renderer::draw_simulation_controls(
    const SimState&    state,
    const RoadNetwork& network,
    float              current_speed_scale,
    SimControlAction&  action) {
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "SIMULATION CONTROLS");
    ImGui::Separator();

    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;

    // Responsive 3-button row (or wrap if narrow)
    const char* pause_label = state.running
        ? (avail_w >= 300.0f ? "Pause (Space)" : "Pause")
        : (avail_w >= 300.0f ? "Resume (Space)" : "Resume");

    if (avail_w >= 240.0f) {
        const float btn_w = (avail_w - 2.0f * spacing) / 3.0f;
        if (state.running) {
            if (ImGui::Button(pause_label, ImVec2(btn_w, 26.0f))) {
                action.request_pause_toggle = true;
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.50f, 0.30f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.60f, 0.35f, 1.00f));
            if (ImGui::Button(pause_label, ImVec2(btn_w, 26.0f))) {
                action.request_pause_toggle = true;
            }
            ImGui::PopStyleColor(2);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Shortcut: Space bar");
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset Sim", ImVec2(btn_w, 26.0f))) {
            action.request_reset = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset View", ImVec2(btn_w, 26.0f))) {
            reset_camera(network);
        }
    } else {
        if (state.running) {
            if (ImGui::Button("Pause (Space)", ImVec2(avail_w, 26.0f))) {
                action.request_pause_toggle = true;
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.50f, 0.30f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.60f, 0.35f, 1.00f));
            if (ImGui::Button("Resume (Space)", ImVec2(avail_w, 26.0f))) {
                action.request_pause_toggle = true;
            }
            ImGui::PopStyleColor(2);
        }

        const float half_w = (avail_w - spacing) * 0.5f;
        if (ImGui::Button("Reset Sim", ImVec2(half_w, 26.0f))) {
            action.request_reset = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset View", ImVec2(half_w, 26.0f))) {
            reset_camera(network);
        }
    }

    // Speed multiplier buttons
    ImGui::Spacing();
    ImGui::Text("Speed Multiplier:");
    const float speed_presets[] = { 0.25f, 0.50f, 1.00f, 2.00f, 5.00f };
    const char* speed_labels[]  = { "0.25x", "0.5x", "1x", "2x", "5x" };
    const float speed_btn_w     = (avail_w - 4.0f * spacing) / 5.0f;

    for (int i = 0; i < 5; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (std::abs(current_speed_scale - speed_presets[i]) < 0.01f);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.80f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.65f, 0.90f, 1.00f));
        }

        if (ImGui::Button(speed_labels[i], ImVec2(speed_btn_w, 22.0f))) {
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
}

void Renderer::draw_simulation_metrics(const SimState& state) {
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
}

void Renderer::draw_debug_controls() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "VIEW / DEBUG");
    ImGui::Separator();

    ImGui::Checkbox("Intersection IDs",      &debug_flags_.show_intersection_ids);
    ImGui::Checkbox("Road IDs",              &debug_flags_.show_road_ids);
    ImGui::Checkbox("Lane Boundaries",       &debug_flags_.show_lane_boundaries);
    ImGui::Checkbox("Vehicle Vectors",       &debug_flags_.show_vehicle_vectors);
    ImGui::Checkbox("Traffic Signals",       &debug_flags_.show_traffic_signals);
    ImGui::Checkbox("Debug Statistics",      &debug_flags_.show_debug_statistics);
}

void Renderer::draw_simulation_panel(
    const SimState&    state,
    const RoadNetwork& network,
    float              current_speed_scale,
    SimControlAction&  action) {
    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    const float panel_w     = std::clamp(screen_w * 0.20f, 280.0f, 330.0f);
    const float margin_x    = (screen_w < 1000.0f) ? 12.0f : 20.0f;
    const float min_panel_w = 260.0f;
    const float max_panel_w = std::min(420.0f, screen_w * 0.35f);
    const float max_panel_h = std::max(200.0f, screen_h - 50.0f);

    const float init_y = std::max(20.0f, (screen_h - 26.0f - 520.0f) * 0.5f);

    ImGui::SetNextWindowPos({ margin_x, init_y }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({ panel_w, 0.0f }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(min_panel_w, 150.0f),
        ImVec2(max_panel_w, max_panel_h));

    if (ImGui::Begin("SyntraQ Simulation")) {
        ImVec2 pos = ImGui::GetWindowPos();
        if (pos.x < 0.0f) {
            ImGui::SetWindowPos(ImVec2(margin_x, pos.y));
        }

        draw_simulation_controls(state, network, current_speed_scale, action);
        if (debug_flags_.show_debug_statistics) {
            draw_simulation_metrics(state);
        }
        draw_debug_controls();
    }
    ImGui::End();
}

void Renderer::draw_signal_status(
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers) {
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "TRAFFIC SIGNALS (BASELINE)");
    ImGui::Separator();

    if (!signal_controllers || signal_controllers->empty()) {
        ImGui::TextDisabled("No active signal controllers.");
        return;
    }

    std::vector<std::pair<IntersectionId, const TrafficSignalController*>> sorted;
    sorted.reserve(signal_controllers->size());
    for (const auto& [id, controller] : *signal_controllers) {
        sorted.emplace_back(id, &controller);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    if (ImGui::BeginTable("SignalControllersTable", 3, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Junction");
        ImGui::TableSetColumnIndex(1); ImGui::Text("Signal State");
        ImGui::TableSetColumnIndex(2); ImGui::Text("Remaining");

        for (const auto& [inter_id, controller] : sorted) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("I-%u", static_cast<uint32_t>(inter_id));

            ImGui::TableSetColumnIndex(1);
            switch (controller->current_stage()) {
                case PhaseStage::Green:
                    ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "GREEN");
                    break;
                case PhaseStage::Yellow:
                    ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.25f, 1.00f), "YELLOW");
                    break;
                case PhaseStage::AllRed:
                    ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.00f), "RED");
                    break;
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.1f s", controller->stage_remaining_s());
        }
        ImGui::EndTable();
    }
}

void Renderer::draw_traffic_information(
    const SimState&    state,
    const RoadNetwork& network,
    SimControlAction&  action) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "TRAFFIC SCENARIO PRESETS");
    ImGui::Separator();

    const char* scenario_names[] = { "Low Traffic", "Medium Traffic", "High Traffic", "Rush Hour" };
    const TrafficScenario scenarios[] = {
        TrafficScenario::Low,
        TrafficScenario::Medium,
        TrafficScenario::High,
        TrafficScenario::RushHour
    };
    const char* scenario_labels[] = { "Low", "Med", "High", "Rush" };

    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float btn_w   = (avail_w - 3.0f * spacing) / 4.0f;

    for (int i = 0; i < 4; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (state.scenario == scenarios[i]);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.60f, 0.80f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.70f, 0.90f, 1.00f));
        }

        if (ImGui::Button(scenario_labels[i], ImVec2(btn_w, 24.0f))) {
            action.request_scenario = scenarios[i];
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Activate %s preset", scenario_names[i]);
        }

        if (is_active) {
            ImGui::PopStyleColor(2);
        }
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
        ImGui::TableSetColumnIndex(0); ImGui::Text("Queued Vehicles");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.queued_vehicles);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Max Queue (Lane)");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.max_queue_len);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Total Spawned");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.total_spawned);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Total Arrived");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.total_arrived);

        ImGui::EndTable();
    }
}

void Renderer::draw_traffic_panel(
    const SimState&                                                    state,
    const RoadNetwork&                                                 network,
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers,
    SimControlAction&                                                  action) {
    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    const float panel_w     = std::clamp(screen_w * 0.20f, 280.0f, 330.0f);
    const float margin_x    = (screen_w < 1000.0f) ? 12.0f : 20.0f;
    const float min_panel_w = 260.0f;
    const float max_panel_w = std::min(420.0f, screen_w * 0.35f);
    const float max_panel_h = std::max(200.0f, screen_h - 50.0f);

    const float init_x = std::max(margin_x + panel_w + 40.0f, screen_w - panel_w - margin_x);
    const float init_y = std::max(20.0f, (screen_h - 26.0f - 560.0f) * 0.5f);

    ImGui::SetNextWindowPos({ init_x, init_y }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({ panel_w, 560.0f }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(min_panel_w, 150.0f),
        ImVec2(max_panel_w, max_panel_h));

    if (ImGui::Begin("SyntraQ Traffic")) {
        ImVec2 pos  = ImGui::GetWindowPos();
        ImVec2 size = ImGui::GetWindowSize();
        if (pos.x + size.x > screen_w - 5.0f) {
            float new_x = std::max(margin_x, screen_w - size.x - margin_x);
            ImGui::SetWindowPos(ImVec2(new_x, pos.y));
        }

        draw_signal_status(signal_controllers);
        draw_traffic_information(state, network, action);
    }
    ImGui::End();
}

SimControlAction Renderer::draw_imgui(
    const SimState&                                                    state,
    const RoadNetwork&                                                 network,
    float                                                              current_speed_scale,
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers) {
    SimControlAction action;

    draw_simulation_panel(state, network, current_speed_scale, action);
    draw_traffic_panel(state, network, signal_controllers, action);

    return action;
}

void Renderer::end_frame() {
    rlImGuiEnd();
    EndDrawing();
}

} // namespace syntraq
