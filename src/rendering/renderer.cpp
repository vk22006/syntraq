//
// src/rendering/renderer.cpp
//
// Renderer implementation.
// All Raylib and ImGui includes are confined to this translation unit.
//

#include "syntraq/rendering/renderer.hpp"
#include "syntraq/signals/rl_controller.hpp"
#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/vehicles/emergency_priority_manager.hpp"
#include "syntraq/routing/traffic_aware_routing.hpp"

// Third-party headers — isolated to this .cpp
#include "raylib.h"
#include "rlImGui.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace syntraq {

// ── Internal Camera Wrapper ──────────────────────────────────────────────────

struct RenderCamera {
    Camera2D  cam{};
    float     min_zoom{ 0.25f };
    float     max_zoom{ 4.00f };

    // Tilemap textures
    Texture2D sidewalks_tex{};
    Texture2D road_tex{};
    bool      textures_loaded{ false };
};

namespace {

std::string resolve_asset_path(const std::string& rel_path) {
    if (std::filesystem::exists(rel_path)) {
        return rel_path;
    }
    if (std::filesystem::exists("../" + rel_path)) {
        return "../" + rel_path;
    }
    if (std::filesystem::exists("../../" + rel_path)) {
        return "../../" + rel_path;
    }
    const std::string app_dir = GetApplicationDirectory();
    if (!app_dir.empty()) {
        const std::filesystem::path p = std::filesystem::path(app_dir) / rel_path;
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }
    return rel_path;
}

} // namespace

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

    // ── Load tilemap textures ──────────────────────────────────────────────
    const std::string sidewalks_path = resolve_asset_path("assets/Sidewalks.png");
    if (std::filesystem::exists(sidewalks_path)) {
        camera_->sidewalks_tex = LoadTexture(sidewalks_path.c_str());
        if (camera_->sidewalks_tex.id > 0) {
            SetTextureFilter(camera_->sidewalks_tex, TEXTURE_FILTER_POINT);
        }
    }

    const std::string road_path = resolve_asset_path("assets/UpdatedDarkRoad.png");
    if (std::filesystem::exists(road_path)) {
        camera_->road_tex = LoadTexture(road_path.c_str());
        if (camera_->road_tex.id > 0) {
            SetTextureFilter(camera_->road_tex, TEXTURE_FILTER_POINT);
        }
    }
    camera_->textures_loaded = (camera_->sidewalks_tex.id > 0 && camera_->road_tex.id > 0);
}

Renderer::~Renderer() {
    if (camera_->sidewalks_tex.id > 0) {
        UnloadTexture(camera_->sidewalks_tex);
    }
    if (camera_->road_tex.id > 0) {
        UnloadTexture(camera_->road_tex);
    }
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
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers,
    SignalControllerType                                               controller_type,
    RoutingStrategyType                                                active_routing,
    const EmergencyCorridorStatus*                                     emergency_status,
    bool                                                               ai_model_enabled,
    float                                                              traffic_density,
    const TrafficAwareRouting*                                         traffic_aware_router,
    const std::unordered_map<RoadId, uint32_t>*                        road_queues) {
    update_camera(network, vehicles);

    begin_frame();

    // ── 2D World Space Rendering ───────────────────────────────────────────
    BeginMode2D(camera_->cam);
    draw_world_grid();
    draw_network(network);

    if (debug_flags_.show_predicted_congestion) {
        draw_visual_debug_predicted_congestion(network, vehicles, traffic_aware_router, road_queues);
    }
    if (debug_flags_.show_routes) {
        draw_visual_debug_routes(network);
    }
    if (debug_flags_.show_queues) {
        draw_visual_debug_queues(network, road_queues, vehicles);
    }
    if (debug_flags_.show_traffic_signals && signal_controllers) {
        draw_traffic_signals(network, *signal_controllers);
    }
    if (debug_flags_.show_intersection_phases && signal_controllers) {
        draw_visual_debug_phases(network, *signal_controllers);
    }
    draw_vehicles(network, vehicles);
    if (debug_flags_.show_selected_paths) {
        draw_visual_debug_selected_paths(network, vehicles);
    }
    EndMode2D();

    // ── Screen Space HUD & UI Overlays ─────────────────────────────────────
    draw_status_bar(state, current_speed_scale, ai_model_enabled);
    SimControlAction action = draw_imgui(state,
                                         network,
                                         vehicles,
                                         current_speed_scale,
                                         signal_controllers,
                                         controller_type,
                                         active_routing,
                                         emergency_status,
                                         ai_model_enabled,
                                         traffic_density,
                                         traffic_aware_router,
                                         road_queues);

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

void Renderer::update_camera(const RoadNetwork& network, const std::vector<Vehicle>& vehicles) {
    if (!camera_initialized_ && network.intersection_count() > 0) {
        reset_camera(network);
        camera_initialized_ = true;
    }

    // Smoothly track selected vehicle if camera follow is active
    if (follow_selected_vehicle_ && selected_vehicle_id_ != kInvalidVehicleId) {
        for (const auto& v : vehicles) {
            if (v.id == selected_vehicle_id_ && v.state != VehicleState::Arrived) {
                const Road* rd = network.road(v.current_road);
                if (rd) {
                    const Intersection* src = network.intersection(rd->from);
                    const Intersection* dst = network.intersection(rd->to);
                    if (src && dst) {
                        const float t = (rd->length_m > 0.0f) ? std::clamp(v.progress_m / rd->length_m, 0.0f, 1.0f) : 0.0f;
                        const float vx = src->position.x + (dst->position.x - src->position.x) * t;
                        const float vy = src->position.y + (dst->position.y - src->position.y) * t;
                        camera_->cam.target.x += (vx - camera_->cam.target.x) * 0.12f;
                        camera_->cam.target.y += (vy - camera_->cam.target.y) * 0.12f;
                    }
                }
                break;
            }
        }
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
            follow_selected_vehicle_ = false; // user took control of camera
        }

        // 3. Left-click to pick and inspect vehicle in 2D world
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            const Vector2 mouse_screen = GetMousePosition();
            const Vector2 mouse_world  = GetScreenToWorld2D(mouse_screen, camera_->cam);

            VehicleId hit_id = kInvalidVehicleId;
            float closest_dist_sq = 28.0f * 28.0f; // 28 world units selection radius

            for (const auto& v : vehicles) {
                if (v.state == VehicleState::Arrived) continue;
                const Road* rd = network.road(v.current_road);
                if (!rd) continue;
                const Intersection* src = network.intersection(rd->from);
                const Intersection* dst = network.intersection(rd->to);
                if (!src || !dst) continue;

                const float t = (rd->length_m > 0.0f) ? std::clamp(v.progress_m / rd->length_m, 0.0f, 1.0f) : 0.0f;
                const float vx = src->position.x + (dst->position.x - src->position.x) * t;
                const float vy = src->position.y + (dst->position.y - src->position.y) * t;

                const float dx = mouse_world.x - vx;
                const float dy = mouse_world.y - vy;
                const float dist_sq = dx * dx + dy * dy;

                if (dist_sq < closest_dist_sq) {
                    closest_dist_sq = dist_sq;
                    hit_id = v.id;
                }
            }
            selected_vehicle_id_ = hit_id;
        }
    }

    // 4. Keep viewport center aligned if window resizes while keeping relative target
    if (IsWindowResized()) {
        camera_->cam.offset = { screen_w * 0.50f, (screen_h - 26.0f) * 0.50f };
    }
}

void Renderer::begin_frame() {
    BeginDrawing();
    ClearBackground(Color{ 75, 216, 51, 255 }); // Grass green matching Sidewalks.png
    rlImGuiBegin();
}

void Renderer::draw_world_grid() {
    if (camera_->sidewalks_tex.id > 0) {
        constexpr float kTileSize = 32.0f;

        // Viewport visible bounds in world coordinates
        const Vector2 top_left = GetScreenToWorld2D({ 0.0f, 0.0f }, camera_->cam);
        const Vector2 bot_right = GetScreenToWorld2D(
            { static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight()) },
            camera_->cam
        );

        // Clamp to generous world bounds covering city extent
        constexpr float kWorldMin = -800.0f;
        constexpr float kWorldMax = 2000.0f;

        const float start_x = std::max(kWorldMin, std::floor((top_left.x - kTileSize) / kTileSize) * kTileSize);
        const float end_x   = std::min(kWorldMax, std::ceil((bot_right.x + kTileSize) / kTileSize) * kTileSize);
        const float start_y = std::max(kWorldMin, std::floor((top_left.y - kTileSize) / kTileSize) * kTileSize);
        const float end_y   = std::min(kWorldMax, std::ceil((bot_right.y + kTileSize) / kTileSize) * kTileSize);

        for (float y = start_y; y <= end_y; y += kTileSize) {
            const int gy = static_cast<int>(std::floor(y / kTileSize));
            for (float x = start_x; x <= end_x; x += kTileSize) {
                const int gx = static_cast<int>(std::floor(x / kTileSize));
                // Deterministic pseudo-random variation across the 6 grass tiles in Sidewalks.png (x=0..192, y=288..320)
                const uint32_t hash = static_cast<uint32_t>(gx * 73856093 ^ gy * 19349663);
                const int variant = static_cast<int>(hash % 6);

                const Rectangle src{ static_cast<float>(variant * 32), 288.0f, 32.0f, 32.0f };
                const Vector2 pos{ x, y };
                DrawTextureRec(camera_->sidewalks_tex, src, pos, WHITE);
            }
        }
    } else {
        // Fallback procedural coordinate grid
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

        if (camera_->road_tex.id > 0) {
            // Render road tiles once per bidirectional pair
            if (static_cast<uint32_t>(road.from) < static_cast<uint32_t>(road.to)) {
                const float dx = b.x - a.x;
                const float dy = b.y - a.y;
                const float len = std::hypot(dx, dy);
                if (len > 0.0f) {
                    const float ux = dx / len;
                    const float uy = dy / len;
                    const float angle_deg = std::atan2(dy, dx) * (180.0f / 3.141592653589793f);

                    constexpr float kJuncHalf = 16.0f; // Half-width of 32x32 junction box
                    constexpr float kCwLen    = 32.0f; // Crosswalk tile length
                    constexpr float kTileLen  = 32.0f; // Road tile length

                    const bool has_cw_a = (network.incoming_roads(road.from).size() > 1 && len >= 96.0f);
                    const bool has_cw_b = (network.incoming_roads(road.to).size() > 1 && len >= 96.0f);

                    const float road_start = kJuncHalf + (has_cw_a ? kCwLen : 0.0f);
                    const float road_end   = len - kJuncHalf - (has_cw_b ? kCwLen : 0.0f);

                    // Crosswalk entering junction A (facing backwards towards A)
                    if (has_cw_a) {
                        const Vector2 pos{ a.x + ux * kJuncHalf, a.y + uy * kJuncHalf };
                        const Rectangle src_cw{ 32.0f, 96.0f, -32.0f, 32.0f };
                        const Rectangle dst_cw{ pos.x, pos.y, 32.0f, 32.0f };
                        DrawTexturePro(camera_->road_tex, src_cw, dst_cw, { 0.0f, 16.0f }, angle_deg, WHITE);
                    }

                    // Tiled straight road segments with asphalt and dashed center lane divider
                    float d = road_start;
                    while (d < road_end) {
                        const float step = std::min(kTileLen, road_end - d);
                        const Vector2 pos{ a.x + ux * d, a.y + uy * d };
                        const Rectangle src_r{ 0.0f, 96.0f, step, 32.0f };
                        const Rectangle dst_r{ pos.x, pos.y, step, 32.0f };
                        DrawTexturePro(camera_->road_tex, src_r, dst_r, { 0.0f, 16.0f }, angle_deg, WHITE);
                        d += step;
                    }

                    // Crosswalk entering junction B (facing forward towards B)
                    if (has_cw_b) {
                        const Vector2 pos{ a.x + ux * (len - kJuncHalf - kCwLen), a.y + uy * (len - kJuncHalf - kCwLen) };
                        const Rectangle src_cw{ 32.0f, 96.0f, 32.0f, 32.0f };
                        const Rectangle dst_cw{ pos.x, pos.y, 32.0f, 32.0f };
                        DrawTexturePro(camera_->road_tex, src_cw, dst_cw, { 0.0f, 16.0f }, angle_deg, WHITE);
                    }
                }
            }
        } else {
            // Procedural fallback
            DrawLineEx(a, b, road_w + 3.0f, pal::road_edge);
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

        if (camera_->road_tex.id > 0) {
            // Textured junction center square from UpdatedDarkRoad.png (32x32)
            const Rectangle junc_src{ 64.0f, 96.0f, 32.0f, 32.0f };
            const Rectangle junc_dest{ cx, cy, 32.0f, 32.0f };
            const Vector2 junc_origin{ 16.0f, 16.0f };
            DrawTexturePro(camera_->road_tex, junc_src, junc_dest, junc_origin, 0.0f, WHITE);
        } else {
            // Junction apron (blends with road asphalt, preventing protruding edges)
            DrawCircleV({ cx, cy }, kIntersectionRadius, pal::node_apron);
            DrawCircleLinesV({ cx, cy }, kIntersectionRadius, pal::road_edge);

            // Technical junction core disc
            DrawCircleV({ cx, cy }, 5.5f, pal::node_core);
            DrawCircleLinesV({ cx, cy }, 5.5f, pal::node_border);
        }

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

        const bool is_emergency = v.is_emergency();

        // Emergency siren pulsing halo
        if (is_emergency) {
            const float t_siren = static_cast<float>(GetTime());
            const float halo_radius = 16.0f + 4.0f * std::sin(t_siren * 12.0f);
            DrawCircleV(pos, halo_radius, { 255, 30, 30, 50 });
        }

        const Color car_col = is_emergency
                            ? Color{ 245, 248, 255, 255 }
                            : pal::vehicle_colors[static_cast<uint32_t>(v.id) % pal::kNumVehicleColors];

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

        // Emergency reflective high-visibility side chevrons
        if (is_emergency) {
            const Rectangle stripe_rec{ pos.x, pos.y, kCarLength * 0.7f, 1.6f };
            DrawRectanglePro(stripe_rec,
                             { (kCarLength * 0.7f) * 0.5f, 0.8f },
                             angle_deg,
                             { 230, 30, 30, 240 });
        }

        // 3. Cabin / Roof (dark tinted glass, slightly set back from center)
        const float cabin_len = kCarLength * 0.46f;
        const float cabin_wid = kCarWidth * 0.72f;
        const Vector2 cabin_pos{ pos.x - udx * 0.6f, pos.y - udy * 0.6f };
        const Rectangle cabin_rec{ cabin_pos.x, cabin_pos.y, cabin_len, cabin_wid };
        DrawRectanglePro(cabin_rec,
                         { cabin_len * 0.5f, cabin_wid * 0.5f },
                         angle_deg,
                         { 18, 22, 32, 230 });

        // Emergency rooftop strobe bar (alternating flashing red and blue strobes)
        if (is_emergency) {
            const float t_strobe = static_cast<float>(GetTime());
            const bool strobe_red = (std::fmod(t_strobe * 8.0f, 2.0f) < 1.0f);
            const Color left_light = strobe_red ? Color{ 255, 30, 30, 255 } : Color{ 60, 10, 10, 180 };
            const Color right_light = !strobe_red ? Color{ 30, 140, 255, 255 } : Color{ 10, 30, 80, 180 };
            DrawCircleV({ cabin_pos.x - perp_x * 2.2f, cabin_pos.y - perp_y * 2.2f }, 1.8f, left_light);
            DrawCircleV({ cabin_pos.x + perp_x * 2.2f, cabin_pos.y + perp_y * 2.2f }, 1.8f, right_light);
        }

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

// ── Visual Debugging Overlays in 2D World Space ─────────────────────────────

void Renderer::draw_visual_debug_routes(const RoadNetwork& network) {
    constexpr Color arrow_col = { 80, 200, 240, 160 };

    for (const auto& [rid, road] : network.roads()) {
        const Intersection* src = network.intersection(road.from);
        const Intersection* dst = network.intersection(road.to);
        if (!src || !dst) continue;

        const float dx = dst->position.x - src->position.x;
        const float dy = dst->position.y - src->position.y;
        const float len = std::hypot(dx, dy);
        if (len < 32.0f) continue;

        const float udx = dx / len;
        const float udy = dy / len;
        const float perp_x = -udy;
        const float perp_y =  udx;

        constexpr float kLaneWidthPx = 14.0f;
        const uint32_t lanes = road.lane_count();

        for (uint32_t l = 0; l < lanes; ++l) {
            const float lane_offset = (static_cast<float>(l) - static_cast<float>(lanes - 1) * 0.5f) * kLaneWidthPx;
            const float start_dist = 28.0f;
            const float step = 45.0f;

            for (float d = start_dist; d < len - 22.0f; d += step) {
                const Vector2 tip = {
                    src->position.x + udx * d + perp_x * lane_offset,
                    src->position.y + udy * d + perp_y * lane_offset
                };
                const Vector2 left = {
                    tip.x - udx * 6.0f + perp_x * 4.0f,
                    tip.y - udy * 6.0f + perp_y * 4.0f
                };
                const Vector2 right = {
                    tip.x - udx * 6.0f - perp_x * 4.0f,
                    tip.y - udy * 6.0f - perp_y * 4.0f
                };
                DrawLineEx(left, tip, 1.8f, arrow_col);
                DrawLineEx(right, tip, 1.8f, arrow_col);
            }
        }
    }
}

void Renderer::draw_visual_debug_queues(
    const RoadNetwork&                          network,
    const std::unordered_map<RoadId, uint32_t>* road_queues,
    const std::vector<Vehicle>&                 vehicles) {

    std::unordered_map<RoadId, uint32_t> local_queues;
    const std::unordered_map<RoadId, uint32_t>* q_map = road_queues;

    if (!q_map) {
        for (const auto& v : vehicles) {
            if (v.state == VehicleState::Arrived) continue;
            if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
                local_queues[v.current_road]++;
            }
        }
        q_map = &local_queues;
    }

    constexpr float kStopDistFromCenter = 26.0f;
    constexpr float kLaneWidthPx = 14.0f;

    for (const auto& [rid, count] : *q_map) {
        if (count == 0) continue;
        const Road* road = network.road(rid);
        if (!road) continue;

        const Intersection* src = network.intersection(road->from);
        const Intersection* dst = network.intersection(road->to);
        if (!src || !dst) continue;

        const float dx = dst->position.x - src->position.x;
        const float dy = dst->position.y - src->position.y;
        const float len = std::hypot(dx, dy);
        if (len <= 0.001f) continue;

        const float udx = dx / len;
        const float udy = dy / len;
        const float perp_x = -udy;
        const float perp_y =  udx;

        const float road_w = static_cast<float>(road->lane_count()) * kLaneWidthPx;
        const float stop_x = dst->position.x - udx * kStopDistFromCenter + perp_x * (road_w * 0.25f);
        const float stop_y = dst->position.y - udy * kStopDistFromCenter + perp_y * (road_w * 0.25f);

        const float q_len = std::min(len * 0.85f, static_cast<float>(count) * 8.0f);
        const float tail_x = stop_x - udx * q_len;
        const float tail_y = stop_y - udy * q_len;

        // Glowing queue strip
        DrawLineEx({ tail_x, tail_y }, { stop_x, stop_y }, road_w * 0.75f, Color{ 255, 45, 45, 75 });
        DrawLineEx({ tail_x, tail_y }, { stop_x, stop_y }, 2.5f, Color{ 255, 80, 80, 220 });

        // Queue count badge near tail
        const float badge_x = tail_x - udx * 8.0f + perp_x * (road_w * 0.5f + 4.0f);
        const float badge_y = tail_y - udy * 8.0f + perp_y * (road_w * 0.5f + 4.0f);

        char q_buf[16];
        std::snprintf(q_buf, sizeof(q_buf), "Q:%u", count);
        const int tw = MeasureText(q_buf, 9);
        DrawRectangle(static_cast<int>(badge_x - tw * 0.5f - 2),
                      static_cast<int>(badge_y - 6),
                      tw + 4, 12, Color{ 20, 24, 32, 230 });
        DrawRectangleLines(static_cast<int>(badge_x - tw * 0.5f - 2),
                           static_cast<int>(badge_y - 6),
                           tw + 4, 12, Color{ 255, 60, 60, 240 });
        DrawText(q_buf, static_cast<int>(badge_x - tw * 0.5f), static_cast<int>(badge_y - 5), 9, Color{ 255, 140, 140, 255 });
    }
}

void Renderer::draw_visual_debug_phases(
    const RoadNetwork&                                                 network,
    const std::unordered_map<IntersectionId, TrafficSignalController>& controllers) {

    constexpr float kEntryDist = 20.0f;
    for (const auto& [inter_id, controller] : controllers) {
        const Intersection* node = network.intersection(inter_id);
        if (!node) continue;

        const Vector2 center{ node->position.x, node->position.y };

        for (const auto& [rid, light] : controller.lights()) {
            const Road* rd = network.road(rid);
            if (!rd) continue;
            const Intersection* from_node = network.intersection(rd->from);
            if (!from_node) continue;

            const float dx = center.x - from_node->position.x;
            const float dy = center.y - from_node->position.y;
            const float len = std::hypot(dx, dy);
            if (len <= 0.001f) continue;

            const float udx = dx / len;
            const float udy = dy / len;
            const float perp_x = -udy;
            const float perp_y =  udx;

            const Vector2 entry_pt{ center.x - udx * kEntryDist, center.y - udy * kEntryDist };

            if (light.color == SignalColor::Green) {
                const Vector2 tip{ center.x - udx * 8.0f, center.y - udy * 8.0f };
                const Vector2 w1 { tip.x - udx * 5.0f + perp_x * 4.0f, tip.y - udy * 5.0f + perp_y * 4.0f };
                const Vector2 w2 { tip.x - udx * 5.0f - perp_x * 4.0f, tip.y - udy * 5.0f - perp_y * 4.0f };
                DrawLineEx(entry_pt, tip, 3.0f, Color{ 45, 245, 105, 230 });
                DrawLineEx(w1, tip, 2.5f, Color{ 45, 245, 105, 255 });
                DrawLineEx(w2, tip, 2.5f, Color{ 45, 245, 105, 255 });
            } else if (light.color == SignalColor::Yellow) {
                const Vector2 bar_p1{ entry_pt.x - perp_x * 6.0f, entry_pt.y - perp_y * 6.0f };
                const Vector2 bar_p2{ entry_pt.x + perp_x * 6.0f, entry_pt.y + perp_y * 6.0f };
                DrawLineEx(bar_p1, bar_p2, 2.5f, Color{ 255, 205, 35, 240 });
            } else {
                const Vector2 bar_p1{ entry_pt.x - perp_x * 6.0f, entry_pt.y - perp_y * 6.0f };
                const Vector2 bar_p2{ entry_pt.x + perp_x * 6.0f, entry_pt.y + perp_y * 6.0f };
                DrawLineEx(bar_p1, bar_p2, 2.5f, Color{ 255, 45, 45, 220 });
            }
        }

        char phase_buf[32];
        const char* st_prefix = (controller.current_stage() == PhaseStage::Green) ? "G"
                              : (controller.current_stage() == PhaseStage::Yellow) ? "Y" : "R";
        std::snprintf(phase_buf, sizeof(phase_buf), "%s:%.1fs", st_prefix, controller.stage_remaining_s());

        Color txt_col = (controller.current_stage() == PhaseStage::Green) ? Color{ 50, 245, 110, 255 }
                      : (controller.current_stage() == PhaseStage::Yellow) ? Color{ 255, 215, 50, 255 }
                      : Color{ 255, 60, 60, 255 };

        const int fs = 10;
        const int tw = MeasureText(phase_buf, fs);
        const int bx = static_cast<int>(center.x - tw * 0.5f - 3);
        const int by = static_cast<int>(center.y - 24);

        DrawRectangle(bx, by, tw + 6, 13, Color{ 16, 20, 28, 220 });
        DrawRectangleLines(bx, by, tw + 6, 13, Color{ 40, 50, 65, 240 });
        DrawText(phase_buf, bx + 3, by + 1, fs, txt_col);
    }
}

void Renderer::draw_visual_debug_predicted_congestion(
    const RoadNetwork&                          network,
    const std::vector<Vehicle>&                 vehicles,
    const TrafficAwareRouting*                  router,
    const std::unordered_map<RoadId, uint32_t>* road_queues) {

    std::unordered_map<RoadId, uint32_t> counts;
    for (const auto& v : vehicles) {
        if (v.state != VehicleState::Arrived) {
            counts[v.current_road]++;
        }
    }

    constexpr float kLaneWidthPx = 14.0f;

    for (const auto& [rid, road] : network.roads()) {
        const Intersection* src = network.intersection(road.from);
        const Intersection* dst = network.intersection(road.to);
        if (!src || !dst) continue;

        float cong = 0.0f;
        float delay_s = 0.0f;

        if (router) {
            const auto cond = router->condition_for(rid);
            cong = cond.congestion_ratio;
            delay_s = cond.predicted_delay_s;
        } else {
            const uint32_t queued = road_queues ? (road_queues->count(rid) ? road_queues->at(rid) : 0u) : 0u;
            const uint32_t total = counts[rid];
            const float cap = std::max(2.0f, (road.length_m / 7.5f) * static_cast<float>(road.lane_count()));
            cong = std::clamp(static_cast<float>(queued) / cap, 0.0f, 1.0f);
            delay_s = static_cast<float>(queued) * 2.0f;
            if (total > 0 && cong < 0.1f) {
                cong = std::clamp(static_cast<float>(total) / (cap * 2.0f), 0.0f, 1.0f);
            }
        }

        if (cong < 0.12f && delay_s < 1.0f) continue;

        const Vector2 a{ src->position.x, src->position.y };
        const Vector2 b{ dst->position.x, dst->position.y };
        const float road_w = static_cast<float>(road.lane_count()) * kLaneWidthPx;

        Color tint;
        if (cong < 0.35f) {
            tint = Color{ 255, 205, 45, 90 };
        } else if (cong < 0.65f) {
            tint = Color{ 255, 115, 30, 115 };
        } else {
            tint = Color{ 235, 30, 45, 140 };
        }

        DrawLineEx(a, b, road_w + 3.0f, tint);

        if (delay_s >= 1.5f) {
            char d_buf[16];
            std::snprintf(d_buf, sizeof(d_buf), "+%.1fs", delay_s);
            const float mid_x = (a.x + b.x) * 0.5f;
            const float mid_y = (a.y + b.y) * 0.5f;
            const int tw = MeasureText(d_buf, 9);
            DrawRectangle(static_cast<int>(mid_x - tw * 0.5f - 2), static_cast<int>(mid_y - 6), tw + 4, 12, Color{ 20, 24, 32, 220 });
            DrawText(d_buf, static_cast<int>(mid_x - tw * 0.5f), static_cast<int>(mid_y - 5), 9, Color{ 255, 180, 50, 240 });
        }
    }
}

void Renderer::draw_visual_debug_selected_paths(
    const RoadNetwork&          network,
    const std::vector<Vehicle>& vehicles) {

    if (selected_vehicle_id_ == kInvalidVehicleId) return;

    const Vehicle* sel_v = nullptr;
    for (const auto& v : vehicles) {
        if (v.id == selected_vehicle_id_) {
            sel_v = &v;
            break;
        }
    }
    if (!sel_v || sel_v->state == VehicleState::Arrived) return;

    const Road* cur_rd = network.road(sel_v->current_road);
    if (!cur_rd) return;
    const Intersection* cur_src = network.intersection(cur_rd->from);
    const Intersection* cur_dst = network.intersection(cur_rd->to);
    if (!cur_src || !cur_dst) return;

    const float t = (cur_rd->length_m > 0.0f) ? std::clamp(sel_v->progress_m / cur_rd->length_m, 0.0f, 1.0f) : 0.0f;
    const Vector2 v_pos = {
        cur_src->position.x + (cur_dst->position.x - cur_src->position.x) * t,
        cur_src->position.y + (cur_dst->position.y - cur_src->position.y) * t
    };

    const float cur_time = static_cast<float>(GetTime());
    const float pulse_r = 13.0f + 2.5f * std::sin(cur_time * 6.0f);
    DrawCircleLinesV(v_pos, pulse_r, Color{ 0, 230, 255, 230 });
    DrawCircleLinesV(v_pos, pulse_r + 3.0f, Color{ 0, 180, 255, 120 });
    DrawCircleV(v_pos, 2.5f, Color{ 0, 255, 255, 255 });

    const auto& r_list = sel_v->route;
    const uint32_t cur_hop = sel_v->route_index;

    for (uint32_t i = 0; i < r_list.size(); ++i) {
        const Road* rd = network.road(r_list[i]);
        if (!rd) continue;
        const Intersection* src = network.intersection(rd->from);
        const Intersection* dst = network.intersection(rd->to);
        if (!src || !dst) continue;

        const Vector2 p1{ src->position.x, src->position.y };
        const Vector2 p2{ dst->position.x, dst->position.y };

        if (i < cur_hop) {
            DrawLineEx(p1, p2, 2.0f, Color{ 110, 135, 160, 100 });
        } else if (i == cur_hop) {
            DrawLineEx(v_pos, p2, 6.0f, Color{ 255, 215, 40, 110 });
            DrawLineEx(v_pos, p2, 2.5f, Color{ 255, 235, 80, 240 });
        } else {
            DrawLineEx(p1, p2, 6.0f, Color{ 0, 210, 255, 120 });
            DrawLineEx(p1, p2, 2.5f, Color{ 100, 240, 255, 255 });

            const Vector2 mid{ (p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f };
            DrawCircleV(mid, 3.5f, Color{ 0, 240, 255, 230 });
        }
    }

    if (!r_list.empty()) {
        const Road* last_rd = network.road(r_list.back());
        if (last_rd) {
            const Intersection* dest_node = network.intersection(last_rd->to);
            if (dest_node) {
                const Vector2 d_pos{ dest_node->position.x, dest_node->position.y };
                const float dest_r = 16.0f + 2.0f * std::sin(cur_time * 8.0f);
                DrawCircleLinesV(d_pos, dest_r, Color{ 255, 215, 0, 220 });
                DrawCircleLinesV(d_pos, dest_r + 4.0f, Color{ 255, 180, 0, 130 });
                DrawCircleV(d_pos, 4.0f, Color{ 255, 225, 30, 255 });
                DrawText("TARGET", static_cast<int>(d_pos.x - 18), static_cast<int>(d_pos.y - 25), 10, Color{ 255, 230, 80, 240 });
            }
        }
    }
}

void Renderer::draw_status_bar(const SimState& state, float current_speed_scale, bool ai_model_enabled) {
    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    DrawRectangle(0, static_cast<int>(screen_h - 26.0f), static_cast<int>(screen_w), 26, pal::status_bg);
    DrawLineEx({ 0.0f, screen_h - 26.0f }, { screen_w, screen_h - 26.0f }, 1.0f, pal::status_border);

    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "SyntraQ | State: %s (%.2fx) | AI: %s | Sim: %.1f s (Tick %llu) | Zoom: %.2fx | Pan: Wheel=Zoom, M/R-Drag=Pan, L-Click=Select Veh, R=Reset View, Space=Pause",
                  state.running ? "RUNNING" : "PAUSED",
                  current_speed_scale,
                  ai_model_enabled ? "ON (ONNX)" : "OFF",
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

    // Responsive 3-button row
    const char* pause_label = state.running
        ? (avail_w >= 280.0f ? "Pause (Space)" : "Pause")
        : (avail_w >= 280.0f ? "Resume (Space)" : "Resume");

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
        selected_vehicle_id_ = kInvalidVehicleId;
        follow_selected_vehicle_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset View", ImVec2(btn_w, 26.0f))) {
        reset_camera(network);
    }

    // Step button (available when paused for frame-by-frame debugging)
    if (!state.running) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.35f, 0.55f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.45f, 0.65f, 1.00f));
        if (ImGui::Button("Step Forward [1 dt]", ImVec2(avail_w, 24.0f))) {
            action.request_step = true;
        }
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Advance simulation by one fixed timestep (0.1s)");
        }
    }

    // Speed multiplier buttons
    ImGui::Spacing();
    ImGui::Text("Simulation Speed:");
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

    // Interactive speed slider for smooth control
    float speed_slider = current_speed_scale;
    if (ImGui::SliderFloat("##SpeedSlider", &speed_slider, 0.1f, 5.0f, "Speed: %.2fx")) {
        action.request_speed_scale = speed_slider;
    }

    // Status indicator line
    ImGui::Spacing();
    if (state.running) {
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.45f, 1.00f), "Status: RUNNING (%.2fx)", current_speed_scale);
    } else {
        ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.00f), "Status: PAUSED [Step Mode]");
    }
}

void Renderer::draw_traffic_density_controls(
    float              traffic_density,
    SimControlAction&  action) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "TRAFFIC DENSITY");
    ImGui::Separator();

    float dens = traffic_density;
    if (ImGui::SliderFloat("##DensitySlider", &dens, 0.2f, 3.0f, "Density: %.2fx")) {
        action.request_traffic_density = dens;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Scales active vehicle fleet and spawning frequency in real-time");
    }

    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float btn_w = (avail_w - 3.0f * spacing) / 4.0f;

    const float dens_presets[] = { 0.5f, 1.0f, 1.5f, 2.0f };
    const char* dens_labels[]  = { "0.5x Light", "1.0x Norm", "1.5x Heavy", "2.0x Rush" };
    for (int i = 0; i < 4; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (std::abs(traffic_density - dens_presets[i]) < 0.05f);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.80f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.65f, 0.90f, 1.00f));
        }
        if (ImGui::Button(dens_labels[i], ImVec2(btn_w, 22.0f))) {
            action.request_traffic_density = dens_presets[i];
        }
        if (is_active) {
            ImGui::PopStyleColor(2);
        }
    }
}

void Renderer::draw_ai_subsystem_controls(
    bool               ai_model_enabled,
    SimControlAction&  action) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "AI POLICY & MODELS");
    ImGui::Separator();

    bool ai_enabled = ai_model_enabled;
    if (ImGui::Checkbox("Enable AI Models (ONNX)", &ai_enabled)) {
        action.request_ai_enabled = ai_enabled;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Toggle native ONNX Runtime neural policies ON/OFF\nWhen disabled, controllers safely run in rule-based heuristic mode.");
    }

    if (ai_model_enabled) {
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "AI State: ONLINE (ONNX Runtime)");
        ImGui::TextDisabled("Policy: traffic_signal_dqn.onnx");
    } else {
        ImGui::TextColored(ImVec4(1.00f, 0.70f, 0.20f, 1.00f), "AI State: DISABLED (Heuristic)");
        ImGui::TextDisabled("Using safe rule-based heuristic fallback");
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
        ImGui::TableSetColumnIndex(0); ImGui::Text("Sim FPS");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", state.simulation_fps);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Render FPS");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%d", GetFPS());

        ImGui::EndTable();
    }
}

void Renderer::draw_debug_controls() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "VISUAL DEBUGGING");
    ImGui::Separator();

    ImGui::Checkbox("Routes & Flow Vectors",       &debug_flags_.show_routes);
    ImGui::Checkbox("Queues & Approach Delays",    &debug_flags_.show_queues);
    ImGui::Checkbox("Intersection Phases & Timers",&debug_flags_.show_intersection_phases);
    ImGui::Checkbox("Predicted Congestion Heat",   &debug_flags_.show_predicted_congestion);
    ImGui::Checkbox("Selected Vehicle Paths",      &debug_flags_.show_selected_paths);

    if (ImGui::CollapsingHeader("Network Annotations")) {
        ImGui::Checkbox("Traffic Signals",         &debug_flags_.show_traffic_signals);
        ImGui::Checkbox("Intersection IDs",        &debug_flags_.show_intersection_ids);
        ImGui::Checkbox("Road IDs",                &debug_flags_.show_road_ids);
        ImGui::Checkbox("Lane Boundaries",         &debug_flags_.show_lane_boundaries);
        ImGui::Checkbox("Vehicle Heading Vectors", &debug_flags_.show_vehicle_vectors);
        ImGui::Checkbox("Telemetry Stats Panel",   &debug_flags_.show_debug_statistics);
    }
}

void Renderer::draw_simulation_panel(
    const SimState&    state,
    const RoadNetwork& network,
    float              current_speed_scale,
    bool               ai_model_enabled,
    float              traffic_density,
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
        draw_traffic_density_controls(traffic_density, action);
        draw_ai_subsystem_controls(ai_model_enabled, action);
        draw_debug_controls();
        if (debug_flags_.show_debug_statistics) {
            draw_simulation_metrics(state);
        }
    }
    ImGui::End();
}

void Renderer::draw_signal_status(
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers,
    SignalControllerType                                               controller_type,
    bool                                                               ai_model_enabled,
    SimControlAction&                                                  action) {
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "SIGNAL CONTROLLER MODE");
    ImGui::Separator();

    const char* ctrl_labels[] = { "Fixed", "Adaptive", "RL", "Coord" };
    const SignalControllerType ctrl_types[] = {
        SignalControllerType::FixedTime,
        SignalControllerType::Adaptive,
        SignalControllerType::ReinforcementLearning,
        SignalControllerType::Coordinated
    };

    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float btn_w   = (avail_w - 3.0f * spacing) / 4.0f;

    for (int i = 0; i < 4; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (controller_type == ctrl_types[i]);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.60f, 0.80f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.70f, 0.90f, 1.00f));
        }

        if (ImGui::Button(ctrl_labels[i], ImVec2(btn_w, 24.0f))) {
            action.request_controller_type = ctrl_types[i];
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Switch to %s signal controller", ctrl_labels[i]);
        }

        if (is_active) {
            ImGui::PopStyleColor(2);
        }
    }

    // RL Telemetry & Aggregated Metrics
    const RLController* first_rlc = nullptr;
    float avg_latency_us = 0.0f;
    float total_dps = 0.0f;
    uint32_t rl_count = 0;
    bool any_model_loaded = false;

    // Coordinated Telemetry
    uint32_t total_wave_ext = 0;
    uint32_t total_spill_mit = 0;
    uint32_t coord_count = 0;

    if (signal_controllers) {
        for (const auto& [id, ctrl] : *signal_controllers) {
            if (const auto* rlc = ctrl.as<RLController>()) {
                if (!first_rlc) first_rlc = rlc;
                avg_latency_us += rlc->average_inference_latency_us();
                total_dps += rlc->decisions_per_second();
                if (rlc->is_model_loaded()) any_model_loaded = true;
                rl_count++;
            } else if (const auto* cc = ctrl.as<CoordinatedController>()) {
                total_wave_ext += cc->green_wave_extensions_count();
                total_spill_mit += cc->spillback_mitigations_count();
                coord_count++;
            }
        }
    }

    if (controller_type == SignalControllerType::Coordinated || coord_count > 0) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "NETWORK COORDINATION TELEMETRY");
        ImGui::Separator();

        if (ImGui::BeginTable("CoordStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Green Waves Extended");
            ImGui::TableSetColumnIndex(1); ImGui::TextColored(ImVec4(0.20f, 0.90f, 0.95f, 1.00f), "%u", total_wave_ext);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Spillback Mitigations");
            ImGui::TableSetColumnIndex(1); ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.00f), "%u", total_spill_mit);

            ImGui::EndTable();
        }
    }

    if (controller_type == SignalControllerType::ReinforcementLearning || rl_count > 0) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "RL POLICY INFERENCE (DQN)");
        ImGui::Separator();

        if (ImGui::BeginTable("RLStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("AI Model");
            ImGui::TableSetColumnIndex(1);
            if (ai_model_enabled && any_model_loaded) {
                ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "LOADED (ONNX)");
                if (ImGui::IsItemHovered() && first_rlc) {
                    ImGui::SetTooltip("%s", first_rlc->model_path().c_str());
                }
            } else {
                ImGui::TextColored(ImVec4(1.00f, 0.70f, 0.20f, 1.00f), "HEURISTIC (No AI)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Running in safe heuristic fallback mode (no AI model loaded)");
                }
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Inference Latency");
            ImGui::TableSetColumnIndex(1);
            if (first_rlc && first_rlc->last_inference_latency_us() > 0.0f) {
                const float mean_lat = (rl_count > 0) ? (avg_latency_us / static_cast<float>(rl_count)) : 0.0f;
                ImGui::Text("%.1f us (avg: %.1f us)", first_rlc->last_inference_latency_us(), mean_lat);
            } else {
                ImGui::TextDisabled("N/A");
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Decisions / Sec");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f /s", total_dps);

            ImGui::EndTable();
        }
    }

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "TRAFFIC SIGNALS");
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

    if (ImGui::BeginTable("SignalControllersTable", 4, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Junction");
        ImGui::TableSetColumnIndex(1); ImGui::Text("Mode");
        ImGui::TableSetColumnIndex(2); ImGui::Text("Signal State");
        ImGui::TableSetColumnIndex(3); ImGui::Text("Remaining");

        for (const auto& [inter_id, controller] : sorted) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("I-%u", static_cast<uint32_t>(inter_id));

            ImGui::TableSetColumnIndex(1);
            if (const auto* rlc = controller->as<RLController>()) {
                if (rlc->last_action() == 1) {
                    ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.25f, 1.00f), "RL [SWT]");
                } else {
                    ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "RL [HOLD]");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Action: %s | Q(Maintain)=%.2f, Q(Switch)=%.2f | Latency: %.1f us",
                                      rlc->last_action() == 1 ? "Switch" : "Maintain",
                                      rlc->last_decision().q_maintain,
                                      rlc->last_decision().q_switch,
                                      rlc->last_inference_latency_us());
                }
            } else if (const auto* ac = controller->as<AdaptiveController>()) {
                switch (ac->last_decision()) {
                    case AdaptiveDecision::ExtendGreen:
                        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "ADAPT [EXT]");
                        break;
                    case AdaptiveDecision::ShortenPhase:
                        ImGui::TextColored(ImVec4(1.00f, 0.65f, 0.20f, 1.00f), "ADAPT [GAP]");
                        break;
                    case AdaptiveDecision::SwitchPhase:
                        ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.25f, 1.00f), "ADAPT [SWT]");
                        break;
                    case AdaptiveDecision::Hold:
                    default:
                        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "ADAPTIVE");
                        break;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", std::string(ac->last_decision_reason()).c_str());
                }
            } else if (const auto* cc = controller->as<CoordinatedController>()) {
                switch (cc->last_decision()) {
                    case CoordinatedDecision::ExtendGreenWave:
                        ImGui::TextColored(ImVec4(0.20f, 0.90f, 0.95f, 1.00f), "COORD [PLT]");
                        break;
                    case CoordinatedDecision::ExtendGreenLocal:
                        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "COORD [EXT]");
                        break;
                    case CoordinatedDecision::MitigateSpillback:
                        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.00f), "COORD [SPL]");
                        break;
                    case CoordinatedDecision::ShortenPhase:
                        ImGui::TextColored(ImVec4(1.00f, 0.65f, 0.20f, 1.00f), "COORD [GAP]");
                        break;
                    case CoordinatedDecision::SwitchPhase:
                        ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.25f, 1.00f), "COORD [SWT]");
                        break;
                    case CoordinatedDecision::Hold:
                    default:
                        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "COORD [HLD]");
                        break;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", std::string(cc->last_decision_reason()).c_str());
                }
            } else {
                ImGui::TextDisabled("FixedTime");
            }

            ImGui::TableSetColumnIndex(2);
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

            ImGui::TableSetColumnIndex(3);
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

    const char* scenario_names[] = { "Low Traffic", "Medium Traffic", "High Traffic", "Rush Hour", "Emergency Priority" };
    const TrafficScenario scenarios[] = {
        TrafficScenario::Low,
        TrafficScenario::Medium,
        TrafficScenario::High,
        TrafficScenario::RushHour,
        TrafficScenario::EmergencyPriority
    };
    const char* scenario_labels[] = { "Low", "Med", "High", "Rush", "Emerg" };

    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float btn_w   = (avail_w - 4.0f * spacing) / 5.0f;

    for (int i = 0; i < 5; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (state.scenario == scenarios[i]);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.25f, 0.25f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.35f, 0.35f, 1.00f));
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
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "TRAFFIC PERFORMANCE METRICS");
    ImGui::Separator();

    if (ImGui::BeginTable("VehStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Active Vehicles");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.active_vehicles);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Queued Vehicles");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.queued_vehicles);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Max Queue (Lane)");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.max_queue_len);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Average Speed");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f km/h (%.2f m/s)", state.avg_speed_mps * 3.6f, state.avg_speed_mps);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Average Wait Time");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f s", state.avg_wait_time_s);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Average Travel Time");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f s", state.avg_travel_time_s);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Throughput");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.0f veh/h", state.throughput_vph);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Congestion Ratio");
        ImGui::TableSetColumnIndex(1);
        const float cong_pct = state.congestion_ratio * 100.0f;
        if (cong_pct < 15.0f) {
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "%.1f%% (Free Flow)", cong_pct);
        } else if (cong_pct < 35.0f) {
            ImGui::TextColored(ImVec4(1.00f, 0.85f, 0.25f, 1.00f), "%.1f%% (Moderate)", cong_pct);
        } else {
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.00f), "%.1f%% (Congested)", cong_pct);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Total Spawned");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.total_spawned);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Total Arrived");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", state.total_arrived);

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
}

void Renderer::draw_selected_vehicle_inspector(
    const std::vector<Vehicle>& vehicles,
    const RoadNetwork&          network,
    SimControlAction&           action) {
    (void)action;

    std::string preview_text = "None (Click in world or select)";
    const Vehicle* selected_v = nullptr;

    for (const auto& v : vehicles) {
        if (v.id == selected_vehicle_id_) {
            selected_v = &v;
            preview_text = "V-" + std::to_string(static_cast<uint32_t>(v.id)) +
                           (v.is_emergency() ? " [EMERGENCY]" : " [Regular]");
            break;
        }
    }

    if (ImGui::BeginCombo("Select Vehicle", preview_text.c_str())) {
        if (ImGui::Selectable("None (Deselect)", selected_vehicle_id_ == kInvalidVehicleId)) {
            selected_vehicle_id_ = kInvalidVehicleId;
            follow_selected_vehicle_ = false;
        }
        for (const auto& v : vehicles) {
            if (v.state == VehicleState::Arrived) continue;
            std::string label = "V-" + std::to_string(static_cast<uint32_t>(v.id)) +
                                (v.is_emergency() ? " [EMERGENCY]" : " [Regular]");
            const bool is_sel = (v.id == selected_vehicle_id_);
            if (ImGui::Selectable(label.c_str(), is_sel)) {
                selected_vehicle_id_ = v.id;
            }
            if (is_sel) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    if (selected_v) {
        ImGui::Checkbox("Follow Vehicle with Camera", &follow_selected_vehicle_);

        if (ImGui::BeginTable("VehInspectTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Vehicle ID");
            ImGui::TableSetColumnIndex(1);
            if (selected_v->is_emergency()) {
                ImGui::TextColored(ImVec4(1.00f, 0.35f, 0.35f, 1.00f), "V-%u (EMERGENCY)", static_cast<uint32_t>(selected_v->id));
            } else {
                ImGui::Text("V-%u (Regular)", static_cast<uint32_t>(selected_v->id));
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Current Speed");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.1f km/h (%.2f m/s)", selected_v->speed_mps * 3.6f, selected_v->speed_mps);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Vehicle State");
            ImGui::TableSetColumnIndex(1);
            if (selected_v->state == VehicleState::Stopped || selected_v->speed_mps < 0.5f) {
                ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.00f), "QUEUED / STOPPED");
            } else {
                ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.00f), "CRUISING / MOVING");
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Current Road");
            ImGui::TableSetColumnIndex(1); ImGui::Text("R-%u (Lane %u)", static_cast<uint32_t>(selected_v->current_road), selected_v->lane_index);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Road Progress");
            ImGui::TableSetColumnIndex(1);
            const Road* cur_rd = network.road(selected_v->current_road);
            const float rlen = cur_rd ? cur_rd->length_m : 1.0f;
            ImGui::Text("%.1f / %.1f m (%.0f%%)", selected_v->progress_m, rlen, (selected_v->progress_m / rlen) * 100.0f);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Destination");
            ImGui::TableSetColumnIndex(1); ImGui::Text("I-%u", static_cast<uint32_t>(selected_v->destination));

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Route Progress");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("Hop %u of %zu roads", selected_v->route_index + 1, selected_v->route.size());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Trip Duration");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f s", selected_v->travel_time_s);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Time in Queue");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f s", selected_v->wait_time_s);

            ImGui::EndTable();
        }

        if (ImGui::Button("Deselect Vehicle", ImVec2(ImGui::GetContentRegionAvail().x, 24.0f))) {
            selected_vehicle_id_ = kInvalidVehicleId;
            follow_selected_vehicle_ = false;
        }
    } else {
        ImGui::TextDisabled("Tip: Left-click any vehicle in the viewport\nto inspect its speed, kinematics and route path.");
    }
}

void Renderer::draw_routing_strategy(
    RoutingStrategyType        active_routing,
    const TrafficAwareRouting* traffic_aware_router,
    SimControlAction&          action) {
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.00f), "ROUTING STRATEGY MODE");
    ImGui::Separator();

    const char* route_labels[] = { "Static (Distance)", "Traffic-Aware (Dynamic)" };
    const RoutingStrategyType route_types[] = {
        RoutingStrategyType::Static,
        RoutingStrategyType::TrafficAware
    };

    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float btn_w   = (avail_w - spacing) / 2.0f;

    for (int i = 0; i < 2; ++i) {
        if (i > 0) ImGui::SameLine();
        const bool is_active = (active_routing == route_types[i]);
        if (is_active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.65f, 0.40f, 1.00f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.75f, 0.45f, 1.00f));
        }

        if (ImGui::Button(route_labels[i], ImVec2(btn_w, 24.0f))) {
            action.request_routing_strategy = route_types[i];
        }
        if (ImGui::IsItemHovered()) {
            if (i == 0) {
                ImGui::SetTooltip("Static routing: based strictly on shortest road distance");
            } else {
                ImGui::SetTooltip("Traffic-aware routing: distance + congestion penalty + predicted delay");
            }
        }

        if (is_active) {
            ImGui::PopStyleColor(2);
        }
    }

    ImGui::Spacing();
    if (active_routing == RoutingStrategyType::TrafficAware) {
        ImGui::TextColored(ImVec4(0.35f, 0.90f, 0.50f, 1.00f), "Formula: Cost = Distance + Congestion + Delay");
        if (traffic_aware_router) {
            ImGui::TextDisabled("Monitored roads: %zu", traffic_aware_router->road_conditions().size());
        }
    } else {
        ImGui::TextColored(ImVec4(0.70f, 0.80f, 0.90f, 1.00f), "Formula: Cost = Road Distance (metres)");
    }
}

void Renderer::draw_traffic_panel(
    const SimState&                                                    state,
    const RoadNetwork&                                                 network,
    const std::vector<Vehicle>&                                        vehicles,
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers,
    SignalControllerType                                               controller_type,
    RoutingStrategyType                                                active_routing,
    const EmergencyCorridorStatus*                                     emergency_status,
    bool                                                               ai_model_enabled,
    const TrafficAwareRouting*                                         traffic_aware_router,
    const std::unordered_map<RoadId, uint32_t>*                        road_queues,
    SimControlAction&                                                  action) {
    (void)road_queues;
    const float screen_w = static_cast<float>(GetScreenWidth());
    const float screen_h = static_cast<float>(GetScreenHeight());

    const float panel_w     = std::clamp(screen_w * 0.22f, 310.0f, 360.0f);
    const float margin_x    = (screen_w < 1000.0f) ? 12.0f : 20.0f;
    const float min_panel_w = 280.0f;
    const float max_panel_w = std::min(450.0f, screen_w * 0.35f);
    const float max_panel_h = std::max(200.0f, screen_h - 50.0f);
    const float default_h   = std::min(max_panel_h, 840.0f);

    const float init_x = std::max(margin_x + panel_w + 40.0f, screen_w - panel_w - margin_x);
    const float init_y = std::max(20.0f, (screen_h - 26.0f - default_h) * 0.5f);

    ImGui::SetNextWindowPos({ init_x, init_y }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({ panel_w, default_h }, ImGuiCond_FirstUseEver);
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

        draw_traffic_information(state, network, action);

        ImGui::Spacing();
        if (ImGui::CollapsingHeader("VEHICLE & ROUTE INSPECTOR", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_selected_vehicle_inspector(vehicles, network, action);
        }

        ImGui::Spacing();
        if (ImGui::CollapsingHeader("EMERGENCY PRIORITY CORRIDOR", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_emergency_priority_panel(emergency_status, action);
        }

        ImGui::Spacing();
        if (ImGui::CollapsingHeader("ROUTING ENGINE", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_routing_strategy(active_routing, traffic_aware_router, action);
        }

        ImGui::Spacing();
        if (ImGui::CollapsingHeader("TRAFFIC SIGNALS", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_signal_status(signal_controllers, controller_type, ai_model_enabled, action);
        }
    }
    ImGui::End();
}

void Renderer::draw_emergency_priority_panel(
    const EmergencyCorridorStatus* emergency_status,
    SimControlAction&              action) {
    ImGui::TextColored(ImVec4(1.00f, 0.40f, 0.40f, 1.00f), "EMERGENCY VEHICLE DISPATCH");
    ImGui::Separator();

    const float avail_w = ImGui::GetContentRegionAvail().x;
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.20f, 0.20f, 1.00f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.30f, 0.30f, 1.00f));
    if (ImGui::Button("DISPATCH EMERGENCY VEHICLE", ImVec2(avail_w, 28.0f))) {
        action.request_trigger_emergency = true;
    }
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Deploy emergency vehicle with priority signal preemption");
    }

    if (emergency_status) {
        if (emergency_status->active) {
            ImGui::TextColored(ImVec4(1.00f, 0.85f, 0.20f, 1.00f),
                               "Active Response: %u vehicle(s)",
                               emergency_status->active_emergency_count);
            ImGui::TextColored(ImVec4(0.40f, 0.90f, 0.50f, 1.00f),
                               "Priority Corridor: %zu road(s), %zu signal(s)",
                               emergency_status->corridor_roads.size(),
                               emergency_status->preempted_intersections.size());
        } else {
            ImGui::TextColored(ImVec4(0.60f, 0.60f, 0.60f, 1.00f), "Corridor: Standby (Normal Traffic)");
        }

        if (emergency_status->completed_trips > 0) {
            if (ImGui::BeginTable("EmergStatsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("Completed Trips");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", emergency_status->completed_trips);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("Avg Travel Time");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f s", emergency_status->avg_travel_time_s);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("Avg Signal Delay");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f s", emergency_status->avg_wait_time_s);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("Avg Speed");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f km/h", emergency_status->avg_speed_mps * 3.6f);

                ImGui::EndTable();
            }
        }
    }
}

SimControlAction Renderer::draw_imgui(
    const SimState&                                                    state,
    const RoadNetwork&                                                 network,
    const std::vector<Vehicle>&                                        vehicles,
    float                                                              current_speed_scale,
    const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers,
    SignalControllerType                                               controller_type,
    RoutingStrategyType                                                active_routing,
    const EmergencyCorridorStatus*                                     emergency_status,
    bool                                                               ai_model_enabled,
    float                                                              traffic_density,
    const TrafficAwareRouting*                                         traffic_aware_router,
    const std::unordered_map<RoadId, uint32_t>*                        road_queues) {
    SimControlAction action;

    draw_simulation_panel(state, network, current_speed_scale, ai_model_enabled, traffic_density, action);
    draw_traffic_panel(state,
                       network,
                       vehicles,
                       signal_controllers,
                       controller_type,
                       active_routing,
                       emergency_status,
                       ai_model_enabled,
                       traffic_aware_router,
                       road_queues,
                       action);

    return action;
}

void Renderer::end_frame() {
    rlImGuiEnd();
    EndDrawing();
}

} // namespace syntraq
