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

#include <string>

namespace syntraq {

// ── Construction / destruction ──────────────────────────────────────────────

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

// ── Public interface ────────────────────────────────────────────────────────

bool Renderer::should_close() const {
    return WindowShouldClose();
}

void Renderer::render_frame(const SimState& state) {
    begin_frame();
    draw_scene(state);
    draw_imgui(state);
    end_frame();
}

// ── Private helpers ─────────────────────────────────────────────────────────

void Renderer::begin_frame() {
    BeginDrawing();
    ClearBackground({ 15, 15, 20, 255 }); // Dark near-black background
    rlImGuiBegin();
}

void Renderer::draw_scene(const SimState& state) {
    // ── Milestone 1: placeholder scene ──────────────────────────────────
    // Draw a simple animated grid to show the window is alive.
    // Road network rendering wires in during Milestone 2+.

    const int W = GetScreenWidth();
    const int H = GetScreenHeight();

    // Subtle grid
    const int cell = 80;
    const Color gridColor = { 35, 40, 55, 255 };
    for (int x = 0; x < W; x += cell)
        DrawLine(x, 0, x, H, gridColor);
    for (int y = 0; y < H; y += cell)
        DrawLine(0, y, W, y, gridColor);

    // Centre logo / placeholder text
    const char* title = "SyntraQ";
    const int   fontSize = 48;
    const int   tw = MeasureText(title, fontSize);
    DrawText(title, (W - tw) / 2, H / 2 - 80, fontSize,
             { 80, 200, 255, 255 });

    const char* sub = "Urban Traffic Simulation & Optimization";
    const int   subSize = 20;
    const int   sw = MeasureText(sub, subSize);
    DrawText(sub, (W - sw) / 2, H / 2 - 20, subSize,
             { 140, 160, 180, 200 });

    // Tick counter (bottom-left)
    std::string tickStr =
        "Tick: " + std::to_string(state.tick) +
        "   Elapsed: " + std::to_string(static_cast<int>(state.elapsed_s)) + "s";
    DrawText(tickStr.c_str(), 16, H - 36, 18, { 100, 180, 100, 255 });
}

void Renderer::draw_imgui(const SimState& state) {
    // ── Debug overlay (Milestone 1) ──────────────────────────────────────
    ImGui::SetNextWindowPos({ 10.0f, 10.0f }, ImGuiCond_Once);
    ImGui::SetNextWindowSize({ 280.0f, 130.0f }, ImGuiCond_Once);
    ImGui::Begin("SyntraQ Debug", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);

    ImGui::Text("Milestone 1 — Foundation");
    ImGui::Separator();
    ImGui::Text("Tick   : %llu", static_cast<unsigned long long>(state.tick));
    ImGui::Text("Elapsed: %.2f s", state.elapsed_s);
    ImGui::Text("FPS    : %.1f", GetFPS());
    ImGui::Text("Status : %s", state.running ? "Running" : "Paused");

    ImGui::End();
}

void Renderer::end_frame() {
    rlImGuiEnd();
    EndDrawing();
}

} // namespace syntraq
