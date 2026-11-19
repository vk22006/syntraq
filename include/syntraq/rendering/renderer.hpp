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

#include "syntraq/core/config.hpp"
#include "syntraq/simulation/simulation.hpp"
#include "syntraq/world/road_network.hpp"
#include "syntraq/vehicles/vehicle.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace syntraq {

/// Visual and debug toggles for rendering.
struct DebugRenderFlags {
    bool show_intersection_ids    { false };
    bool show_road_ids            { false };
    bool show_lane_boundaries     { false };
    bool show_vehicle_vectors     { false };
    bool show_traffic_signals     { true };
    bool show_debug_statistics    { true };

    // Final Visualization overlays
    bool show_routes              { false }; ///< Visual flow arrows and connectivity paths
    bool show_queues              { true };  ///< Visual queue highlighting and approach delay tags
    bool show_intersection_phases { true };  ///< Visual junction phase indicators, countdowns & action badges
    bool show_predicted_congestion{ false }; ///< Real-time congestion heatmap overlay on road segments
    bool show_selected_paths      { true };  ///< Complete path polyline for selected vehicle
};

/// Simulation commands triggered from UI buttons or keyboard shortcuts.
struct SimControlAction {
    bool                                request_pause_toggle{ false };
    bool                                request_reset       { false };
    bool                                request_step        { false };
    std::optional<float>                request_speed_scale {};
    std::optional<TrafficScenario>      request_scenario    {};
    std::optional<float>                request_traffic_density {};
    std::optional<SignalControllerType> request_controller_type {};
    std::optional<RoutingStrategyType>  request_routing_strategy{};
    std::optional<bool>                 request_ai_enabled  {};
    bool                                request_trigger_emergency{ false };
    std::optional<bool>                 request_emergency_priority_toggle{};
    std::optional<bool>                 request_emergency_rerouting_toggle{};
    std::optional<VehicleId>            request_selected_vehicle{};
};

/// Forward declaration for emergency corridor telemetry
struct EmergencyCorridorStatus;

/// Forward declaration for traffic-aware routing engine
class TrafficAwareRouting;

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
    SimControlAction render_frame(
        const SimState&                                                    state,
        const RoadNetwork&                                                 network,
        const std::vector<Vehicle>&                                        vehicles,
        float                                                              current_speed_scale = 1.0f,
        const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers = nullptr,
        SignalControllerType                                               controller_type = SignalControllerType::FixedTime,
        RoutingStrategyType                                                active_routing = RoutingStrategyType::Static,
        const EmergencyCorridorStatus*                                     emergency_status = nullptr,
        bool                                                               ai_model_enabled = true,
        float                                                              traffic_density = 1.0f,
        const TrafficAwareRouting*                                         traffic_aware_router = nullptr,
        const std::unordered_map<RoadId, uint32_t>*                        road_queues = nullptr);

    /// Reset camera view to center and frame the road network with comfortable margins.
    void reset_camera(const RoadNetwork& network);

    /// Save current frame screenshot to file.
    void take_screenshot(const std::string& path) const;

    [[nodiscard]] const DebugRenderFlags& debug_flags() const noexcept { return debug_flags_; }
    [[nodiscard]]       DebugRenderFlags& debug_flags()       noexcept { return debug_flags_; }

    [[nodiscard]] VehicleId selected_vehicle_id() const noexcept { return selected_vehicle_id_; }
    void set_selected_vehicle_id(VehicleId id) noexcept { selected_vehicle_id_ = id; }

private:
    void update_camera(const RoadNetwork& network, const std::vector<Vehicle>& vehicles);
    void begin_frame();
    void draw_world_grid();
    void draw_network(const RoadNetwork& network);
    void draw_traffic_signals(
        const RoadNetwork&                                                 network,
        const std::unordered_map<IntersectionId, TrafficSignalController>& controllers);
    void draw_vehicles(const RoadNetwork& network,
                       const std::vector<Vehicle>& vehicles);

    // ── Visual Debugging Overlays in 2D World Space ─────────────────────────
    void draw_visual_debug_routes(const RoadNetwork& network);
    void draw_visual_debug_queues(
        const RoadNetwork&                          network,
        const std::unordered_map<RoadId, uint32_t>* road_queues,
        const std::vector<Vehicle>&                 vehicles);
    void draw_visual_debug_phases(
        const RoadNetwork&                                                 network,
        const std::unordered_map<IntersectionId, TrafficSignalController>& controllers);
    void draw_visual_debug_predicted_congestion(
        const RoadNetwork&                          network,
        const std::vector<Vehicle>&                 vehicles,
        const TrafficAwareRouting*                  router,
        const std::unordered_map<RoadId, uint32_t>* road_queues);
    void draw_visual_debug_selected_paths(
        const RoadNetwork&          network,
        const std::vector<Vehicle>& vehicles);

    SimControlAction draw_imgui(
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
        const std::unordered_map<RoadId, uint32_t>*                        road_queues);

    // ── Focused UI Panel Functions ─────────────────────────────────────────
    void draw_simulation_panel(
        const SimState&    state,
        const RoadNetwork& network,
        float              current_speed_scale,
        bool               ai_model_enabled,
        float              traffic_density,
        SimControlAction&  action);
    void draw_simulation_controls(
        const SimState&    state,
        const RoadNetwork& network,
        float              current_speed_scale,
        SimControlAction&  action);
    void draw_traffic_density_controls(
        float              traffic_density,
        SimControlAction&  action);
    void draw_ai_subsystem_controls(
        bool               ai_model_enabled,
        SimControlAction&  action);
    void draw_simulation_metrics(const SimState& state);
    void draw_debug_controls();

    void draw_traffic_panel(
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
        SimControlAction&                                                  action);
    void draw_signal_status(
        const std::unordered_map<IntersectionId, TrafficSignalController>* signal_controllers,
        SignalControllerType                                               controller_type,
        bool                                                               ai_model_enabled,
        SimControlAction&                                                  action);
    void draw_routing_strategy(
        RoutingStrategyType        active_routing,
        const TrafficAwareRouting* traffic_aware_router,
        SimControlAction&          action);
    void draw_selected_vehicle_inspector(
        const std::vector<Vehicle>& vehicles,
        const RoadNetwork&          network,
        SimControlAction&           action);
    void draw_emergency_priority_panel(
        const EmergencyCorridorStatus* emergency_status,
        SimControlAction&              action);
    void draw_traffic_information(
        const SimState&    state,
        const RoadNetwork& network,
        SimControlAction&  action);

    void draw_status_bar(const SimState& state, float current_speed_scale, bool ai_model_enabled);
    void end_frame();

    const Config&                 cfg_;
    DebugRenderFlags              debug_flags_{};
    std::unique_ptr<RenderCamera> camera_;
    bool                          camera_initialized_{ false };
    VehicleId                     selected_vehicle_id_{ kInvalidVehicleId };
    bool                          follow_selected_vehicle_{ false };
    int                           last_screen_w_{ 0 };
    int                           last_screen_h_{ 0 };
};

} // namespace syntraq
