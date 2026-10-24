#pragma once

//
// syntraq/signals/traffic_signal_controller.hpp
//
// TrafficSignalController — Polymorphic envelope / handle for traffic signal controllers.
// Wraps an underlying ITrafficSignalController implementation (FixedTimeController,
// AdaptiveController, etc.) while providing copyable, value-semantic storage in maps
// and containers without object slicing.
//

#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/signals/fixed_time_controller.hpp"
#include "syntraq/signals/rl_signal_controller.hpp"
#include "syntraq/signals/signal_controller_interface.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace syntraq {

class RoadNetwork;

class TrafficSignalController {
public:
    /// Default constructor creates an empty/invalid controller handle
    TrafficSignalController();

    /// Construct wrapping a shared implementation
    explicit TrafficSignalController(std::shared_ptr<ITrafficSignalController> impl);

    /// Converting constructor from FixedTimeController
    TrafficSignalController(FixedTimeController controller); // NOLINT

    /// Converting constructor from AdaptiveController
    TrafficSignalController(AdaptiveController controller); // NOLINT

    /// Converting constructor from RLSignalController
    TrafficSignalController(RLSignalController controller); // NOLINT

    /// Legacy fixed-time constructor (constructs FixedTimeController internally)
    TrafficSignalController(IntersectionId           intersection_id,
                            std::vector<SignalPhase> phases,
                            std::vector<RoadId>      all_incoming_roads = {});

    // ── Simulation update ──────────────────────────────────────────────────

    void tick(float dt);
    void reset();

    /// Observe incoming approach queues and wait times (passed to adaptive controllers)
    void observe_traffic(const std::unordered_map<RoadId, uint32_t>& queues,
                         const std::unordered_map<RoadId, float>&    wait_times = {});

    // ── Queries ────────────────────────────────────────────────────────────

    [[nodiscard]] IntersectionId intersection_id() const noexcept;
    [[nodiscard]] SignalControllerType type() const noexcept;
    [[nodiscard]] std::string_view type_name() const noexcept;

    [[nodiscard]] SignalColor signal_for_road(RoadId road_id) const noexcept;
    [[nodiscard]] bool can_proceed(RoadId road_id) const noexcept;
    [[nodiscard]] const TrafficLight* light(RoadId road_id) const noexcept;
    [[nodiscard]] const std::unordered_map<RoadId, TrafficLight>& lights() const noexcept;

    [[nodiscard]] size_t current_phase_index() const noexcept;
    [[nodiscard]] const SignalPhase* current_phase() const noexcept;
    [[nodiscard]] PhaseStage current_stage() const noexcept;
    [[nodiscard]] float stage_elapsed_s() const noexcept;
    [[nodiscard]] float current_stage_duration() const noexcept;
    [[nodiscard]] float stage_remaining_s() const noexcept;
    [[nodiscard]] float cycle_duration_s() const noexcept;
    [[nodiscard]] size_t phase_count() const noexcept;
    [[nodiscard]] const std::vector<SignalPhase>& phases() const noexcept;

    [[nodiscard]] bool is_valid() const noexcept;
    bool validate(std::string* error_msg = nullptr) const;

    void set_phases(std::vector<SignalPhase> phases,
                    std::vector<RoadId>      all_incoming_roads = {});

    void set_phase_timings(size_t phase_idx,
                           float  green_s,
                           float  yellow_s,
                           float  all_red_s);

    // ── Implementation Access & Casting ────────────────────────────────────

    [[nodiscard]] ITrafficSignalController* get() noexcept {
        return impl_.get();
    }

    [[nodiscard]] const ITrafficSignalController* get() const noexcept {
        return impl_.get();
    }

    template <typename T>
    [[nodiscard]] T* as() noexcept {
        return dynamic_cast<T*>(impl_.get());
    }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept {
        return dynamic_cast<const T*>(impl_.get());
    }

    template <typename T>
    [[nodiscard]] bool is() const noexcept {
        return dynamic_cast<const T*>(impl_.get()) != nullptr;
    }

    // ── Factory Helpers ────────────────────────────────────────────────────

    static TrafficSignalController make_fixed_time(
        const RoadNetwork& network,
        IntersectionId     intersection_id,
        float              green_s   = 10.0f,
        float              yellow_s  =  3.0f,
        float              all_red_s =  2.0f);

    static TrafficSignalController make_adaptive(
        const RoadNetwork&       network,
        IntersectionId           intersection_id,
        AdaptiveControllerConfig config = {},
        float                    yellow_s  = 3.0f,
        float                    all_red_s = 2.0f);

    static TrafficSignalController make_rl(
        const RoadNetwork& network,
        IntersectionId     intersection_id,
        RLControllerConfig config = {});

private:
    std::shared_ptr<ITrafficSignalController> impl_;
};

} // namespace syntraq
