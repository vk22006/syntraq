//
// src/signals/traffic_signal_controller.cpp
//

#include "syntraq/signals/traffic_signal_controller.hpp"
#include "syntraq/world/road_network.hpp"

namespace syntraq {

namespace {

const std::unordered_map<RoadId, TrafficLight> kEmptyLights{};
const std::vector<SignalPhase>                 kEmptyPhases{};

} // namespace

TrafficSignalController::TrafficSignalController()
    : impl_(std::make_shared<FixedTimeController>()) {}

TrafficSignalController::TrafficSignalController(std::shared_ptr<ITrafficSignalController> impl)
    : impl_(std::move(impl)) {
    if (!impl_) {
        impl_ = std::make_shared<FixedTimeController>();
    }
}

TrafficSignalController::TrafficSignalController(FixedTimeController controller)
    : impl_(std::make_shared<FixedTimeController>(std::move(controller))) {}

TrafficSignalController::TrafficSignalController(AdaptiveController controller)
    : impl_(std::make_shared<AdaptiveController>(std::move(controller))) {}

TrafficSignalController::TrafficSignalController(RLSignalController controller)
    : impl_(std::make_shared<RLSignalController>(std::move(controller))) {}

TrafficSignalController::TrafficSignalController(
    IntersectionId           intersection_id,
    std::vector<SignalPhase> phases,
    std::vector<RoadId>      all_incoming_roads)
    : impl_(std::make_shared<FixedTimeController>(
          intersection_id, std::move(phases), std::move(all_incoming_roads))) {}

// ── Simulation update ────────────────────────────────────────────────────────

void TrafficSignalController::tick(float dt) {
    if (impl_) {
        impl_->tick(dt);
    }
}

void TrafficSignalController::reset() {
    if (impl_) {
        impl_->reset();
    }
}

void TrafficSignalController::observe_traffic(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>&    wait_times) {
    if (impl_) {
        impl_->observe_traffic(queues, wait_times);
    }
}

// ── Queries ──────────────────────────────────────────────────────────────────

IntersectionId TrafficSignalController::intersection_id() const noexcept {
    return impl_ ? impl_->intersection_id() : kInvalidIntersectionId;
}

SignalControllerType TrafficSignalController::type() const noexcept {
    return impl_ ? impl_->type() : SignalControllerType::FixedTime;
}

std::string_view TrafficSignalController::type_name() const noexcept {
    return impl_ ? impl_->type_name() : "FixedTime";
}

SignalColor TrafficSignalController::signal_for_road(RoadId road_id) const noexcept {
    return impl_ ? impl_->signal_for_road(road_id) : SignalColor::Red;
}

bool TrafficSignalController::can_proceed(RoadId road_id) const noexcept {
    return impl_ ? impl_->can_proceed(road_id) : false;
}

const TrafficLight* TrafficSignalController::light(RoadId road_id) const noexcept {
    return impl_ ? impl_->light(road_id) : nullptr;
}

const std::unordered_map<RoadId, TrafficLight>& TrafficSignalController::lights() const noexcept {
    return impl_ ? impl_->lights() : kEmptyLights;
}

size_t TrafficSignalController::current_phase_index() const noexcept {
    return impl_ ? impl_->current_phase_index() : 0;
}

const SignalPhase* TrafficSignalController::current_phase() const noexcept {
    return impl_ ? impl_->current_phase() : nullptr;
}

PhaseStage TrafficSignalController::current_stage() const noexcept {
    return impl_ ? impl_->current_stage() : PhaseStage::AllRed;
}

float TrafficSignalController::stage_elapsed_s() const noexcept {
    return impl_ ? impl_->stage_elapsed_s() : 0.0f;
}

float TrafficSignalController::current_stage_duration() const noexcept {
    return impl_ ? impl_->current_stage_duration() : 0.0f;
}

float TrafficSignalController::stage_remaining_s() const noexcept {
    return impl_ ? impl_->stage_remaining_s() : 0.0f;
}

float TrafficSignalController::cycle_duration_s() const noexcept {
    return impl_ ? impl_->cycle_duration_s() : 0.0f;
}

size_t TrafficSignalController::phase_count() const noexcept {
    return impl_ ? impl_->phase_count() : 0;
}

const std::vector<SignalPhase>& TrafficSignalController::phases() const noexcept {
    return impl_ ? impl_->phases() : kEmptyPhases;
}

bool TrafficSignalController::is_valid() const noexcept {
    return impl_ ? impl_->is_valid() : false;
}

bool TrafficSignalController::validate(std::string* error_msg) const {
    return impl_ ? impl_->validate(error_msg) : false;
}

void TrafficSignalController::set_phases(std::vector<SignalPhase> phases,
                                         std::vector<RoadId>      all_incoming_roads) {
    if (impl_) {
        impl_->set_phases(std::move(phases), std::move(all_incoming_roads));
    }
}

void TrafficSignalController::set_phase_timings(size_t phase_idx,
                                                float  green_s,
                                                float  yellow_s,
                                                float  all_red_s) {
    if (impl_) {
        impl_->set_phase_timings(phase_idx, green_s, yellow_s, all_red_s);
    }
}

// ── Factory Helpers ──────────────────────────────────────────────────────────

TrafficSignalController TrafficSignalController::make_fixed_time(
    const RoadNetwork& network,
    IntersectionId     intersection_id,
    float              green_s,
    float              yellow_s,
    float              all_red_s) {
    return TrafficSignalController(
        FixedTimeController::make_fixed_time(network, intersection_id, green_s, yellow_s, all_red_s));
}

TrafficSignalController TrafficSignalController::make_adaptive(
    const RoadNetwork&       network,
    IntersectionId           intersection_id,
    AdaptiveControllerConfig config,
    float                    yellow_s,
    float                    all_red_s) {
    return TrafficSignalController(
        AdaptiveController::make_adaptive(network, intersection_id, config, yellow_s, all_red_s));
}

TrafficSignalController TrafficSignalController::make_rl(
    const RoadNetwork& network,
    IntersectionId     intersection_id,
    RLControllerConfig config) {
    return TrafficSignalController(
        RLSignalController::make_rl(network, intersection_id, config));
}

} // namespace syntraq
