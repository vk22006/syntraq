//
// src/signals/traffic_signal_controller.cpp
//

#include "syntraq/signals/traffic_signal_controller.h"
#include "syntraq/world/road_network.h"

#include <algorithm>
#include <cmath>

namespace syntraq {

// ── Constructor ──────────────────────────────────────────────────────────────

TrafficSignalController::TrafficSignalController(
    IntersectionId           intersection_id,
    std::vector<SignalPhase> phases,
    std::vector<RoadId>      all_incoming_roads)
    : intersection_id_(intersection_id)
    , phases_(std::move(phases))
    , all_incoming_roads_(std::move(all_incoming_roads))
    , current_phase_idx_(0)
    , current_stage_(PhaseStage::Green)
    , stage_elapsed_s_(0.0f) {

    // If all_incoming_roads was not explicitly supplied, infer from phases
    if (all_incoming_roads_.empty()) {
        for (const auto& p : phases_) {
            for (RoadId r : p.green_roads) {
                if (std::find(all_incoming_roads_.begin(), all_incoming_roads_.end(), r) == all_incoming_roads_.end()) {
                    all_incoming_roads_.push_back(r);
                }
            }
        }
    }

    // Initialize lights map for all incoming roads
    for (RoadId r : all_incoming_roads_) {
        lights_[r] = TrafficLight{ r, intersection_id_, SignalColor::Red };
    }

    update_lights();
}

// ── Simulation update ────────────────────────────────────────────────────────

void TrafficSignalController::tick(float dt) {
    if (phases_.empty() || dt <= 0.0f) return;

    stage_elapsed_s_ += dt;

    constexpr float kEpsilon = 1e-5f;

    // Advance state machine for as many transitions as elapsed time covers
    while (true) {
        const float stage_dur = current_stage_duration();
        if ((stage_elapsed_s_ + kEpsilon) < stage_dur || stage_dur <= 0.0f) {
            break;
        }

        stage_elapsed_s_ = std::max(0.0f, stage_elapsed_s_ - stage_dur);
        const auto& phase = phases_[current_phase_idx_];

        switch (current_stage_) {
            case PhaseStage::Green:
                if (phase.yellow_duration_s > 0.0f) {
                    current_stage_ = PhaseStage::Yellow;
                } else if (phase.all_red_duration_s > 0.0f) {
                    current_stage_ = PhaseStage::AllRed;
                } else {
                    current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
                    current_stage_     = PhaseStage::Green;
                }
                break;

            case PhaseStage::Yellow:
                if (phase.all_red_duration_s > 0.0f) {
                    current_stage_ = PhaseStage::AllRed;
                } else {
                    current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
                    current_stage_     = PhaseStage::Green;
                }
                break;

            case PhaseStage::AllRed:
                current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
                current_stage_     = PhaseStage::Green;
                break;
        }
    }

    update_lights();
}

void TrafficSignalController::reset() {
    current_phase_idx_ = 0;
    current_stage_     = PhaseStage::Green;
    stage_elapsed_s_   = 0.0f;
    update_lights();
}

// ── Queries ──────────────────────────────────────────────────────────────────

SignalColor TrafficSignalController::signal_for_road(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return it->second.color;
    }
    return SignalColor::Red;
}

bool TrafficSignalController::can_proceed(RoadId road_id) const noexcept {
    return signal_for_road(road_id) == SignalColor::Green;
}

const TrafficLight* TrafficSignalController::light(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return &it->second;
    }
    return nullptr;
}

const SignalPhase* TrafficSignalController::current_phase() const noexcept {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        return nullptr;
    }
    return &phases_[current_phase_idx_];
}

float TrafficSignalController::current_stage_duration() const noexcept {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        return 0.0f;
    }
    const auto& p = phases_[current_phase_idx_];
    switch (current_stage_) {
        case PhaseStage::Green:  return p.green_duration_s;
        case PhaseStage::Yellow: return p.yellow_duration_s;
        case PhaseStage::AllRed: return p.all_red_duration_s;
    }
    return 0.0f;
}

float TrafficSignalController::stage_remaining_s() const noexcept {
    return std::max(0.0f, current_stage_duration() - stage_elapsed_s_);
}

float TrafficSignalController::cycle_duration_s() const noexcept {
    float total = 0.0f;
    for (const auto& p : phases_) {
        total += p.total_duration_s();
    }
    return total;
}

// ── Configuration & Validation ───────────────────────────────────────────────

void TrafficSignalController::set_phases(std::vector<SignalPhase> phases,
                                         std::vector<RoadId>      all_incoming_roads) {
    phases_             = std::move(phases);
    all_incoming_roads_ = std::move(all_incoming_roads);

    if (all_incoming_roads_.empty()) {
        for (const auto& p : phases_) {
            for (RoadId r : p.green_roads) {
                if (std::find(all_incoming_roads_.begin(), all_incoming_roads_.end(), r) == all_incoming_roads_.end()) {
                    all_incoming_roads_.push_back(r);
                }
            }
        }
    }

    lights_.clear();
    for (RoadId r : all_incoming_roads_) {
        lights_[r] = TrafficLight{ r, intersection_id_, SignalColor::Red };
    }

    reset();
}

void TrafficSignalController::set_phase_timings(size_t phase_idx,
                                                float  green_s,
                                                float  yellow_s,
                                                float  all_red_s) {
    if (phase_idx >= phases_.size()) return;
    phases_[phase_idx].green_duration_s   = green_s;
    phases_[phase_idx].yellow_duration_s  = yellow_s;
    phases_[phase_idx].all_red_duration_s = all_red_s;
}

bool TrafficSignalController::is_valid() const noexcept {
    return validate(nullptr);
}

bool TrafficSignalController::validate(std::string* error_msg) const {
    if (intersection_id_ == kInvalidIntersectionId) {
        if (error_msg) *error_msg = "Invalid intersection ID.";
        return false;
    }
    if (phases_.empty()) {
        if (error_msg) *error_msg = "Controller has no signal phases configured.";
        return false;
    }

    for (size_t i = 0; i < phases_.size(); ++i) {
        const auto& p = phases_[i];
        if (p.green_roads.empty()) {
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " has no green roads.";
            return false;
        }
        if (p.green_duration_s <= 0.0f) {
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " green duration must be positive.";
            return false;
        }
        if (p.yellow_duration_s < 0.0f) {
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " yellow duration cannot be negative.";
            return false;
        }
        if (p.all_red_duration_s < 0.0f) {
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " all-red duration cannot be negative.";
            return false;
        }
    }

    if (cycle_duration_s() <= 0.0f) {
        if (error_msg) *error_msg = "Total cycle duration must be positive.";
        return false;
    }

    return true;
}

// ── Private helpers ──────────────────────────────────────────────────────────

void TrafficSignalController::update_lights() {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        for (auto& [road_id, light_obj] : lights_) {
            light_obj.color = SignalColor::Red;
        }
        return;
    }

    const auto& phase = phases_[current_phase_idx_];

    for (auto& [road_id, light_obj] : lights_) {
        light_obj.road_id         = road_id;
        light_obj.intersection_id = intersection_id_;

        if (current_stage_ == PhaseStage::AllRed) {
            light_obj.color = SignalColor::Red;
        } else if (phase.contains_road(road_id)) {
            light_obj.color = (current_stage_ == PhaseStage::Green)
                            ? SignalColor::Green
                            : SignalColor::Yellow;
        } else {
            light_obj.color = SignalColor::Red;
        }
    }
}

// ── Factory Helpers ──────────────────────────────────────────────────────────

TrafficSignalController TrafficSignalController::make_fixed_time(
    const RoadNetwork& network,
    IntersectionId     intersection_id,
    float              green_s,
    float              yellow_s,
    float              all_red_s) {

    const Intersection* center = network.intersection(intersection_id);
    if (!center) {
        return {};
    }

    const std::vector<RoadId> incoming_roads = network.incoming_roads(intersection_id);
    if (incoming_roads.empty()) {
        return {};
    }

    // Split incoming roads by axis: North-South (vertical) vs East-West (horizontal)
    std::vector<RoadId> ns_roads;
    std::vector<RoadId> ew_roads;

    for (RoadId r_id : incoming_roads) {
        const Road* r = network.road(r_id);
        if (!r) continue;
        const Intersection* from_node = network.intersection(r->from);
        if (!from_node) continue;

        const float dx = std::abs(center->position.x - from_node->position.x);
        const float dy = std::abs(center->position.y - from_node->position.y);

        if (dy > dx) {
            ns_roads.push_back(r_id);
        } else {
            ew_roads.push_back(r_id);
        }
    }

    std::vector<SignalPhase> phases;
    if (!ns_roads.empty() && !ew_roads.empty()) {
        phases.push_back(SignalPhase{ "North-South", ns_roads, green_s, yellow_s, all_red_s });
        phases.push_back(SignalPhase{ "East-West",   ew_roads, green_s, yellow_s, all_red_s });
    } else if (!ns_roads.empty()) {
        phases.push_back(SignalPhase{ "North-South", ns_roads, green_s, yellow_s, all_red_s });
    } else if (!ew_roads.empty()) {
        phases.push_back(SignalPhase{ "East-West",   ew_roads, green_s, yellow_s, all_red_s });
    } else {
        phases.push_back(SignalPhase{ "All-Incoming", incoming_roads, green_s, yellow_s, all_red_s });
    }

    return TrafficSignalController(intersection_id, std::move(phases), incoming_roads);
}

} // namespace syntraq
