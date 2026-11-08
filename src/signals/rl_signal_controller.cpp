//
// src/signals/rl_signal_controller.cpp
//

#include "syntraq/signals/rl_signal_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <cmath>

namespace syntraq {

// ── Constructor ──────────────────────────────────────────────────────────────

RLSignalController::RLSignalController(
    IntersectionId           intersection_id,
    std::vector<SignalPhase> phases,
    std::vector<RoadId>      all_incoming_roads,
    RLControllerConfig       config)
    : intersection_id_(intersection_id)
    , phases_(std::move(phases))
    , all_incoming_roads_(std::move(all_incoming_roads))
    , current_phase_idx_(0)
    , current_stage_(PhaseStage::Green)
    , stage_elapsed_s_(0.0f)
    , config_(config)
    , switch_requested_(false) {

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

void RLSignalController::tick(float dt) {
    if (phases_.empty() || dt <= 0.0f) return;

    stage_elapsed_s_ += dt;

    if (current_stage_ == PhaseStage::Green) {
        constexpr float kEps = 1e-4f;
        const bool should_switch = (switch_requested_ && (stage_elapsed_s_ + kEps) >= config_.min_green_s)
                                 || ((stage_elapsed_s_ + kEps) >= config_.max_green_s);

        if (should_switch) {
            switch_requested_ = false;
            const float yellow_dur = (current_phase_idx_ < phases_.size())
                                   ? phases_[current_phase_idx_].yellow_duration_s
                                   : config_.yellow_s;
            const float red_dur    = (current_phase_idx_ < phases_.size())
                                   ? phases_[current_phase_idx_].all_red_duration_s
                                   : config_.all_red_s;

            if (yellow_dur > 0.0f) {
                current_stage_   = PhaseStage::Yellow;
                stage_elapsed_s_ = 0.0f;
            } else if (red_dur > 0.0f) {
                current_stage_   = PhaseStage::AllRed;
                stage_elapsed_s_ = 0.0f;
            } else {
                current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
                current_stage_     = PhaseStage::Green;
                stage_elapsed_s_   = 0.0f;
            }
        }
    } else if (current_stage_ == PhaseStage::Yellow) {
        const float yellow_dur = (current_phase_idx_ < phases_.size())
                               ? phases_[current_phase_idx_].yellow_duration_s
                               : config_.yellow_s;
        if (stage_elapsed_s_ >= yellow_dur) {
            stage_elapsed_s_ = 0.0f;
            const float red_dur = (current_phase_idx_ < phases_.size())
                                ? phases_[current_phase_idx_].all_red_duration_s
                                : config_.all_red_s;
            if (red_dur > 0.0f) {
                current_stage_ = PhaseStage::AllRed;
            } else {
                current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
                current_stage_     = PhaseStage::Green;
            }
        }
    } else if (current_stage_ == PhaseStage::AllRed) {
        const float red_dur = (current_phase_idx_ < phases_.size())
                            ? phases_[current_phase_idx_].all_red_duration_s
                            : config_.all_red_s;
        if (stage_elapsed_s_ >= red_dur) {
            stage_elapsed_s_   = 0.0f;
            current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
            current_stage_     = PhaseStage::Green;
        }
    }

    update_lights();
}

void RLSignalController::reset() {
    current_phase_idx_ = 0;
    current_stage_     = PhaseStage::Green;
    stage_elapsed_s_   = 0.0f;
    switch_requested_  = false;
    last_observed_queues_.clear();
    last_observed_waits_.clear();
    update_lights();
}

void RLSignalController::observe_traffic(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>&    wait_times) {
    last_observed_queues_ = queues;
    last_observed_waits_  = wait_times;
}

// ── RL Action Interface ──────────────────────────────────────────────────────

void RLSignalController::maintain_phase() noexcept {
    switch_requested_ = false;
}

bool RLSignalController::switch_phase() noexcept {
    if (current_stage_ == PhaseStage::Green) {
        constexpr float kEps = 1e-4f;
        if ((stage_elapsed_s_ + kEps) >= config_.min_green_s) {
            switch_requested_ = true;
            return true;
        }
    }
    return false;
}

// ── Queries ──────────────────────────────────────────────────────────────────

SignalColor RLSignalController::signal_for_road(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    return (it != lights_.end()) ? it->second.color : SignalColor::Red;
}

bool RLSignalController::can_proceed(RoadId road_id) const noexcept {
    return signal_for_road(road_id) == SignalColor::Green;
}

const TrafficLight* RLSignalController::light(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    return (it != lights_.end()) ? &it->second : nullptr;
}

const SignalPhase* RLSignalController::current_phase() const noexcept {
    if (current_phase_idx_ < phases_.size()) {
        return &phases_[current_phase_idx_];
    }
    return nullptr;
}

float RLSignalController::current_stage_duration() const noexcept {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        return 0.0f;
    }
    const auto& phase = phases_[current_phase_idx_];
    switch (current_stage_) {
        case PhaseStage::Green:  return config_.max_green_s;
        case PhaseStage::Yellow: return (phase.yellow_duration_s > 0.0f) ? phase.yellow_duration_s : config_.yellow_s;
        case PhaseStage::AllRed: return (phase.all_red_duration_s > 0.0f) ? phase.all_red_duration_s : config_.all_red_s;
    }
    return 0.0f;
}

float RLSignalController::stage_remaining_s() const noexcept {
    const float dur = current_stage_duration();
    return std::max(0.0f, dur - stage_elapsed_s_);
}

float RLSignalController::cycle_duration_s() const noexcept {
    float total = 0.0f;
    for (const auto& p : phases_) {
        total += p.green_duration_s + p.yellow_duration_s + p.all_red_duration_s;
    }
    return total;
}

void RLSignalController::set_phases(std::vector<SignalPhase> phases,
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

    current_phase_idx_ = 0;
    current_stage_     = PhaseStage::Green;
    stage_elapsed_s_   = 0.0f;
    switch_requested_  = false;
    update_lights();
}

void RLSignalController::set_phase_timings(size_t phase_idx,
                                           float  green_s,
                                           float  yellow_s,
                                           float  all_red_s) {
    if (phase_idx < phases_.size()) {
        phases_[phase_idx].green_duration_s   = green_s;
        phases_[phase_idx].yellow_duration_s  = yellow_s;
        phases_[phase_idx].all_red_duration_s = all_red_s;
    }
}

bool RLSignalController::is_valid() const noexcept {
    return validate(nullptr);
}

bool RLSignalController::validate(std::string* error_msg) const {
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

    if (config_.min_green_s <= 0.0f) {
        if (error_msg) *error_msg = "min_green_s must be positive.";
        return false;
    }
    if (config_.max_green_s < config_.min_green_s) {
        if (error_msg) *error_msg = "max_green_s cannot be less than min_green_s.";
        return false;
    }

    return true;
}

// ── Private update ───────────────────────────────────────────────────────────

void RLSignalController::update_lights() {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        for (auto& [_, light_obj] : lights_) {
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

RLSignalController RLSignalController::make_rl(
    const RoadNetwork& network,
    IntersectionId     intersection_id,
    RLControllerConfig config) {

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
        phases.push_back(SignalPhase{ "North-South", ns_roads, config.min_green_s, config.yellow_s, config.all_red_s });
        phases.push_back(SignalPhase{ "East-West",   ew_roads, config.min_green_s, config.yellow_s, config.all_red_s });
    } else if (!ns_roads.empty()) {
        phases.push_back(SignalPhase{ "North-South", ns_roads, config.min_green_s, config.yellow_s, config.all_red_s });
    } else if (!ew_roads.empty()) {
        phases.push_back(SignalPhase{ "East-West",   ew_roads, config.min_green_s, config.yellow_s, config.all_red_s });
    } else {
        phases.push_back(SignalPhase{ "All-Incoming", incoming_roads, config.min_green_s, config.yellow_s, config.all_red_s });
    }

    return RLSignalController(intersection_id, std::move(phases), incoming_roads, config);
}

} // namespace syntraq
