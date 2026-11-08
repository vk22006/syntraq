//
// src/signals/adaptive_controller.cpp
//

#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace syntraq {

namespace {

std::string format_float(float val, int precision = 1) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << val;
    return ss.str();
}

} // namespace

// ── Constructor ──────────────────────────────────────────────────────────────

AdaptiveController::AdaptiveController(
    IntersectionId           intersection_id,
    std::vector<SignalPhase> phases,
    std::vector<RoadId>      all_incoming_roads,
    AdaptiveControllerConfig config)
    : intersection_id_(intersection_id)
    , phases_(std::move(phases))
    , all_incoming_roads_(std::move(all_incoming_roads))
    , config_(config)
    , current_phase_idx_(0)
    , current_stage_(PhaseStage::Green)
    , stage_elapsed_s_(0.0f)
    , last_decision_(AdaptiveDecision::Hold)
    , last_decision_reason_("Initial green phase started") {

    if (all_incoming_roads_.empty()) {
        for (const auto& p : phases_) {
            for (RoadId r : p.green_roads) {
                if (std::find(all_incoming_roads_.begin(), all_incoming_roads_.end(), r) == all_incoming_roads_.end()) {
                    all_incoming_roads_.push_back(r);
                }
            }
        }
    }

    for (RoadId r : all_incoming_roads_) {
        lights_[r] = TrafficLight{ r, intersection_id_, SignalColor::Red };
    }

    update_lights();
}

// ── ITrafficSignalController Overrides ─────────────────────────────────────

void AdaptiveController::observe_traffic(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>&    wait_times) {
    observed_queues_     = queues;
    observed_wait_times_ = wait_times;
}

void AdaptiveController::tick(float dt) {
    if (phases_.empty() || dt <= 0.0f) return;

    if (current_stage_ == PhaseStage::Green) {
        stage_elapsed_s_ += dt;
        evaluate_green_decision();
    } else {
        stage_elapsed_s_ += dt;
        const float dur = current_stage_duration();
        if (stage_elapsed_s_ >= dur) {
            stage_elapsed_s_ = std::max(0.0f, stage_elapsed_s_ - dur);
            const auto& phase = phases_[current_phase_idx_];

            if (current_stage_ == PhaseStage::Yellow) {
                if (phase.all_red_duration_s > 0.0f) {
                    current_stage_ = PhaseStage::AllRed;
                } else {
                    current_phase_idx_    = (current_phase_idx_ + 1) % phases_.size();
                    current_stage_        = PhaseStage::Green;
                    last_decision_        = AdaptiveDecision::Hold;
                    last_decision_reason_ = "Phase transition to green";
                }
            } else if (current_stage_ == PhaseStage::AllRed) {
                current_phase_idx_    = (current_phase_idx_ + 1) % phases_.size();
                current_stage_        = PhaseStage::Green;
                last_decision_        = AdaptiveDecision::Hold;
                last_decision_reason_ = "Clearance complete; starting green";
            }
        }
    }

    update_lights();
}

void AdaptiveController::reset() {
    current_phase_idx_    = 0;
    current_stage_        = PhaseStage::Green;
    stage_elapsed_s_      = 0.0f;
    last_decision_        = AdaptiveDecision::Hold;
    last_decision_reason_ = "Controller reset";
    extensions_count_     = 0;
    shortenings_count_    = 0;
    switches_count_       = 0;
    observed_queues_.clear();
    observed_wait_times_.clear();
    update_lights();
}

// ── Decision Logic ───────────────────────────────────────────────────────────

void AdaptiveController::evaluate_green_decision() {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) return;
    const auto& phase = phases_[current_phase_idx_];

    // 1. Calculate active queue on current green approach roads
    uint32_t current_queue = 0;
    for (RoadId r : phase.green_roads) {
        const auto it = observed_queues_.find(r);
        if (it != observed_queues_.end()) {
            current_queue += it->second;
        }
    }

    // 2. Calculate opposing queued vehicles and max waiting time
    uint32_t opposing_queue    = 0;
    float    max_opposing_wait = 0.0f;
    for (RoadId r : all_incoming_roads_) {
        if (!phase.contains_road(r)) {
            const auto q_it = observed_queues_.find(r);
            if (q_it != observed_queues_.end()) {
                opposing_queue += q_it->second;
            }
            const auto w_it = observed_wait_times_.find(r);
            if (w_it != observed_wait_times_.end() && w_it->second > max_opposing_wait) {
                max_opposing_wait = w_it->second;
            }
        }
    }

    // ── Rule 1: Mandatory Minimum Green Constraint ───────────────────────────
    if (stage_elapsed_s_ < config_.min_green_s) {
        last_decision_ = AdaptiveDecision::Hold;
        last_decision_reason_ = "Serving minimum green (" +
                                format_float(stage_elapsed_s_) + "s / " +
                                format_float(config_.min_green_s) + "s)";
        return;
    }

    // ── Rule 2: Shorten Phase (Early Gap-Out) ────────────────────────────────
    if (stage_elapsed_s_ < config_.nominal_green_s) {
        if (current_queue == 0 && opposing_queue > 0) {
            last_decision_ = AdaptiveDecision::ShortenPhase;
            shortenings_count_++;
            last_decision_reason_ = "Shortening green early: queue cleared (0 veh) while opposing queue (" +
                                    std::to_string(opposing_queue) + " veh) waiting at " +
                                    format_float(stage_elapsed_s_) + "s";
            transition_to_yellow();
            return;
        }

        last_decision_ = AdaptiveDecision::Hold;
        last_decision_reason_ = "Serving nominal green (" +
                                format_float(stage_elapsed_s_) + "s / " +
                                format_float(config_.nominal_green_s) + "s, queue: " +
                                std::to_string(current_queue) + ")";
        return;
    }

    // ── Rule 3: Max Green Cap Reached ────────────────────────────────────────
    if (stage_elapsed_s_ >= config_.max_green_s) {
        last_decision_ = AdaptiveDecision::SwitchPhase;
        switches_count_++;
        last_decision_reason_ = "Switching phase: max green limit reached (" +
                                format_float(config_.max_green_s) + "s)";
        transition_to_yellow();
        return;
    }

    // ── Rule 4: Starvation Prevention (Opposing Wait Exceeds Threshold) ───────
    if (max_opposing_wait >= config_.max_wait_threshold_s && opposing_queue > 0) {
        last_decision_ = AdaptiveDecision::SwitchPhase;
        switches_count_++;
        last_decision_reason_ = "Switching phase: opposing wait (" +
                                format_float(max_opposing_wait) + "s) reached starvation threshold (" +
                                format_float(config_.max_wait_threshold_s) + "s)";
        transition_to_yellow();
        return;
    }

    // ── Rule 5: Demand-Responsive Green Extension ─────────────────────────────
    const bool current_has_demand = (current_queue > 0);
    const bool higher_or_equal_demand = (current_queue * config_.queue_priority_ratio >= opposing_queue);

    if (current_has_demand && (higher_or_equal_demand || opposing_queue == 0)) {
        last_decision_ = AdaptiveDecision::ExtendGreen;
        extensions_count_++;
        last_decision_reason_ = "Extending green: active current queue (" +
                                std::to_string(current_queue) + " veh) >= opposing (" +
                                std::to_string(opposing_queue) + " veh, max wait: " +
                                format_float(max_opposing_wait) + "s)";
        return;
    }

    // ── Rule 6: Normal Expiration / Switch to Higher Opposing Demand ──────────
    last_decision_ = AdaptiveDecision::SwitchPhase;
    switches_count_++;
    last_decision_reason_ = "Switching phase: opposing demand (" +
                            std::to_string(opposing_queue) + " veh) exceeds current (" +
                            std::to_string(current_queue) + " veh)";
    transition_to_yellow();
}

void AdaptiveController::transition_to_yellow() {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) return;
    const auto& phase = phases_[current_phase_idx_];

    stage_elapsed_s_ = 0.0f;
    if (phase.yellow_duration_s > 0.0f) {
        current_stage_ = PhaseStage::Yellow;
    } else if (phase.all_red_duration_s > 0.0f) {
        current_stage_ = PhaseStage::AllRed;
    } else {
        current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
        current_stage_     = PhaseStage::Green;
    }
}

// ── Queries ──────────────────────────────────────────────────────────────────

SignalColor AdaptiveController::signal_for_road(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return it->second.color;
    }
    return SignalColor::Red;
}

bool AdaptiveController::can_proceed(RoadId road_id) const noexcept {
    return signal_for_road(road_id) == SignalColor::Green;
}

const TrafficLight* AdaptiveController::light(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return &it->second;
    }
    return nullptr;
}

const SignalPhase* AdaptiveController::current_phase() const noexcept {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        return nullptr;
    }
    return &phases_[current_phase_idx_];
}

float AdaptiveController::current_stage_duration() const noexcept {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) {
        return 0.0f;
    }
    const auto& p = phases_[current_phase_idx_];
    switch (current_stage_) {
        case PhaseStage::Green:  return config_.nominal_green_s;
        case PhaseStage::Yellow: return p.yellow_duration_s;
        case PhaseStage::AllRed: return p.all_red_duration_s;
    }
    return 0.0f;
}

float AdaptiveController::stage_remaining_s() const noexcept {
    return std::max(0.0f, current_stage_duration() - stage_elapsed_s_);
}

float AdaptiveController::cycle_duration_s() const noexcept {
    float total = 0.0f;
    for (const auto& p : phases_) {
        total += config_.nominal_green_s + p.yellow_duration_s + p.all_red_duration_s;
    }
    return total;
}

bool AdaptiveController::is_valid() const noexcept {
    return validate(nullptr);
}

bool AdaptiveController::validate(std::string* error_msg) const {
    if (intersection_id_ == kInvalidIntersectionId) {
        if (error_msg) *error_msg = "Invalid intersection ID.";
        return false;
    }
    if (phases_.empty()) {
        if (error_msg) *error_msg = "Controller has no signal phases configured.";
        return false;
    }
    if (config_.min_green_s <= 0.0f) {
        if (error_msg) *error_msg = "min_green_s must be positive.";
        return false;
    }
    if (config_.max_green_s < config_.min_green_s) {
        if (error_msg) *error_msg = "max_green_s cannot be less than min_green_s.";
        return false;
    }

    for (size_t i = 0; i < phases_.size(); ++i) {
        const auto& p = phases_[i];
        if (p.green_roads.empty()) {
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " has no green roads.";
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

    return true;
}

void AdaptiveController::set_phases(std::vector<SignalPhase> phases,
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

void AdaptiveController::set_phase_timings(size_t phase_idx,
                                           float  green_s,
                                           float  yellow_s,
                                           float  all_red_s) {
    if (phase_idx >= phases_.size()) return;
    phases_[phase_idx].green_duration_s   = green_s;
    phases_[phase_idx].yellow_duration_s  = yellow_s;
    phases_[phase_idx].all_red_duration_s = all_red_s;
}

void AdaptiveController::update_lights() {
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

AdaptiveController AdaptiveController::make_adaptive(
    const RoadNetwork&       network,
    IntersectionId           intersection_id,
    AdaptiveControllerConfig config,
    float                    yellow_s,
    float                    all_red_s) {

    const Intersection* center = network.intersection(intersection_id);
    if (!center) {
        return {};
    }

    const std::vector<RoadId> incoming_roads = network.incoming_roads(intersection_id);
    if (incoming_roads.empty()) {
        return {};
    }

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
        phases.push_back(SignalPhase{ "North-South", ns_roads, config.nominal_green_s, yellow_s, all_red_s });
        phases.push_back(SignalPhase{ "East-West",   ew_roads, config.nominal_green_s, yellow_s, all_red_s });
    } else if (!ns_roads.empty()) {
        phases.push_back(SignalPhase{ "North-South", ns_roads, config.nominal_green_s, yellow_s, all_red_s });
    } else if (!ew_roads.empty()) {
        phases.push_back(SignalPhase{ "East-West",   ew_roads, config.nominal_green_s, yellow_s, all_red_s });
    } else {
        phases.push_back(SignalPhase{ "All-Incoming", incoming_roads, config.nominal_green_s, yellow_s, all_red_s });
    }

    return AdaptiveController(intersection_id, std::move(phases), incoming_roads, config);
}

} // namespace syntraq
