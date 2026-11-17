//
// src/signals/adaptive_controller.cpp
//

#include "syntraq/signals/adaptive_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace syntraq {

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

    local_queues_.assign(all_incoming_roads_.size(), 0u);
    local_wait_times_.assign(all_incoming_roads_.size(), 0.0f);
    last_decision_reason_.reserve(160);

    for (RoadId r : all_incoming_roads_) {
        lights_[r] = TrafficLight{ r, intersection_id_, SignalColor::Red };
    }

    update_lights();
}

// ── ITrafficSignalController Overrides ─────────────────────────────────────

void AdaptiveController::observe_traffic(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>&    wait_times) {
    const size_t n = all_incoming_roads_.size();
    if (local_queues_.size() != n) {
        local_queues_.assign(n, 0u);
        local_wait_times_.assign(n, 0.0f);
    }

    if (queues.empty()) {
        std::fill(local_queues_.begin(), local_queues_.end(), 0u);
    } else {
        for (size_t i = 0; i < n; ++i) {
            const auto it = queues.find(all_incoming_roads_[i]);
            local_queues_[i] = (it != queues.end()) ? it->second : 0u;
        }
    }

    if (wait_times.empty()) {
        std::fill(local_wait_times_.begin(), local_wait_times_.end(), 0.0f);
    } else {
        for (size_t i = 0; i < n; ++i) {
            const auto it = wait_times.find(all_incoming_roads_[i]);
            local_wait_times_[i] = (it != wait_times.end()) ? it->second : 0.0f;
        }
    }
}

void AdaptiveController::tick(float dt) {
    if (phases_.empty() || dt <= 0.0f) return;

    // ── Emergency Signal Preemption Priority Override ────────────────────────
    if (emergency_preemption_road_ != kInvalidRoadId) {
        size_t target_phase_idx = phases_.size();
        for (size_t i = 0; i < phases_.size(); ++i) {
            if (phases_[i].contains_road(emergency_preemption_road_)) {
                target_phase_idx = i;
                break;
            }
        }

        if (target_phase_idx < phases_.size()) {
            const auto& phase = phases_[current_phase_idx_];
            if (current_phase_idx_ == target_phase_idx) {
                if (current_stage_ == PhaseStage::Green) {
                    // Hold green indefinitely while preemption request is active
                    stage_elapsed_s_ += dt;
                    last_decision_ = AdaptiveDecision::Hold;
                    last_decision_reason_ = "Holding green for emergency preemption";
                    update_lights();
                    return;
                } else if (current_stage_ == PhaseStage::Yellow) {
                    stage_elapsed_s_ += dt;
                    if (stage_elapsed_s_ >= phase.yellow_duration_s) {
                        stage_elapsed_s_ = 0.0f;
                        current_stage_ = (phase.all_red_duration_s > 0.0f)
                                       ? PhaseStage::AllRed : PhaseStage::Green;
                    }
                    update_lights();
                    return;
                } else if (current_stage_ == PhaseStage::AllRed) {
                    stage_elapsed_s_ += dt;
                    if (stage_elapsed_s_ >= phase.all_red_duration_s) {
                        stage_elapsed_s_ = 0.0f;
                        current_stage_ = PhaseStage::Green;
                    }
                    update_lights();
                    return;
                }
            } else {
                // Conflicting phase: clear traffic safely and switch right-of-way
                if (current_stage_ == PhaseStage::Green) {
                    stage_elapsed_s_ += dt;
                    constexpr float kMinSafetyGreen = 2.0f;
                    if (stage_elapsed_s_ >= kMinSafetyGreen) {
                        transition_to_yellow();
                        last_decision_ = AdaptiveDecision::SwitchPhase;
                        last_decision_reason_ = "Preempting signal phase for approaching emergency vehicle";
                    }
                    update_lights();
                    return;
                } else if (current_stage_ == PhaseStage::Yellow) {
                    stage_elapsed_s_ += dt;
                    if (stage_elapsed_s_ >= phase.yellow_duration_s) {
                        stage_elapsed_s_ = 0.0f;
                        current_stage_ = (phase.all_red_duration_s > 0.0f)
                                       ? PhaseStage::AllRed : PhaseStage::Green;
                        if (current_stage_ == PhaseStage::Green) {
                            current_phase_idx_ = target_phase_idx;
                        }
                    }
                    update_lights();
                    return;
                } else if (current_stage_ == PhaseStage::AllRed) {
                    stage_elapsed_s_ += dt;
                    if (stage_elapsed_s_ >= phase.all_red_duration_s) {
                        stage_elapsed_s_ = 0.0f;
                        current_phase_idx_ = target_phase_idx;
                        current_stage_ = PhaseStage::Green;
                    }
                    update_lights();
                    return;
                }
            }
        }
    }

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
    current_phase_idx_         = 0;
    current_stage_             = PhaseStage::Green;
    stage_elapsed_s_           = 0.0f;
    last_decision_             = AdaptiveDecision::Hold;
    last_decision_reason_      = "Controller reset";
    extensions_count_          = 0;
    shortenings_count_         = 0;
    switches_count_            = 0;
    emergency_preemption_road_ = kInvalidRoadId;
    local_queues_.assign(all_incoming_roads_.size(), 0u);
    local_wait_times_.assign(all_incoming_roads_.size(), 0.0f);
    update_lights();
}

void AdaptiveController::request_emergency_preemption(RoadId road_id) {
    emergency_preemption_road_ = road_id;
}

void AdaptiveController::clear_emergency_preemption() {
    emergency_preemption_road_ = kInvalidRoadId;
    stage_elapsed_s_ = 0.0f;
}

// ── Decision Logic ───────────────────────────────────────────────────────────

void AdaptiveController::evaluate_green_decision() {
    if (phases_.empty() || current_phase_idx_ >= phases_.size()) return;
    const auto& phase = phases_[current_phase_idx_];

    // Single pass over local incoming roads (2-4 entries, zero hash lookups)
    uint32_t current_queue     = 0;
    uint32_t opposing_queue    = 0;
    float    max_opposing_wait = 0.0f;
    const size_t n = std::min(all_incoming_roads_.size(), local_queues_.size());
    for (size_t i = 0; i < n; ++i) {
        const RoadId r = all_incoming_roads_[i];
        if (phase.contains_road(r)) {
            current_queue += local_queues_[i];
        } else {
            opposing_queue += local_queues_[i];
            if (local_wait_times_[i] > max_opposing_wait) {
                max_opposing_wait = local_wait_times_[i];
            }
        }
    }

    char reason_buf[160];

    // ── Rule 1: Mandatory Minimum Green Constraint ───────────────────────────
    if (stage_elapsed_s_ < config_.min_green_s) {
        last_decision_ = AdaptiveDecision::Hold;
        const int len = std::snprintf(reason_buf, sizeof(reason_buf),
            "Serving minimum green (%.1fs / %.1fs)",
            static_cast<double>(stage_elapsed_s_), static_cast<double>(config_.min_green_s));
        if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
        return;
    }

    // ── Rule 2: Shorten Phase (Early Gap-Out) ────────────────────────────────
    if (stage_elapsed_s_ < config_.nominal_green_s) {
        if (current_queue == 0 && opposing_queue > 0) {
            last_decision_ = AdaptiveDecision::ShortenPhase;
            shortenings_count_++;
            const int len = std::snprintf(reason_buf, sizeof(reason_buf),
                "Shortening green early: queue cleared (0 veh) while opposing queue (%u veh) waiting at %.1fs",
                opposing_queue, static_cast<double>(stage_elapsed_s_));
            if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
            transition_to_yellow();
            return;
        }

        last_decision_ = AdaptiveDecision::Hold;
        const int len = std::snprintf(reason_buf, sizeof(reason_buf),
            "Serving nominal green (%.1fs / %.1fs, queue: %u)",
            static_cast<double>(stage_elapsed_s_), static_cast<double>(config_.nominal_green_s), current_queue);
        if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
        return;
    }

    // ── Rule 3: Max Green Cap Reached ────────────────────────────────────────
    if (stage_elapsed_s_ >= config_.max_green_s) {
        last_decision_ = AdaptiveDecision::SwitchPhase;
        switches_count_++;
        const int len = std::snprintf(reason_buf, sizeof(reason_buf),
            "Switching phase: max green limit reached (%.1fs)",
            static_cast<double>(config_.max_green_s));
        if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
        transition_to_yellow();
        return;
    }

    // ── Rule 4: Starvation Prevention (Opposing Wait Exceeds Threshold) ───────
    if (max_opposing_wait >= config_.max_wait_threshold_s && opposing_queue > 0) {
        last_decision_ = AdaptiveDecision::SwitchPhase;
        switches_count_++;
        const int len = std::snprintf(reason_buf, sizeof(reason_buf),
            "Switching phase: opposing wait (%.1fs) reached starvation threshold (%.1fs)",
            static_cast<double>(max_opposing_wait), static_cast<double>(config_.max_wait_threshold_s));
        if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
        transition_to_yellow();
        return;
    }

    // ── Rule 5: Demand-Responsive Green Extension ─────────────────────────────
    const bool current_has_demand = (current_queue > 0);
    const bool higher_or_equal_demand = (current_queue * config_.queue_priority_ratio >= opposing_queue);

    if (current_has_demand && (higher_or_equal_demand || opposing_queue == 0)) {
        last_decision_ = AdaptiveDecision::ExtendGreen;
        extensions_count_++;
        const int len = std::snprintf(reason_buf, sizeof(reason_buf),
            "Extending green: active current queue (%u veh) >= opposing (%u veh, max wait: %.1fs)",
            current_queue, opposing_queue, static_cast<double>(max_opposing_wait));
        if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
        return;
    }

    // ── Rule 6: Normal Expiration / Switch to Higher Opposing Demand ──────────
    last_decision_ = AdaptiveDecision::SwitchPhase;
    switches_count_++;
    const int len = std::snprintf(reason_buf, sizeof(reason_buf),
        "Switching phase: opposing demand (%u veh) exceeds current (%u veh)",
        opposing_queue, current_queue);
    if (len > 0) last_decision_reason_.assign(reason_buf, static_cast<size_t>(std::min<int>(len, static_cast<int>(sizeof(reason_buf) - 1))));
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
