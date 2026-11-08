//
// src/signals/coordinated_controller.cpp
//

#include "syntraq/signals/coordinated_controller.hpp"
#include "syntraq/world/road_network.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace syntraq {

CoordinatedController::CoordinatedController(
    IntersectionId              intersection_id,
    std::vector<SignalPhase>    phases,
    std::vector<RoadId>         all_incoming_roads,
    CoordinatedControllerConfig config)
    : intersection_id_       (intersection_id)
    , phases_                (std::move(phases))
    , config_                (config)
    , current_phase_idx_     (0)
    , current_stage_         (PhaseStage::Green)
    , stage_elapsed_s_       (0.0f)
    , current_stage_duration_(config.nominal_green_s)
    , last_decision_         (CoordinatedDecision::Hold)
    , last_reason_           ("Initial green stage")
{
    local_state_.intersection_id    = intersection_id_;
    local_state_.active_phase_index = 0;
    local_state_.stage              = PhaseStage::Green;

    // Initialize traffic lights for all incoming roads
    std::vector<RoadId> roads = std::move(all_incoming_roads);
    if (roads.empty()) {
        for (const auto& p : phases_) {
            for (RoadId r : p.green_roads) {
                if (std::find(roads.begin(), roads.end(), r) == roads.end()) {
                    roads.push_back(r);
                }
            }
        }
    }

    for (RoadId r : roads) {
        TrafficLight light_obj;
        light_obj.road_id = r;
        light_obj.color   = SignalColor::Red;
        lights_.emplace(r, light_obj);
    }

    if (!phases_.empty()) {
        current_stage_duration_ = phases_[0].green_duration_s > 0.0f
                                ? phases_[0].green_duration_s
                                : config_.nominal_green_s;
    }

    apply_stage_lights();
}

void CoordinatedController::reset() {
    current_phase_idx_      = 0;
    current_stage_          = PhaseStage::Green;
    stage_elapsed_s_        = 0.0f;
    current_stage_duration_ = !phases_.empty() ? phases_[0].green_duration_s : config_.nominal_green_s;
    last_decision_          = CoordinatedDecision::Hold;
    last_reason_            = "Controller reset to Phase 0 Green";
    neighbor_states_.clear();

    local_state_.active_phase_index = 0;
    local_state_.stage              = PhaseStage::Green;
    local_state_.stage_elapsed_s    = 0.0f;
    local_state_.stage_remaining_s  = current_stage_duration_;
    local_state_.active_phase_queue = 0;
    local_state_.cross_phase_queue  = 0;
    local_state_.cross_phase_max_wait_s = 0.0f;

    apply_stage_lights();
}

void CoordinatedController::observe_traffic(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>&    wait_times)
{
    refresh_local_state(queues, wait_times);
}

void CoordinatedController::receive_neighbor_state(const NeighborIntersectionState& state) {
    auto it = std::find_if(neighbor_states_.begin(), neighbor_states_.end(),
        [&](const NeighborIntersectionState& n) {
            return n.neighbor_id == state.neighbor_id && n.connecting_road == state.connecting_road;
        });
    if (it != neighbor_states_.end()) {
        *it = state;
    } else {
        neighbor_states_.push_back(state);
    }
}

void CoordinatedController::receive_neighbor_states(const std::vector<NeighborIntersectionState>& states) {
    for (const auto& s : states) {
        receive_neighbor_state(s);
    }
}

void CoordinatedController::clear_neighbor_states() noexcept {
    neighbor_states_.clear();
}

void CoordinatedController::refresh_local_state(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>&    wait_times)
{
    local_state_.intersection_id    = intersection_id_;
    local_state_.active_phase_index = current_phase_idx_;
    local_state_.stage              = current_stage_;
    local_state_.stage_elapsed_s    = stage_elapsed_s_;
    local_state_.stage_remaining_s  = stage_remaining_s();
    local_state_.active_phase_queue = 0;
    local_state_.cross_phase_queue  = 0;
    local_state_.cross_phase_max_wait_s = 0.0f;

    const SignalPhase* active_p = current_phase();

    for (const auto& [r_id, _] : lights_) {
        ApproachLocalState approach;
        approach.road_id = r_id;

        auto q_it = queues.find(r_id);
        if (q_it != queues.end()) approach.queue_length = q_it->second;

        auto w_it = wait_times.find(r_id);
        if (w_it != wait_times.end()) approach.max_wait_time_s = w_it->second;

        local_state_.approaches[r_id] = approach;

        const bool is_active_green = active_p &&
            (std::find(active_p->green_roads.begin(), active_p->green_roads.end(), r_id) != active_p->green_roads.end());

        if (is_active_green) {
            local_state_.active_phase_queue += approach.queue_length;
        } else {
            local_state_.cross_phase_queue += approach.queue_length;
            if (approach.max_wait_time_s > local_state_.cross_phase_max_wait_s) {
                local_state_.cross_phase_max_wait_s = approach.max_wait_time_s;
            }
        }
    }
}

void CoordinatedController::tick(float dt) {
    if (phases_.empty()) return;

    stage_elapsed_s_ += dt;
    local_state_.stage_elapsed_s = stage_elapsed_s_;
    local_state_.stage_remaining_s = stage_remaining_s();

    if (current_stage_ == PhaseStage::Green) {
        const auto decision = evaluate_coordination_decision();
        if (decision == CoordinatedDecision::MitigateSpillback ||
            decision == CoordinatedDecision::SwitchPhase ||
            decision == CoordinatedDecision::ShortenPhase) {
            transition_to_stage(PhaseStage::Yellow);
            return;
        }

        if (stage_elapsed_s_ >= current_stage_duration_) {
            last_decision_ = CoordinatedDecision::SwitchPhase;
            last_reason_   = "Nominal green stage expired";
            transition_to_stage(PhaseStage::Yellow);
            return;
        }
    } else if (current_stage_ == PhaseStage::Yellow) {
        last_decision_ = CoordinatedDecision::Hold;
        last_reason_   = "Serving yellow clearance";
        if (stage_elapsed_s_ >= current_stage_duration_) {
            transition_to_stage(PhaseStage::AllRed);
            return;
        }
    } else if (current_stage_ == PhaseStage::AllRed) {
        last_decision_ = CoordinatedDecision::Hold;
        last_reason_   = "Serving all-red clearance";
        if (stage_elapsed_s_ >= current_stage_duration_) {
            advance_to_next_phase();
            transition_to_stage(PhaseStage::Green);
            return;
        }
    }
}

CoordinatedDecision CoordinatedController::evaluate_coordination_decision() {
    // 1. Mandatory Minimum Green Safety
    if (stage_elapsed_s_ < config_.min_green_s) {
        last_decision_ = CoordinatedDecision::Hold;
        last_reason_   = "Serving mandatory minimum green time (safety buffer)";
        return CoordinatedDecision::Hold;
    }

    // 2. Downstream Spillback Gating (Network Queue Spillback Prevention)
    if (config_.enable_spillback_prevention) {
        for (const auto& nb : neighbor_states_) {
            if (!nb.is_upstream) { // Downstream neighbor receiving flow from us
                const bool queue_exceeded = nb.downstream_queue >= config_.spillback_queue_threshold;
                const bool ratio_exceeded = nb.downstream_occupancy_ratio >= config_.spillback_occupancy_ratio;

                if (nb.spillback_imminent || queue_exceeded || ratio_exceeded) {
                    spillback_mitigations_count_++;
                    last_decision_ = CoordinatedDecision::MitigateSpillback;
                    std::ostringstream ss;
                    ss << "Downstream link (to Intersection " << static_cast<uint32_t>(nb.neighbor_id)
                       << ") congested [queue=" << nb.downstream_queue
                       << "] -> truncating green to avoid queue spillback";
                    last_reason_ = ss.str();
                    return CoordinatedDecision::MitigateSpillback;
                }
            }
        }
    }

    // 3. Upstream Green Wave Platoon Progression (Progression Synchronization)
    if (config_.enable_green_wave) {
        const SignalPhase* active_p = current_phase();
        for (const auto& nb : neighbor_states_) {
            if (nb.is_upstream && nb.platoon_dispatched) {
                // Check if this upstream neighbor connects to our currently green approach
                const bool feeds_active_green = active_p &&
                    (std::find(active_p->green_roads.begin(), active_p->green_roads.end(), nb.connecting_road) != active_p->green_roads.end());

                if (feeds_active_green) {
                    if (nb.estimated_platoon_eta_s >= 0.0f &&
                        nb.estimated_platoon_eta_s <= config_.progression_window_s) {
                        if (stage_elapsed_s_ < config_.max_green_s) {
                            green_wave_extensions_count_++;
                            last_decision_ = CoordinatedDecision::ExtendGreenWave;
                            const float target_ext = std::min(config_.max_green_s,
                                                              stage_elapsed_s_ + config_.progression_window_s);
                            current_stage_duration_ = std::max(current_stage_duration_, target_ext);
                            std::ostringstream ss;
                            ss << "Holding green for arriving upstream platoon from Intersection "
                               << static_cast<uint32_t>(nb.neighbor_id)
                               << " (ETA=" << nb.estimated_platoon_eta_s << "s, size="
                               << nb.estimated_platoon_size << " veh)";
                            last_reason_ = ss.str();
                            return CoordinatedDecision::ExtendGreenWave;
                        }
                    }
                }
            }
        }
    }

    // 4. Maximum Green Failsafe
    if (stage_elapsed_s_ >= config_.max_green_s) {
        last_decision_ = CoordinatedDecision::SwitchPhase;
        last_reason_   = "Maximum green cap reached";
        return CoordinatedDecision::SwitchPhase;
    }

    // 5. Cross-Traffic Starvation Prevention
    if (local_state_.cross_phase_max_wait_s >= config_.max_wait_threshold_s &&
        local_state_.cross_phase_queue > 0) {
        last_decision_ = CoordinatedDecision::SwitchPhase;
        last_reason_   = "Cross traffic exceeded starvation waiting threshold";
        return CoordinatedDecision::SwitchPhase;
    }

    // 6. Local Queue Demand Extension
    if (local_state_.active_phase_queue > 0) {
        last_decision_ = CoordinatedDecision::ExtendGreenLocal;
        last_reason_   = "Extending green for active phase queue demand";
        current_stage_duration_ = std::min(config_.max_green_s, current_stage_duration_ + 1.0f);
        return CoordinatedDecision::ExtendGreenLocal;
    }

    // 7. Early Gap-Out when active queue clears and cross traffic is waiting
    if (local_state_.active_phase_queue == 0 && local_state_.cross_phase_queue > 0) {
        last_decision_ = CoordinatedDecision::ShortenPhase;
        last_reason_   = "Active queue dissipated while cross traffic is waiting (gap-out)";
        return CoordinatedDecision::ShortenPhase;
    }

    // 8. Default Hold
    last_decision_ = CoordinatedDecision::Hold;
    last_reason_   = "Maintaining active green within nominal duration";
    return CoordinatedDecision::Hold;
}

void CoordinatedController::transition_to_stage(PhaseStage next_stage) {
    current_stage_   = next_stage;
    stage_elapsed_s_ = 0.0f;

    const SignalPhase* p = current_phase();
    if (p) {
        switch (next_stage) {
            case PhaseStage::Green:
                current_stage_duration_ = p->green_duration_s > 0.0f ? p->green_duration_s : config_.nominal_green_s;
                break;
            case PhaseStage::Yellow:
                current_stage_duration_ = p->yellow_duration_s;
                break;
            case PhaseStage::AllRed:
                current_stage_duration_ = p->all_red_duration_s;
                break;
        }
    } else {
        current_stage_duration_ = 0.0f;
    }

    apply_stage_lights();
}

void CoordinatedController::advance_to_next_phase() {
    if (phases_.empty()) return;
    current_phase_idx_ = (current_phase_idx_ + 1) % phases_.size();
}

void CoordinatedController::apply_stage_lights() {
    const SignalPhase* active_p = current_phase();

    for (auto& [r_id, light_obj] : lights_) {
        const bool is_in_active_phase = active_p &&
            (std::find(active_p->green_roads.begin(), active_p->green_roads.end(), r_id) != active_p->green_roads.end());

        if (is_in_active_phase) {
            if (current_stage_ == PhaseStage::Green) {
                light_obj.color = SignalColor::Green;
            } else if (current_stage_ == PhaseStage::Yellow) {
                light_obj.color = SignalColor::Yellow;
            } else {
                light_obj.color = SignalColor::Red;
            }
        } else {
            light_obj.color = SignalColor::Red;
        }
    }
}

SignalColor CoordinatedController::signal_for_road(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return it->second.color;
    }
    return SignalColor::Red;
}

bool CoordinatedController::can_proceed(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return it->second.can_proceed();
    }
    return false;
}

const TrafficLight* CoordinatedController::light(RoadId road_id) const noexcept {
    const auto it = lights_.find(road_id);
    if (it != lights_.end()) {
        return &it->second;
    }
    return nullptr;
}

float CoordinatedController::cycle_duration_s() const noexcept {
    float total = 0.0f;
    for (const auto& p : phases_) {
        total += p.total_duration_s();
    }
    return total;
}

void CoordinatedController::set_phases(std::vector<SignalPhase> phases,
                                       std::vector<RoadId>      all_incoming_roads) {
    phases_ = std::move(phases);
    std::vector<RoadId> roads = std::move(all_incoming_roads);
    if (roads.empty()) {
        for (const auto& p : phases_) {
            for (RoadId r : p.green_roads) {
                if (std::find(roads.begin(), roads.end(), r) == roads.end()) {
                    roads.push_back(r);
                }
            }
        }
    }

    lights_.clear();
    for (RoadId r : roads) {
        TrafficLight light_obj;
        light_obj.road_id = r;
        light_obj.color   = SignalColor::Red;
        lights_.emplace(r, light_obj);
    }

    reset();
}

void CoordinatedController::set_phase_timings(size_t phase_idx,
                                              float  green_s,
                                              float  yellow_s,
                                              float  all_red_s) {
    if (phase_idx < phases_.size()) {
        phases_[phase_idx].green_duration_s   = green_s;
        phases_[phase_idx].yellow_duration_s  = yellow_s;
        phases_[phase_idx].all_red_duration_s = all_red_s;
    }
}

bool CoordinatedController::validate(std::string* error_msg) const {
    if (intersection_id_ == kInvalidIntersectionId) {
        if (error_msg) *error_msg = "Invalid intersection ID.";
        return false;
    }
    if (phases_.empty()) {
        if (error_msg) *error_msg = "CoordinatedController has no configured phases";
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
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " yellow duration must be non-negative.";
            return false;
        }
        if (p.all_red_duration_s < 0.0f) {
            if (error_msg) *error_msg = "Phase " + std::to_string(i) + " all-red duration must be non-negative.";
            return false;
        }
    }
    return true;
}

CoordinatedController CoordinatedController::make_default_2phase(
    const RoadNetwork&          network,
    IntersectionId              id,
    CoordinatedControllerConfig config,
    float                       yellow_duration_s,
    float                       all_red_duration_s)
{
    const Intersection* center = network.intersection(id);
    if (!center) return {};

    const std::vector<RoadId> incoming_roads = network.incoming_roads(id);
    if (incoming_roads.empty()) return {};

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
        phases.push_back(SignalPhase{ "North-South", ns_roads, config.nominal_green_s, yellow_duration_s, all_red_duration_s });
        phases.push_back(SignalPhase{ "East-West",   ew_roads, config.nominal_green_s, yellow_duration_s, all_red_duration_s });
    } else if (!ns_roads.empty()) {
        phases.push_back(SignalPhase{ "North-South", ns_roads, config.nominal_green_s, yellow_duration_s, all_red_duration_s });
    } else if (!ew_roads.empty()) {
        phases.push_back(SignalPhase{ "East-West",   ew_roads, config.nominal_green_s, yellow_duration_s, all_red_duration_s });
    } else {
        phases.push_back(SignalPhase{ "All-Incoming", incoming_roads, config.nominal_green_s, yellow_duration_s, all_red_duration_s });
    }

    return CoordinatedController(id, std::move(phases), incoming_roads, config);
}

} // namespace syntraq
