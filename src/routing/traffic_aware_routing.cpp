//
// src/routing/traffic_aware_routing.cpp
//

#include "syntraq/routing/traffic_aware_routing.hpp"

#include <algorithm>

namespace syntraq {

TrafficAwareRouting::TrafficAwareRouting(
    TrafficAwareCostConfig config,
    DelayProvider delay_provider)
    : cost_model_(config, std::move(delay_provider))
{
    setup_router();
}

void TrafficAwareRouting::setup_router() {
    router_ = std::make_unique<AStarRouter>(
        [this](const Road& road) {
            return compute_road_cost(road);
        }
    );
}

void TrafficAwareRouting::set_road_condition(RoadId road_id, const RoadTrafficCondition& condition) {
    conditions_[road_id] = condition;
    cost_model_.set_condition(road_id, condition);
}

void TrafficAwareRouting::set_road_conditions(std::unordered_map<RoadId, RoadTrafficCondition> conditions) {
    conditions_ = std::move(conditions);
    cost_model_.set_conditions(conditions_);
}

void TrafficAwareRouting::clear_road_conditions() noexcept {
    conditions_.clear();
    cost_model_.clear_conditions();
}

void TrafficAwareRouting::update_from_simulation(
    const RoadNetwork& network,
    const std::vector<Vehicle>& vehicles)
{
    (void)network;
    std::unordered_map<RoadId, uint32_t> counts;
    std::unordered_map<RoadId, uint32_t> queues;
    std::unordered_map<RoadId, float>    speed_sums;

    for (const auto& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;
        if (v.current_road == kInvalidRoadId) continue;

        counts[v.current_road]++;
        speed_sums[v.current_road] += v.speed_mps;

        if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
            queues[v.current_road]++;
        }
    }

    conditions_.clear();
    for (const auto& [rid, total] : counts) {
        RoadTrafficCondition cond{};
        cond.vehicle_count    = total;
        cond.queued_vehicles  = queues[rid];
        cond.avg_speed_mps    = total > 0 ? (speed_sums[rid] / static_cast<float>(total)) : 0.0f;
        cond.congestion_ratio = total > 0 ? (static_cast<float>(cond.queued_vehicles) / static_cast<float>(total)) : 0.0f;

        conditions_[rid] = cond;
    }

    cost_model_.set_conditions(conditions_);
}

void TrafficAwareRouting::update_from_queues(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>& avg_speeds)
{
    conditions_.clear();
    for (const auto& [rid, q] : queues) {
        RoadTrafficCondition cond{};
        cond.queued_vehicles  = q;
        cond.vehicle_count    = q;
        cond.congestion_ratio = q > 0 ? 1.0f : 0.0f;

        auto s_it = avg_speeds.find(rid);
        if (s_it != avg_speeds.end()) {
            cond.avg_speed_mps = s_it->second;
        } else {
            cond.avg_speed_mps = q > 0 ? 0.0f : 13.89f;
        }

        conditions_[rid] = cond;
    }

    cost_model_.set_conditions(conditions_);
}

void TrafficAwareRouting::set_delay_provider(DelayProvider provider) {
    cost_model_.set_delay_provider(std::move(provider));
}

void TrafficAwareRouting::set_cost_config(const TrafficAwareCostConfig& config) noexcept {
    cost_model_.set_config(config);
}

const TrafficAwareCostConfig& TrafficAwareRouting::cost_config() const noexcept {
    return cost_model_.config();
}

RoadTrafficCondition TrafficAwareRouting::condition_for(RoadId id) const noexcept {
    auto it = conditions_.find(id);
    if (it != conditions_.end()) {
        return it->second;
    }
    return RoadTrafficCondition{};
}

float TrafficAwareRouting::compute_road_cost(const Road& road) const noexcept {
    return cost_model_(road);
}

RouteResult TrafficAwareRouting::find_route_with_info(
    const RoadNetwork& network,
    IntersectionId     source,
    IntersectionId     destination) const
{
    if (!router_) {
        return { {}, 0.0f, 0, false };
    }
    return router_->find_route_with_info(network, source, destination);
}

} // namespace syntraq
