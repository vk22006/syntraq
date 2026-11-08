#pragma once

//
// syntraq/routing/route_cost.h
//
// Cost function abstractions for edge traversal weighting.
// Supports:
//   - Distance-only cost (metres)
//   - Free-flow travel time cost (seconds)
//   - Traffic-aware cost: distance + congestion penalty + predicted delay
//
// Decoupled from any specific AI model through pluggable provider callbacks.
//

#include "syntraq/world/road.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>

namespace syntraq {

/// Callable type for calculating the traversal cost of a road edge.
using RoadCostFunction = std::function<float(const Road& road)>;

/// Distance-based cost (metres). Default edge weighting strategy.
struct DistanceCost {
    [[nodiscard]] float operator()(const Road& road) const noexcept {
        return road.length_m;
    }
};

/// Free-flow travel time cost (seconds).
/// Uses length / speed limit, falling back to length if speed limit is invalid.
struct FreeFlowTravelTimeCost {
    [[nodiscard]] float operator()(const Road& road) const noexcept {
        if (road.speed_limit_mps <= 0.0f) {
            return road.length_m;
        }
        return road.length_m / road.speed_limit_mps;
    }
};

// ── Traffic-Aware Cost Components ────────────────────────────────────────────

/// Snapshot of real-time or estimated traffic conditions on a specific road.
struct RoadTrafficCondition {
    uint32_t queued_vehicles  { 0 };     ///< Number of stopped / queued vehicles (speed < 0.5 m/s)
    uint32_t vehicle_count    { 0 };     ///< Total active vehicles occupying this road
    float    avg_speed_mps    { 0.0f };  ///< Average speed across vehicles on this road (m/s)
    float    congestion_ratio { 0.0f };  ///< Congestion factor in [0.0, 1.0] (e.g. queued / active)
    float    predicted_delay_s{ 0.0f };  ///< Predicted queue dissipation or wait delay in seconds
};

/// Generic callback for predicting traversal or signal delay on a road.
/// Pluggable and decoupled from any specific machine learning or heuristic engine.
using DelayProvider = std::function<float(const Road& road, const RoadTrafficCondition& condition)>;

/// Generic callback for querying the traffic condition of a road.
using TrafficConditionProvider = std::function<RoadTrafficCondition(RoadId id)>;

/// Configuration weights for traffic-aware cost computation.
struct TrafficAwareCostConfig {
    float distance_weight       { 1.0f };  ///< Weight applied to raw distance (m)
    float congestion_weight     { 1.0f };  ///< Weight applied to congestion penalty
    float delay_weight          { 1.0f };  ///< Weight applied to predicted delay cost
    float queue_penalty_per_veh { 25.0f }; ///< Penalty per queued vehicle in equivalent metres (m/veh)
    float delay_to_cost_factor  { 13.89f };///< Scaling factor converting delay (s) to equivalent distance cost (m/s)
    float jam_density_penalty   { 100.0f };///< Additional scaling penalty as congestion approaches 1.0
};

/// Traffic-aware road cost calculator.
/// Evaluates:
///   road cost = (w_distance * distance)
///             + (w_congestion * congestion_penalty)
///             + (w_delay * predicted_delay)
///
/// Under zero congestion, smoothly reduces to distance-only cost.
class TrafficAwareRoadCost {
public:
    explicit TrafficAwareRoadCost(
        TrafficAwareCostConfig config = {},
        DelayProvider delay_provider = nullptr)
        : config_(config)
        , delay_provider_(std::move(delay_provider))
    {
    }

    /// Analytical static cost evaluation:
    ///   road_cost = (w_dist * distance) + (w_cong * congestion_penalty) + (w_delay * predicted_delay)
    [[nodiscard]] static constexpr float compute_cost(
        float distance,
        float congestion_penalty,
        float predicted_delay,
        float distance_weight = 1.0f,
        float congestion_weight = 1.0f,
        float delay_weight = 1.0f) noexcept {
        const float c = (distance_weight * distance) +
                        (congestion_weight * congestion_penalty) +
                        (delay_weight * predicted_delay);
        return c >= 0.0f ? c : 0.0f;
    }

    /// Computes congestion penalty for a road from its condition.
    [[nodiscard]] float compute_congestion_penalty(
        const Road& road,
        const RoadTrafficCondition& condition) const noexcept {
        (void)road;
        const float queue_pen = static_cast<float>(condition.queued_vehicles) * config_.queue_penalty_per_veh;
        const float jam_pen   = std::clamp(condition.congestion_ratio, 0.0f, 1.0f) * config_.jam_density_penalty;
        return queue_pen + jam_pen;
    }

    /// Computes predicted delay cost for a road from its condition.
    [[nodiscard]] float compute_predicted_delay(
        const Road& road,
        const RoadTrafficCondition& condition) const noexcept {
        // 1. Pluggable custom delay provider (external model or heuristic)
        if (delay_provider_) {
            const float delay_s = delay_provider_(road, condition);
            return std::max(0.0f, delay_s) * config_.delay_to_cost_factor;
        }

        // 2. Explicit predicted delay passed in condition
        if (condition.predicted_delay_s > 0.0f) {
            return condition.predicted_delay_s * config_.delay_to_cost_factor;
        }

        // 3. Deterministic queuing theory fallback (service headway ~ 2.0s per queued vehicle)
        if (condition.queued_vehicles > 0) {
            const float lanes = std::max(1.0f, static_cast<float>(road.lane_count()));
            constexpr float kServiceHeadwayS = 2.0f;
            const float queue_delay_s = (static_cast<float>(condition.queued_vehicles) * kServiceHeadwayS) / lanes;
            return queue_delay_s * config_.delay_to_cost_factor;
        }

        // 4. Speed deficit delay (slow moving traffic below free-flow)
        if (condition.vehicle_count > 0 && condition.avg_speed_mps > 0.1f && road.speed_limit_mps > 0.1f) {
            if (condition.avg_speed_mps < road.speed_limit_mps) {
                const float free_flow_time = road.length_m / road.speed_limit_mps;
                const float actual_time    = road.length_m / condition.avg_speed_mps;
                const float delay_s        = std::max(0.0f, actual_time - free_flow_time);
                return delay_s * config_.delay_to_cost_factor;
            }
        }

        return 0.0f;
    }

    /// Compute full cost for a road given its explicit condition.
    [[nodiscard]] float operator()(
        const Road& road,
        const RoadTrafficCondition& condition) const noexcept {
        const float dist        = road.length_m;
        const float cong_pen    = compute_congestion_penalty(road, condition);
        const float pred_delay  = compute_predicted_delay(road, condition);

        return compute_cost(dist,
                            cong_pen,
                            pred_delay,
                            config_.distance_weight,
                            config_.congestion_weight,
                            config_.delay_weight);
    }

    /// Compute cost for a road querying the condition provider or internal condition cache.
    [[nodiscard]] float operator()(const Road& road) const noexcept {
        if (condition_provider_) {
            return (*this)(road, condition_provider_(road.id));
        }
        auto it = conditions_.find(road.id);
        if (it != conditions_.end()) {
            return (*this)(road, it->second);
        }
        // Zero traffic fallback: raw distance
        return road.length_m * config_.distance_weight;
    }

    void set_condition_provider(TrafficConditionProvider provider) {
        condition_provider_ = std::move(provider);
    }

    void set_delay_provider(DelayProvider provider) {
        delay_provider_ = std::move(provider);
    }

    void set_conditions(std::unordered_map<RoadId, RoadTrafficCondition> conditions) {
        conditions_ = std::move(conditions);
    }

    void set_condition(RoadId id, const RoadTrafficCondition& condition) {
        conditions_[id] = condition;
    }

    void clear_conditions() noexcept {
        conditions_.clear();
    }

    [[nodiscard]] const TrafficAwareCostConfig& config() const noexcept {
        return config_;
    }

    void set_config(const TrafficAwareCostConfig& config) noexcept {
        config_ = config;
    }

private:
    TrafficAwareCostConfig                           config_{};
    DelayProvider                                    delay_provider_{ nullptr };
    TrafficConditionProvider                         condition_provider_{ nullptr };
    std::unordered_map<RoadId, RoadTrafficCondition> conditions_{};
};

} // namespace syntraq
