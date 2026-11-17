//
// src/routing/traffic_aware_routing.cpp
//

#include "syntraq/routing/traffic_aware_routing.hpp"

#include <algorithm>

namespace syntraq {

namespace {

constexpr size_t kMaxRouteCacheEntries = 4096;

[[nodiscard]] constexpr uint64_t pack_od_key(IntersectionId src, IntersectionId dst) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(dst));
}

} // namespace

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

void TrafficAwareRouting::ensure_dense_capacity(uint32_t road_idx) {
    if (road_idx >= dense_conditions_.size()) {
        const size_t new_size = std::max<size_t>(static_cast<size_t>(road_idx) + 1,
                                                 dense_conditions_.size() * 2 + 64);
        dense_conditions_.resize(new_size);
        dense_epochs_.resize(new_size, 0u);
        scratch_speed_sums_.resize(new_size, 0.0f);
    }
}

void TrafficAwareRouting::advance_epoch() noexcept {
    ++current_epoch_;
    if (current_epoch_ == 0) {
        std::fill(dense_epochs_.begin(), dense_epochs_.end(), 0u);
        current_epoch_ = 1;
    }
    active_road_ids_.clear();
}

void TrafficAwareRouting::set_road_condition(RoadId road_id, const RoadTrafficCondition& condition) {
    if (road_id == kInvalidRoadId) return;
    const auto idx = static_cast<uint32_t>(road_id);
    ensure_dense_capacity(idx);
    if (dense_epochs_[idx] != current_epoch_) {
        dense_epochs_[idx] = current_epoch_;
        active_road_ids_.push_back(idx);
    }
    dense_conditions_[idx] = condition;
    conditions_view_dirty_ = true;
    ++state_version_;
}

void TrafficAwareRouting::set_road_conditions(std::unordered_map<RoadId, RoadTrafficCondition> conditions) {
    advance_epoch();
    for (const auto& [rid, cond] : conditions) {
        if (rid == kInvalidRoadId) continue;
        const auto idx = static_cast<uint32_t>(rid);
        ensure_dense_capacity(idx);
        dense_epochs_[idx]     = current_epoch_;
        dense_conditions_[idx] = cond;
        active_road_ids_.push_back(idx);
    }
    conditions_view_dirty_ = true;
    ++state_version_;
}

void TrafficAwareRouting::clear_road_conditions() noexcept {
    if (!active_road_ids_.empty()) {
        advance_epoch();
        conditions_view_dirty_ = true;
        ++state_version_;
    }
}

void TrafficAwareRouting::update_from_simulation(
    const RoadNetwork& network,
    const std::vector<Vehicle>& vehicles)
{
    const uint32_t max_rid = network.max_road_id();
    if (max_rid > 0) {
        ensure_dense_capacity(max_rid - 1);
    }

    advance_epoch();

    for (const auto& v : vehicles) {
        if (v.state == VehicleState::Arrived) continue;
        if (v.current_road == kInvalidRoadId) continue;

        const auto idx = static_cast<uint32_t>(v.current_road);
        if (idx >= dense_conditions_.size()) {
            ensure_dense_capacity(idx);
        }

        if (dense_epochs_[idx] != current_epoch_) {
            dense_epochs_[idx]       = current_epoch_;
            dense_conditions_[idx]   = RoadTrafficCondition{};
            scratch_speed_sums_[idx] = 0.0f;
            active_road_ids_.push_back(idx);
        }

        auto& cond = dense_conditions_[idx];
        cond.vehicle_count++;
        scratch_speed_sums_[idx] += v.speed_mps;

        if (v.state == VehicleState::Stopped || v.speed_mps < 0.5f) {
            cond.queued_vehicles++;
        }
    }

    for (uint32_t idx : active_road_ids_) {
        auto& cond = dense_conditions_[idx];
        if (cond.vehicle_count > 0) {
            const float inv = 1.0f / static_cast<float>(cond.vehicle_count);
            cond.avg_speed_mps    = scratch_speed_sums_[idx] * inv;
            cond.congestion_ratio = static_cast<float>(cond.queued_vehicles) * inv;
        }
    }

    conditions_view_dirty_ = true;
    ++state_version_;
}

void TrafficAwareRouting::update_from_queues(
    const std::unordered_map<RoadId, uint32_t>& queues,
    const std::unordered_map<RoadId, float>& avg_speeds)
{
    advance_epoch();
    for (const auto& [rid, q] : queues) {
        if (rid == kInvalidRoadId) continue;
        const auto idx = static_cast<uint32_t>(rid);
        ensure_dense_capacity(idx);

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

        dense_epochs_[idx]     = current_epoch_;
        dense_conditions_[idx] = cond;
        active_road_ids_.push_back(idx);
    }

    conditions_view_dirty_ = true;
    ++state_version_;
}

void TrafficAwareRouting::set_delay_provider(DelayProvider provider) {
    cost_model_.set_delay_provider(std::move(provider));
    ++state_version_;
}

void TrafficAwareRouting::set_cost_config(const TrafficAwareCostConfig& config) noexcept {
    cost_model_.set_config(config);
    ++state_version_;
}

const TrafficAwareCostConfig& TrafficAwareRouting::cost_config() const noexcept {
    return cost_model_.config();
}

const std::unordered_map<RoadId, RoadTrafficCondition>&
TrafficAwareRouting::road_conditions() const noexcept {
    if (conditions_view_dirty_) {
        conditions_view_.clear();
        for (uint32_t idx : active_road_ids_) {
            if (idx < dense_epochs_.size() && dense_epochs_[idx] == current_epoch_) {
                conditions_view_[static_cast<RoadId>(idx)] = dense_conditions_[idx];
            }
        }
        conditions_view_dirty_ = false;
    }
    return conditions_view_;
}

RoadTrafficCondition TrafficAwareRouting::condition_for(RoadId id) const noexcept {
    const auto idx = static_cast<uint32_t>(id);
    if (idx < dense_epochs_.size() && dense_epochs_[idx] == current_epoch_) {
        return dense_conditions_[idx];
    }
    return RoadTrafficCondition{};
}

void TrafficAwareRouting::set_emergency_corridor_roads(std::unordered_set<RoadId> roads) {
    if (roads == emergency_corridor_roads_) return;
    emergency_corridor_roads_ = std::move(roads);
    std::fill(corridor_mask_.begin(), corridor_mask_.end(), static_cast<uint8_t>(0));
    has_emergency_corridors_ = !emergency_corridor_roads_.empty();
    for (RoadId rid : emergency_corridor_roads_) {
        if (rid == kInvalidRoadId) continue;
        const auto idx = static_cast<uint32_t>(rid);
        if (idx >= corridor_mask_.size()) {
            corridor_mask_.resize(static_cast<size_t>(idx) + 1, static_cast<uint8_t>(0));
        }
        corridor_mask_[idx] = 1u;
    }
    ++state_version_;
}

void TrafficAwareRouting::clear_emergency_corridor_roads() noexcept {
    if (!has_emergency_corridors_ && emergency_corridor_roads_.empty()) return;
    emergency_corridor_roads_.clear();
    std::fill(corridor_mask_.begin(), corridor_mask_.end(), static_cast<uint8_t>(0));
    has_emergency_corridors_ = false;
    ++state_version_;
}

float TrafficAwareRouting::compute_road_cost(const Road& road) const noexcept {
    const auto idx = static_cast<uint32_t>(road.id);
    float cost = 0.0f;
    if (idx < dense_epochs_.size() && dense_epochs_[idx] == current_epoch_) {
        cost = cost_model_(road, dense_conditions_[idx]);
    } else {
        // Zero-traffic fallback (still honors custom delay_provider_ if installed)
        static const RoadTrafficCondition kEmptyCond{};
        cost = cost_model_(road, kEmptyCond);
    }

    if (has_emergency_corridors_ && idx < corridor_mask_.size() && corridor_mask_[idx] != 0u) {
        cost += cost_model_.config().emergency_corridor_penalty;
    }
    return cost;
}

void TrafficAwareRouting::set_cache_enabled(bool enabled) noexcept {
    cache_enabled_ = enabled;
    if (!enabled) {
        clear_cache();
    }
}

void TrafficAwareRouting::clear_cache() const noexcept {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    route_cache_.clear();
}

RouteResult TrafficAwareRouting::find_route_with_info(
    const RoadNetwork& network,
    IntersectionId     source,
    IntersectionId     destination) const
{
    if (!router_) {
        return { {}, 0.0f, 0, false };
    }

    if (!cache_enabled_) {
        return router_->find_route_with_info(network, source, destination);
    }

    const uint64_t key = pack_od_key(source, destination);
    const uint64_t topo_ver = network.topology_version();

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cached_state_version_ != state_version_ ||
            cached_network_ != &network ||
            cached_topology_version_ != topo_ver) {
            route_cache_.clear();
            cached_state_version_    = state_version_;
            cached_network_          = &network;
            cached_topology_version_ = topo_ver;
        } else {
            auto it = route_cache_.find(key);
            if (it != route_cache_.end()) {
                ++cache_hits_;
                return it->second;
            }
        }
        ++cache_misses_;
    }

    RouteResult result = router_->find_route_with_info(network, source, destination);

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cached_state_version_ == state_version_ &&
            cached_network_ == &network &&
            cached_topology_version_ == topo_ver) {
            if (route_cache_.size() >= kMaxRouteCacheEntries) {
                route_cache_.clear();
            }
            route_cache_.emplace(key, result);
        }
    }

    return result;
}

} // namespace syntraq
