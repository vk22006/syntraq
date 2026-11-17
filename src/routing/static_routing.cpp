//
// src/routing/static_routing.cpp
//

#include "syntraq/routing/static_routing.hpp"

#include <mutex>
#include <unordered_map>

namespace syntraq {

namespace {

constexpr size_t kMaxStaticCacheEntries = 8192;

[[nodiscard]] constexpr uint64_t pack_od_key(IntersectionId src, IntersectionId dst) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(dst));
}

} // namespace

struct StaticRouting::RouteCache {
    std::mutex                                mutex;
    std::unordered_map<uint64_t, RouteResult> entries;
    const RoadNetwork*                        network_ptr{ nullptr };
    uint64_t                                  topology_version{ 0 };
    uint64_t                                  hits{ 0 };
    uint64_t                                  misses{ 0 };
};

StaticRouting::StaticRouting(Algorithm algo, RoadCostFunction cost_fn)
    : algorithm_(algo)
    , cost_fn_(std::move(cost_fn))
    , cache_(std::make_unique<RouteCache>())
{
    rebuild_backends();
}

StaticRouting::~StaticRouting() = default;
StaticRouting::StaticRouting(StaticRouting&&) noexcept = default;
StaticRouting& StaticRouting::operator=(StaticRouting&&) noexcept = default;

void StaticRouting::set_algorithm(Algorithm algo) {
    algorithm_ = algo;
    rebuild_backends();
    clear_cache();
}

void StaticRouting::set_cost_function(RoadCostFunction cost_fn) {
    cost_fn_ = std::move(cost_fn);
    rebuild_backends();
    clear_cache();
}

void StaticRouting::set_cache_enabled(bool enabled) noexcept {
    cache_enabled_ = enabled;
    if (!enabled) {
        clear_cache();
    }
}

void StaticRouting::clear_cache() const noexcept {
    if (!cache_) return;
    std::lock_guard<std::mutex> lock(cache_->mutex);
    cache_->entries.clear();
}

uint64_t StaticRouting::cache_hits() const noexcept {
    if (!cache_) return 0;
    std::lock_guard<std::mutex> lock(cache_->mutex);
    return cache_->hits;
}

uint64_t StaticRouting::cache_misses() const noexcept {
    if (!cache_) return 0;
    std::lock_guard<std::mutex> lock(cache_->mutex);
    return cache_->misses;
}

void StaticRouting::rebuild_backends() {
    if (algorithm_ == Algorithm::AStar) {
        astar_ = std::make_unique<AStarRouter>(cost_fn_);
        dijkstra_.reset();
    } else {
        dijkstra_ = std::make_unique<DijkstraRouter>(cost_fn_);
        astar_.reset();
    }
}

RouteResult StaticRouting::find_route_with_info(
    const RoadNetwork& network,
    IntersectionId     source,
    IntersectionId     destination) const
{
    auto compute = [&]() -> RouteResult {
        if (algorithm_ == Algorithm::AStar && astar_) {
            return astar_->find_route_with_info(network, source, destination);
        }
        if (dijkstra_) {
            return dijkstra_->find_route_with_info(network, source, destination);
        }
        return { {}, 0.0f, 0, false };
    };

    if (!cache_enabled_ || !cache_) {
        return compute();
    }

    const uint64_t key      = pack_od_key(source, destination);
    const uint64_t topo_ver = network.topology_version();

    {
        std::lock_guard<std::mutex> lock(cache_->mutex);
        if (cache_->network_ptr != &network || cache_->topology_version != topo_ver) {
            cache_->entries.clear();
            cache_->network_ptr      = &network;
            cache_->topology_version = topo_ver;
        } else {
            auto it = cache_->entries.find(key);
            if (it != cache_->entries.end()) {
                ++cache_->hits;
                return it->second;
            }
        }
        ++cache_->misses;
    }

    RouteResult result = compute();

    {
        std::lock_guard<std::mutex> lock(cache_->mutex);
        if (cache_->network_ptr == &network && cache_->topology_version == topo_ver) {
            if (cache_->entries.size() >= kMaxStaticCacheEntries) {
                cache_->entries.clear();
            }
            cache_->entries.emplace(key, result);
        }
    }

    return result;
}

} // namespace syntraq
