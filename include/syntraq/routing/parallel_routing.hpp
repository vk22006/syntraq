#pragma once

//
// syntraq/routing/parallel_routing.hpp
//
// Safe deterministic multi-threaded batch pathfinding utility.
// Leverages thread-local router workspaces and read-only RoadNetwork access
// to evaluate independent O-D route queries concurrently across CPU cores.
//

#include "syntraq/routing/router.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace syntraq {

/// Evaluate a batch of (source, destination) routing queries in parallel.
/// Guarantees identical deterministic results in the exact input order as
/// sequential calls to `router.find_route_with_info(network, src, dst)`.
///
/// Safe because RoadNetwork queries are read-only const operations and
/// AStarRouter / DijkstraRouter use thread-local search workspaces.
[[nodiscard]] inline std::vector<RouteResult> find_routes_parallel(
    const IRouter&                                              router,
    const RoadNetwork&                                          network,
    const std::vector<std::pair<IntersectionId, IntersectionId>>& queries,
    uint32_t                                                    num_threads = 0)
{
    const size_t n = queries.size();
    std::vector<RouteResult> results(n);
    if (n == 0) return results;

    uint32_t workers = num_threads;
    if (workers == 0) {
        const uint32_t hw = std::thread::hardware_concurrency();
        workers = (hw > 1) ? hw : 1u;
    }

    // Avoid thread-launch overhead for small batches
    constexpr size_t kMinBatchPerThread = 4;
    if (workers <= 1 || n < kMinBatchPerThread * 2) {
        for (size_t i = 0; i < n; ++i) {
            results[i] = router.find_route_with_info(network, queries[i].first, queries[i].second);
        }
        return results;
    }

    const size_t max_useful = n / kMinBatchPerThread;
    workers = static_cast<uint32_t>(std::min<size_t>(workers, std::max<size_t>(1, max_useful)));

    std::vector<std::thread> pool;
    pool.reserve(workers - 1);

    const size_t chunk = (n + workers - 1) / workers;
    for (uint32_t w = 1; w < workers; ++w) {
        const size_t begin = static_cast<size_t>(w) * chunk;
        if (begin >= n) break;
        const size_t end = std::min(n, begin + chunk);
        pool.emplace_back([&, begin, end]() {
            for (size_t i = begin; i < end; ++i) {
                results[i] = router.find_route_with_info(network, queries[i].first, queries[i].second);
            }
        });
    }

    // Main thread processes chunk 0
    const size_t main_end = std::min(n, chunk);
    for (size_t i = 0; i < main_end; ++i) {
        results[i] = router.find_route_with_info(network, queries[i].first, queries[i].second);
    }

    for (auto& t : pool) {
        if (t.joinable()) t.join();
    }

    return results;
}

} // namespace syntraq

