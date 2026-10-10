//
// benchmarks/bench_routing.cpp
//
// Reproducible benchmark comparing Dijkstra vs A* on synthetic grid networks.
// Measures runtime (microseconds), nodes expanded, and validates 100% path optimality.
//

#include "syntraq/routing/astar_router.h"
#include "syntraq/routing/dijkstra_router.h"
#include "syntraq/routing/graph_generator.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace syntraq {

struct BenchmarkMetrics {
    std::string name;
    double      total_time_us{ 0.0 };
    double      avg_time_us{ 0.0 };
    uint64_t    total_nodes_visited{ 0 };
    double      avg_nodes_visited{ 0.0 };
    double      total_cost{ 0.0 };
    uint32_t    queries{ 0 };
};

struct PairQuery {
    IntersectionId src;
    IntersectionId dst;
};

void run_benchmark_suite() {
    std::cout << "========================================================================================\n";
    std::cout << "               SyntraQ Milestone 5 - Routing Engine Benchmark Suite                     \n";
    std::cout << "========================================================================================\n\n";

    struct GridConfig {
        int cols;
        int rows;
        std::string label;
        int corner_runs;
        int random_runs;
    };

    const std::vector<GridConfig> configs = {
        { 10, 10, "Small Grid (10x10 = 100 nodes)",     500, 200 },
        { 25, 25, "Medium Grid (25x25 = 625 nodes)",    100, 100 },
        { 50, 50, "Large Grid (50x50 = 2,500 nodes)",    25,  50 }
    };

    DijkstraRouter dijkstra;
    AStarRouter    astar;

    for (const auto& cfg : configs) {
        RoadNetwork net = make_grid_network(cfg.cols, cfg.rows, 100.0f);
        uint32_t total_nodes = net.intersection_count();
        uint32_t total_roads = net.road_count();

        std::cout << "----------------------------------------------------------------------------------------\n";
        std::cout << " Target: " << cfg.label << " | Nodes: " << total_nodes
                  << " | Directed Edges: " << total_roads << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n";

        // ── 1. Corner-to-Corner Test ──────────────────────────────────────────
        IntersectionId corner_src = static_cast<IntersectionId>(0);
        IntersectionId corner_dst = static_cast<IntersectionId>(total_nodes - 1);

        BenchmarkMetrics d_corner{ "Dijkstra" };
        BenchmarkMetrics a_corner{ "A*" };

        // Warm up
        (void)dijkstra.find_route_with_info(net, corner_src, corner_dst);
        (void)astar.find_route_with_info(net, corner_src, corner_dst);

        // Run Dijkstra Corner
        for (int i = 0; i < cfg.corner_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = dijkstra.find_route_with_info(net, corner_src, corner_dst);
            auto t1 = std::chrono::high_resolution_clock::now();

            d_corner.total_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            d_corner.total_nodes_visited += res.nodes_visited;
            d_corner.total_cost += res.total_cost;
            d_corner.queries++;
        }
        d_corner.avg_time_us = d_corner.total_time_us / d_corner.queries;
        d_corner.avg_nodes_visited = static_cast<double>(d_corner.total_nodes_visited) / d_corner.queries;

        // Run A* Corner
        for (int i = 0; i < cfg.corner_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = astar.find_route_with_info(net, corner_src, corner_dst);
            auto t1 = std::chrono::high_resolution_clock::now();

            a_corner.total_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            a_corner.total_nodes_visited += res.nodes_visited;
            a_corner.total_cost += res.total_cost;
            a_corner.queries++;
        }
        a_corner.avg_time_us = a_corner.total_time_us / a_corner.queries;
        a_corner.avg_nodes_visited = static_cast<double>(a_corner.total_nodes_visited) / a_corner.queries;

        // ── 2. Horizontal Cross-Grid Traversal (West to East) ────────────────
        IntersectionId mid_left  = static_cast<IntersectionId>((cfg.rows / 2) * cfg.cols);
        IntersectionId mid_right = static_cast<IntersectionId>((cfg.rows / 2) * cfg.cols + (cfg.cols - 1));

        BenchmarkMetrics d_horiz{ "Dijkstra" };
        BenchmarkMetrics a_horiz{ "A*" };

        for (int i = 0; i < cfg.corner_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = dijkstra.find_route_with_info(net, mid_left, mid_right);
            auto t1 = std::chrono::high_resolution_clock::now();
            d_horiz.total_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            d_horiz.total_nodes_visited += res.nodes_visited;
            d_horiz.queries++;
        }
        d_horiz.avg_time_us = d_horiz.total_time_us / d_horiz.queries;
        d_horiz.avg_nodes_visited = static_cast<double>(d_horiz.total_nodes_visited) / d_horiz.queries;

        for (int i = 0; i < cfg.corner_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = astar.find_route_with_info(net, mid_left, mid_right);
            auto t1 = std::chrono::high_resolution_clock::now();
            a_horiz.total_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            a_horiz.total_nodes_visited += res.nodes_visited;
            a_horiz.queries++;
        }
        a_horiz.avg_time_us = a_horiz.total_time_us / a_horiz.queries;
        a_horiz.avg_nodes_visited = static_cast<double>(a_horiz.total_nodes_visited) / a_horiz.queries;

        // ── 3. Random Pairs Test ──────────────────────────────────────────────
        std::mt19937 rng(1337); // Deterministic seed for reproducible benchmarks
        std::uniform_int_distribution<uint32_t> dist(0, total_nodes - 1);

        std::vector<PairQuery> queries;
        queries.reserve(static_cast<size_t>(cfg.random_runs));
        for (int i = 0; i < cfg.random_runs; ++i) {
            IntersectionId s = static_cast<IntersectionId>(dist(rng));
            IntersectionId t = static_cast<IntersectionId>(dist(rng));
            queries.push_back({ s, t });
        }

        BenchmarkMetrics d_random{ "Dijkstra" };
        BenchmarkMetrics a_random{ "A*" };

        for (const auto& q : queries) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res_d = dijkstra.find_route_with_info(net, q.src, q.dst);
            auto t1 = std::chrono::high_resolution_clock::now();

            auto t2 = std::chrono::high_resolution_clock::now();
            auto res_a = astar.find_route_with_info(net, q.src, q.dst);
            auto t3 = std::chrono::high_resolution_clock::now();

            // Correctness check
            if (std::abs(res_d.total_cost - res_a.total_cost) > 1e-3f) {
                std::cerr << "ERROR: Cost mismatch on pair (" << static_cast<uint32_t>(q.src)
                          << " -> " << static_cast<uint32_t>(q.dst) << ")\n";
                std::exit(1);
            }

            d_random.total_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            d_random.total_nodes_visited += res_d.nodes_visited;
            d_random.queries++;

            a_random.total_time_us += std::chrono::duration<double, std::micro>(t3 - t2).count();
            a_random.total_nodes_visited += res_a.nodes_visited;
            a_random.queries++;
        }
        d_random.avg_time_us = d_random.total_time_us / d_random.queries;
        d_random.avg_nodes_visited = static_cast<double>(d_random.total_nodes_visited) / d_random.queries;

        a_random.avg_time_us = a_random.total_time_us / a_random.queries;
        a_random.avg_nodes_visited = static_cast<double>(a_random.total_nodes_visited) / a_random.queries;

        // Print Formatted Report
        std::cout << std::fixed << std::setprecision(2);
        std::cout << " [1] Corner-to-Corner Traversal (" << cfg.corner_runs << " runs):\n";
        std::cout << "     - Dijkstra : " << std::setw(8) << d_corner.avg_time_us << " us/query | "
                  << std::setw(6) << d_corner.avg_nodes_visited << " nodes visited ("
                  << (100.0 * d_corner.avg_nodes_visited / total_nodes) << "% of graph)\n";
        std::cout << "     - A*       : " << std::setw(8) << a_corner.avg_time_us << " us/query | "
                  << std::setw(6) << a_corner.avg_nodes_visited << " nodes visited ("
                  << (100.0 * a_corner.avg_nodes_visited / total_nodes) << "% of graph)\n";
        double corner_node_red = 100.0 * (d_corner.avg_nodes_visited - a_corner.avg_nodes_visited) / d_corner.avg_nodes_visited;
        double corner_speedup  = d_corner.avg_time_us / (a_corner.avg_time_us > 0.0 ? a_corner.avg_time_us : 1.0);
        std::cout << "     ==> Node Expansion Reduction: " << corner_node_red << "%\n";
        std::cout << "     ==> Speedup Factor          : " << corner_speedup << "x\n\n";

        std::cout << " [2] Horizontal Cross-Grid Traversal (" << cfg.corner_runs << " runs):\n";
        std::cout << "     - Dijkstra : " << std::setw(8) << d_horiz.avg_time_us << " us/query | "
                  << std::setw(6) << d_horiz.avg_nodes_visited << " nodes visited ("
                  << (100.0 * d_horiz.avg_nodes_visited / total_nodes) << "% of graph)\n";
        std::cout << "     - A*       : " << std::setw(8) << a_horiz.avg_time_us << " us/query | "
                  << std::setw(6) << a_horiz.avg_nodes_visited << " nodes visited ("
                  << (100.0 * a_horiz.avg_nodes_visited / total_nodes) << "% of graph)\n";
        double horiz_node_red = 100.0 * (d_horiz.avg_nodes_visited - a_horiz.avg_nodes_visited) / (d_horiz.avg_nodes_visited > 0.0 ? d_horiz.avg_nodes_visited : 1.0);
        double horiz_speedup  = d_horiz.avg_time_us / (a_horiz.avg_time_us > 0.0 ? a_horiz.avg_time_us : 1.0);
        std::cout << "     ==> Node Expansion Reduction: " << horiz_node_red << "%\n";
        std::cout << "     ==> Speedup Factor          : " << horiz_speedup << "x\n\n";

        std::cout << " [3] Random Pairs Workload (" << cfg.random_runs << " queries):\n";
        std::cout << "     - Dijkstra : " << std::setw(8) << d_random.avg_time_us << " us/query | "
                  << std::setw(6) << d_random.avg_nodes_visited << " avg nodes visited\n";
        std::cout << "     - A*       : " << std::setw(8) << a_random.avg_time_us << " us/query | "
                  << std::setw(6) << a_random.avg_nodes_visited << " avg nodes visited\n";
        double rand_node_red = 100.0 * (d_random.avg_nodes_visited - a_random.avg_nodes_visited) / (d_random.avg_nodes_visited > 0.0 ? d_random.avg_nodes_visited : 1.0);
        double rand_speedup  = d_random.avg_time_us / (a_random.avg_time_us > 0.0 ? a_random.avg_time_us : 1.0);
        std::cout << "     ==> Node Expansion Reduction: " << rand_node_red << "%\n";
        std::cout << "     ==> Speedup Factor          : " << rand_speedup << "x\n";
        std::cout << "     ==> Optimality Verification : 100% MATCH (Dijkstra vs A* path costs)\n\n";
    }

    std::cout << "========================================================================================\n";
    std::cout << " Benchmark completed successfully. All path optimality checks passed.\n";
    std::cout << "========================================================================================\n";
}

} // namespace syntraq

int main() {
    syntraq::run_benchmark_suite();
    return 0;
}
