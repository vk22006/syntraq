//
// src/routing/graph_generator.cpp
//

#include "syntraq/routing/graph_generator.h"

#include <string>
#include <vector>

namespace syntraq {

RoadNetwork make_grid_network(int cols, int rows, float spacing_m, float speed_limit_mps) {
    RoadNetwork net;
    if (cols <= 0 || rows <= 0) return net;

    std::vector<std::vector<IntersectionId>> ids(
        static_cast<size_t>(rows),
        std::vector<IntersectionId>(static_cast<size_t>(cols)));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            Vec2 pos{
                static_cast<float>(c) * spacing_m,
                static_cast<float>(r) * spacing_m
            };
            std::string label = "I(" + std::to_string(c) + "," + std::to_string(r) + ")";
            ids[static_cast<size_t>(r)][static_cast<size_t>(c)] =
                net.add_intersection(pos, std::move(label));
        }
    }

    auto add_bidirectional = [&](IntersectionId a, IntersectionId b, const std::string& label) {
        net.add_road(a, b, spacing_m, speed_limit_mps, 1, label + " ->");
        net.add_road(b, a, spacing_m, speed_limit_mps, 1, label + " <-");
    };

    // Horizontal edges
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols - 1; ++c) {
            const std::string lbl =
                "H(" + std::to_string(c) + "-" + std::to_string(c + 1) + "," + std::to_string(r) + ")";
            add_bidirectional(ids[static_cast<size_t>(r)][static_cast<size_t>(c)],
                              ids[static_cast<size_t>(r)][static_cast<size_t>(c + 1)],
                              lbl);
        }
    }

    // Vertical edges
    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols; ++c) {
            const std::string lbl =
                "V(" + std::to_string(c) + "," + std::to_string(r) + "-" + std::to_string(r + 1) + ")";
            add_bidirectional(ids[static_cast<size_t>(r)][static_cast<size_t>(c)],
                              ids[static_cast<size_t>(r + 1)][static_cast<size_t>(c)],
                              lbl);
        }
    }

    return net;
}

} // namespace syntraq
