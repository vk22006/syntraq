#pragma once

//
// syntraq/routing/graph_generator.h
//
// Synthetic graph generation utilities for benchmarking and scaling tests.
//

#include "syntraq/world/road_network.h"

namespace syntraq {

/// Generate a regular 2D grid network of `cols` × `rows` intersections.
/// Intersections are spaced by `spacing_m` metres horizontally and vertically.
/// Adjacent intersections are connected by bi-directional directed roads.
RoadNetwork make_grid_network(int   cols,
                              int   rows,
                              float spacing_m       = 100.0f,
                              float speed_limit_mps = 13.89f);

} // namespace syntraq
