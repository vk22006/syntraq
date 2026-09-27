//
// tests/test_road_network.cpp
//
// Unit tests for the RoadNetwork directed graph.
// Covers: construction, valid/invalid additions, queries, connectivity,
// make_test_map(), and the Road/Intersection types.
//

#include "syntraq/world/road_network.h"
#include "syntraq/world/road.h"
#include "syntraq/world/intersection.h"

#include <gtest/gtest.h>

namespace syntraq {

// ═══════════════════════════════════════════════════════════════════
// Intersection tests
// ═══════════════════════════════════════════════════════════════════

TEST(IntersectionTest, DefaultIsInvalid) {
    Intersection node;
    EXPECT_FALSE(node.is_valid());
    EXPECT_EQ(node.id, kInvalidIntersectionId);
}

TEST(IntersectionTest, OutgoingAndIncomingStartEmpty) {
    Intersection node;
    EXPECT_TRUE(node.outgoing_roads.empty());
    EXPECT_TRUE(node.incoming_roads.empty());
}

// ═══════════════════════════════════════════════════════════════════
// Road / Lane tests
// ═══════════════════════════════════════════════════════════════════

TEST(RoadTest, DefaultIsInvalid) {
    Road r;
    EXPECT_FALSE(r.is_valid());
}

TEST(RoadTest, LaneCountAndWidthHelpers) {
    Road r;
    r.id     = static_cast<RoadId>(0);
    r.from   = static_cast<IntersectionId>(0);
    r.to     = static_cast<IntersectionId>(1);
    r.length_m = 100.f;
    r.lanes  = { Lane{0, 3.5f}, Lane{1, 3.5f} };
    EXPECT_EQ(r.lane_count(), 2u);
    EXPECT_FLOAT_EQ(r.total_width_m(), 7.f);
    EXPECT_TRUE(r.is_valid());
}

// ═══════════════════════════════════════════════════════════════════
// RoadNetwork — empty network
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, EmptyNetworkHasNoNodes) {
    RoadNetwork net;
    EXPECT_EQ(net.intersection_count(), 0u);
    EXPECT_EQ(net.road_count(),         0u);
}

TEST(RoadNetworkTest, QueryOnEmptyReturnsNull) {
    RoadNetwork net;
    EXPECT_EQ(net.intersection(static_cast<IntersectionId>(0)), nullptr);
    EXPECT_EQ(net.road(static_cast<RoadId>(0)),                 nullptr);
}

// ═══════════════════════════════════════════════════════════════════
// RoadNetwork — adding intersections
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, AddIntersectionReturnsValidId) {
    RoadNetwork net;
    auto id = net.add_intersection({ 100.f, 200.f }, "A");
    EXPECT_NE(id, kInvalidIntersectionId);
    EXPECT_EQ(net.intersection_count(), 1u);
}

TEST(RoadNetworkTest, AddedIntersectionIsQueryable) {
    RoadNetwork net;
    auto id = net.add_intersection({ 50.f, 75.f }, "P");
    const Intersection* node = net.intersection(id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->id,          id);
    EXPECT_FLOAT_EQ(node->position.x, 50.f);
    EXPECT_FLOAT_EQ(node->position.y, 75.f);
    EXPECT_EQ(node->name,        "P");
    EXPECT_TRUE(node->is_valid());
}

TEST(RoadNetworkTest, MultipleIntersectionsGetDistinctIds) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 200, 0 });
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
    EXPECT_EQ(net.intersection_count(), 3u);
}

// ═══════════════════════════════════════════════════════════════════
// RoadNetwork — adding roads
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, AddRoadBetweenValidIntersections) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });

    auto rid = net.add_road(a, b, 100.f, 13.89f, 2);
    EXPECT_NE(rid, kInvalidRoadId);
    EXPECT_EQ(net.road_count(), 1u);
}

TEST(RoadNetworkTest, AddedRoadIsQueryable) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 200, 0 });

    auto rid = net.add_road(a, b, 200.f, 13.89f, 1, "Main St");
    const Road* r = net.road(rid);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->from,       a);
    EXPECT_EQ(r->to,         b);
    EXPECT_FLOAT_EQ(r->length_m, 200.f);
    EXPECT_EQ(r->lane_count(), 1u);
    EXPECT_EQ(r->name,       "Main St");
    EXPECT_TRUE(r->is_valid());
}

TEST(RoadNetworkTest, RoadWiresAdjacencyLists) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    auto rid = net.add_road(a, b, 100.f);

    // `a` should list the road as outgoing
    auto out = net.outgoing_roads(a);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], rid);

    // `b` should list the road as incoming
    auto inc = net.incoming_roads(b);
    ASSERT_EQ(inc.size(), 1u);
    EXPECT_EQ(inc[0], rid);
}

// ═══════════════════════════════════════════════════════════════════
// RoadNetwork — invalid / rejected operations
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, RejectRoadWithNonExistentFrom) {
    RoadNetwork net;
    auto b = net.add_intersection({ 100, 0 });
    auto rid = net.add_road(kInvalidIntersectionId, b, 100.f);
    EXPECT_EQ(rid, kInvalidRoadId);
    EXPECT_EQ(net.road_count(), 0u);
}

TEST(RoadNetworkTest, RejectRoadWithNonExistentTo) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto rid = net.add_road(a, kInvalidIntersectionId, 100.f);
    EXPECT_EQ(rid, kInvalidRoadId);
}

TEST(RoadNetworkTest, RejectSelfLoopRoad) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto rid = net.add_road(a, a, 100.f);
    EXPECT_EQ(rid, kInvalidRoadId);
}

TEST(RoadNetworkTest, RejectDuplicateDirectedRoad) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    net.add_road(a, b, 100.f);

    // Adding the same direction again must fail
    auto rid2 = net.add_road(a, b, 100.f);
    EXPECT_EQ(rid2, kInvalidRoadId);
    EXPECT_EQ(net.road_count(), 1u);
}

TEST(RoadNetworkTest, ReverseDirectionIsAccepted) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    net.add_road(a, b, 100.f);

    // b→a is a different directed edge — should succeed
    auto rid2 = net.add_road(b, a, 100.f);
    EXPECT_NE(rid2, kInvalidRoadId);
    EXPECT_EQ(net.road_count(), 2u);
}

TEST(RoadNetworkTest, RejectZeroLengthRoad) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    auto rid = net.add_road(a, b, 0.f);
    EXPECT_EQ(rid, kInvalidRoadId);
}

TEST(RoadNetworkTest, RejectZeroLaneRoad) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    auto rid = net.add_road(a, b, 100.f, 13.89f, 0);
    EXPECT_EQ(rid, kInvalidRoadId);
}

// ═══════════════════════════════════════════════════════════════════
// RoadNetwork — has_* queries
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, HasIntersectionReturnsTrueForExisting) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    EXPECT_TRUE(net.has_intersection(a));
    EXPECT_FALSE(net.has_intersection(kInvalidIntersectionId));
}

TEST(RoadNetworkTest, HasRoadDirectionAware) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    net.add_road(a, b, 100.f);

    EXPECT_TRUE (net.has_road(a, b));
    EXPECT_FALSE(net.has_road(b, a));  // reverse not added
}

// ═══════════════════════════════════════════════════════════════════
// RoadNetwork — clear
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, ClearResetsNetwork) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    net.add_road(a, b, 100.f);

    net.clear();
    EXPECT_EQ(net.intersection_count(), 0u);
    EXPECT_EQ(net.road_count(),         0u);

    // IDs restart from 0 after clear
    auto c = net.add_intersection({ 50, 50 });
    EXPECT_EQ(static_cast<uint32_t>(c), 0u);
}

// ═══════════════════════════════════════════════════════════════════
// make_test_map — structural invariants
// ═══════════════════════════════════════════════════════════════════

TEST(TestMapTest, CorrectIntersectionCount) {
    auto net = make_test_map();
    // 4 columns × 3 rows = 12 intersections
    EXPECT_EQ(net.intersection_count(), 12u);
}

TEST(TestMapTest, CorrectRoadCount) {
    auto net = make_test_map();
    // Horizontal edges: 3 per row × 3 rows = 9, × 2 directions = 18
    // Vertical edges:   4 per col × 2 gaps = 8, × 2 directions = 16
    // Total = 34
    EXPECT_EQ(net.road_count(), 34u);
}

TEST(TestMapTest, AllRoadsAreValid) {
    auto net = make_test_map();
    for (const auto& [rid, r] : net.roads()) {
        EXPECT_TRUE(r.is_valid()) << "Road " << static_cast<uint32_t>(rid)
                                  << " is invalid";
    }
}

TEST(TestMapTest, AllIntersectionsAreValid) {
    auto net = make_test_map();
    for (const auto& [iid, node] : net.intersections()) {
        EXPECT_TRUE(node.is_valid()) << "Intersection "
                                     << static_cast<uint32_t>(iid)
                                     << " is invalid";
    }
}

TEST(TestMapTest, EachIntersectionHasOutgoingRoads) {
    auto net = make_test_map();
    for (const auto& [iid, node] : net.intersections()) {
        EXPECT_FALSE(node.outgoing_roads.empty())
            << "Intersection " << node.name << " has no outgoing roads";
    }
}

TEST(TestMapTest, CornerIntersectionHasTwoOutgoing) {
    // Corner nodes in a 4×3 grid have degree 2 (outgoing)
    auto net = make_test_map();
    // Intersection 0 = top-left corner → right and down
    const Intersection* corner = net.intersection(static_cast<IntersectionId>(0));
    ASSERT_NE(corner, nullptr);
    EXPECT_EQ(corner->outgoing_roads.size(), 2u);
}

TEST(TestMapTest, InteriorIntersectionHasFourOutgoing) {
    // An interior node (not on any edge) has 4 outgoing roads
    // Intersection at (1,1) = index 5 in row-major order (row=1, col=1)
    auto net = make_test_map();
    const Intersection* interior = net.intersection(static_cast<IntersectionId>(5));
    ASSERT_NE(interior, nullptr);
    EXPECT_EQ(interior->outgoing_roads.size(), 4u);
}

TEST(TestMapTest, BidirectionalRoadsExist) {
    auto net = make_test_map();
    // Intersection 0 and 1 should have roads in both directions
    auto id0 = static_cast<IntersectionId>(0);
    auto id1 = static_cast<IntersectionId>(1);
    EXPECT_TRUE(net.has_road(id0, id1));
    EXPECT_TRUE(net.has_road(id1, id0));
}

TEST(TestMapTest, RoadsHaveTwoLanes) {
    auto net = make_test_map();
    for (const auto& [rid, r] : net.roads()) {
        EXPECT_EQ(r.lane_count(), 2u)
            << "Road " << r.name << " should have 2 lanes";
    }
}

TEST(TestMapTest, AllRoadsHavePositiveLength) {
    auto net = make_test_map();
    for (const auto& [rid, r] : net.roads()) {
        EXPECT_GT(r.length_m, 0.f);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Graph connectivity — simple reachability check
// ═══════════════════════════════════════════════════════════════════

TEST(RoadNetworkTest, ConnectedGraphIsTraversable) {
    // Build a simple 3-node linear chain: A → B → C
    RoadNetwork net;
    auto a = net.add_intersection({ 0,   0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 200, 0 });
    net.add_road(a, b, 100.f);
    net.add_road(b, c, 100.f);

    // From A, follow outgoing roads and reach C
    auto from_a = net.outgoing_roads(a);
    ASSERT_EQ(from_a.size(), 1u);
    const Road* ab = net.road(from_a[0]);
    ASSERT_NE(ab, nullptr);
    EXPECT_EQ(ab->to, b);

    auto from_b = net.outgoing_roads(b);
    ASSERT_EQ(from_b.size(), 1u);
    const Road* bc = net.road(from_b[0]);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->to, c);
}

} // namespace syntraq
