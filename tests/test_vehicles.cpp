//
// tests/test_vehicles.cpp
//
// Unit tests for the vehicle system (Vehicle, VehicleMovementSystem, VehicleSpawner).
// All tests are headless — no Raylib, no rendering.
//

#include "syntraq/vehicles/vehicle.h"
#include "syntraq/vehicles/vehicle_movement_system.h"
#include "syntraq/vehicles/vehicle_spawner.h"
#include "syntraq/world/road_network.h"

#include <gtest/gtest.h>

namespace syntraq {

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a simple 3-intersection linear network:  A --road_ab--> B --road_bc--> C
/// Each road is 100 m long, 2 lanes, 50 km/h.
struct LinearNet {
    RoadNetwork    net;
    IntersectionId a, b, c;
    RoadId         road_ab, road_bc;

    LinearNet() {
        a = net.add_intersection({ 0,   0 }, "A");
        b = net.add_intersection({ 100, 0 }, "B");
        c = net.add_intersection({ 200, 0 }, "C");
        road_ab = net.add_road(a, b, 100.f, 13.89f, 2, "AB");
        road_bc = net.add_road(b, c, 100.f, 13.89f, 2, "BC");
    }
};

/// Build a minimal single-road network: A --road--> B (200 m, 1 lane, 10 m/s)
struct SingleRoadNet {
    RoadNetwork    net;
    IntersectionId a, b;
    RoadId         road;

    SingleRoadNet() {
        a    = net.add_intersection({ 0,   0 }, "A");
        b    = net.add_intersection({ 200, 0 }, "B");
        road = net.add_road(a, b, 200.f, 10.f, 1, "AB");
    }
};

/// Construct a Vehicle on road_ab with route [road_ab, road_bc].
Vehicle make_vehicle(VehicleId id,
                     RoadId    first,
                     RoadId    second,
                     IntersectionId dst) {
    Vehicle v;
    v.id          = id;
    v.route       = { first, second };
    v.route_index = 0;
    v.destination = dst;
    v.current_road = first;
    v.progress_m  = 0.f;
    v.speed_mps   = 0.f;
    v.max_speed_mps = 13.89f;
    v.accel_mps2  = 3.f;
    v.decel_mps2  = 5.f;
    v.state       = VehicleState::Moving;
    return v;
}

// ═══════════════════════════════════════════════════════════════════
// Vehicle struct tests
// ═══════════════════════════════════════════════════════════════════

TEST(VehicleTest, DefaultIsInvalid) {
    Vehicle v;
    EXPECT_FALSE(v.is_valid());
    EXPECT_EQ(v.id, kInvalidVehicleId);
}

TEST(VehicleTest, ValidVehicleHasRoute) {
    LinearNet ln;
    auto v = make_vehicle(static_cast<VehicleId>(0), ln.road_ab, ln.road_bc, ln.c);
    EXPECT_TRUE(v.is_valid());
    EXPECT_EQ(v.roads_remaining(), 2u);
    EXPECT_FALSE(v.has_arrived());
}

TEST(VehicleTest, InitialStateIsMoving) {
    LinearNet ln;
    auto v = make_vehicle(static_cast<VehicleId>(0), ln.road_ab, ln.road_bc, ln.c);
    EXPECT_EQ(v.state, VehicleState::Moving);
    EXPECT_FLOAT_EQ(v.speed_mps, 0.f);
    EXPECT_FLOAT_EQ(v.progress_m, 0.f);
}

// ═══════════════════════════════════════════════════════════════════
// VehicleMovementSystem — single tick
// ═══════════════════════════════════════════════════════════════════

TEST(MovementSystemTest, FirstTickAccelerates) {
    SingleRoadNet sn;
    VehicleMovementSystem sys;

    Vehicle v;
    v.id          = static_cast<VehicleId>(0);
    v.route       = { sn.road };
    v.route_index = 0;
    v.destination = sn.b;
    v.current_road = sn.road;
    v.progress_m  = 0.f;
    v.speed_mps   = 0.f;
    v.max_speed_mps = 10.f;
    v.accel_mps2  = 3.f;
    v.decel_mps2  = 5.f;
    v.state       = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, sn.net, 1.0f);  // dt = 1 s

    // After 1 s: speed = min(10, 0 + 3*1) = 3 m/s; progress = 3 m
    EXPECT_FLOAT_EQ(vehicles[0].speed_mps,  3.f);
    EXPECT_FLOAT_EQ(vehicles[0].progress_m, 3.f);
    EXPECT_EQ(vehicles[0].state, VehicleState::Moving);
}

TEST(MovementSystemTest, SpeedCapsAtMaxSpeed) {
    SingleRoadNet sn;
    VehicleMovementSystem sys;

    Vehicle v;
    v.id = static_cast<VehicleId>(0);
    v.route = { sn.road };
    v.route_index = 0;
    v.destination = sn.b;
    v.current_road = sn.road;
    v.progress_m = 0.f;
    v.speed_mps  = 9.5f;   // Already near cap
    v.max_speed_mps = 10.f;
    v.accel_mps2 = 3.f;
    v.decel_mps2 = 5.f;
    v.state = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, sn.net, 1.0f);

    // speed = min(10, 9.5 + 3) = 10
    EXPECT_FLOAT_EQ(vehicles[0].speed_mps, 10.f);
}

TEST(MovementSystemTest, ProgressAdvancesEachTick) {
    SingleRoadNet sn;
    VehicleMovementSystem sys;

    Vehicle v;
    v.id = static_cast<VehicleId>(0);
    v.route = { sn.road };
    v.route_index = 0;
    v.destination = sn.b;
    v.current_road = sn.road;
    v.progress_m = 0.f;
    v.speed_mps  = 10.f;  // Already at max
    v.max_speed_mps = 10.f;
    v.accel_mps2 = 3.f;
    v.decel_mps2 = 5.f;
    v.state = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };

    sys.update(vehicles, sn.net, 0.5f);
    EXPECT_FLOAT_EQ(vehicles[0].progress_m, 5.f);

    sys.update(vehicles, sn.net, 0.5f);
    EXPECT_FLOAT_EQ(vehicles[0].progress_m, 10.f);
}

TEST(MovementSystemTest, DistanceAndTimeAccumulate) {
    SingleRoadNet sn;
    VehicleMovementSystem sys;

    Vehicle v;
    v.id = static_cast<VehicleId>(0);
    v.route = { sn.road };
    v.route_index = 0;
    v.destination = sn.b;
    v.current_road = sn.road;
    v.progress_m = 0.f;
    v.speed_mps = 10.f;
    v.max_speed_mps = 10.f;
    v.accel_mps2 = 3.f;
    v.decel_mps2 = 5.f;
    v.state = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, sn.net, 2.0f);

    EXPECT_FLOAT_EQ(vehicles[0].distance_m,    20.f);
    EXPECT_FLOAT_EQ(vehicles[0].travel_time_s,  2.f);
}

// ═══════════════════════════════════════════════════════════════════
// VehicleMovementSystem — road transitions
// ═══════════════════════════════════════════════════════════════════

TEST(MovementSystemTest, TransitionsToNextRoadAtEnd) {
    LinearNet ln;
    VehicleMovementSystem sys;

    auto v = make_vehicle(static_cast<VehicleId>(0), ln.road_ab, ln.road_bc, ln.c);
    // Road AB limit is 13.89 m/s. Set speed at limit so step = 13.89 * dt.
    // dt = 8 s → step = 111.12 m. Road AB is 100 m → overshoot = 11.12 m onto BC.
    v.speed_mps = 13.89f;
    v.max_speed_mps = 13.89f;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, ln.net, 8.0f);

    EXPECT_EQ(vehicles[0].current_road, ln.road_bc);
    // Overshoot = 13.89*8 - 100 = 111.12 - 100 = 11.12, clamped to BC length (100)
    EXPECT_GT(vehicles[0].progress_m, 0.f);
    EXPECT_LT(vehicles[0].progress_m, 100.f);
    EXPECT_EQ(vehicles[0].state, VehicleState::Moving);
}

TEST(MovementSystemTest, ArrivesWhenRouteExhausted) {
    // Build a tiny network with a very high speed limit so we can cover
    // both roads in a single tick without clamping.
    RoadNetwork net;
    auto a = net.add_intersection({ 0,   0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 200, 0 });
    auto rab = net.add_road(a, b, 100.f, 200.f, 1);  // 200 m/s limit
    auto rbc = net.add_road(b, c, 100.f, 200.f, 1);

    VehicleMovementSystem sys;

    Vehicle v;
    v.id = static_cast<VehicleId>(0);
    v.route = { rab, rbc };
    v.route_index = 0;
    v.destination = c;
    v.current_road = rab;
    v.progress_m = 0.f;
    v.speed_mps = 200.f;
    v.max_speed_mps = 200.f;
    v.accel_mps2 = 3.f;
    v.decel_mps2 = 5.f;
    v.state = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    // dt = 2 s → step = 400 m. AB (100 m) + BC (100 m) = 200 m total → arrive.
    sys.update(vehicles, net, 2.0f);

    EXPECT_EQ(vehicles[0].state, VehicleState::Arrived);
    EXPECT_TRUE(vehicles[0].has_arrived());
}

TEST(MovementSystemTest, ArrivedVehicleIsSkipped) {
    LinearNet ln;
    VehicleMovementSystem sys;

    auto v = make_vehicle(static_cast<VehicleId>(0), ln.road_ab, ln.road_bc, ln.c);
    v.state       = VehicleState::Arrived;
    v.speed_mps   = 0.f;
    v.progress_m  = 99.f;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, ln.net, 1.0f);

    // Arrived vehicle must not move
    EXPECT_FLOAT_EQ(vehicles[0].progress_m, 99.f);
    EXPECT_EQ(vehicles[0].state, VehicleState::Arrived);
}

// ═══════════════════════════════════════════════════════════════════
// VehicleMovementSystem — multiple vehicles
// ═══════════════════════════════════════════════════════════════════

TEST(MovementSystemTest, UpdatesAllVehiclesIndependently) {
    LinearNet ln;
    VehicleMovementSystem sys;

    auto v0 = make_vehicle(static_cast<VehicleId>(0), ln.road_ab, ln.road_bc, ln.c);
    auto v1 = make_vehicle(static_cast<VehicleId>(1), ln.road_ab, ln.road_bc, ln.c);
    v0.speed_mps = 10.f; v0.max_speed_mps = 10.f;
    v1.speed_mps = 5.f;  v1.max_speed_mps = 5.f;

    std::vector<Vehicle> vehicles{ v0, v1 };
    sys.update(vehicles, ln.net, 1.0f);

    EXPECT_GT(vehicles[0].progress_m, vehicles[1].progress_m);
}

// ═══════════════════════════════════════════════════════════════════
// VehicleSpawner — route building
// ═══════════════════════════════════════════════════════════════════

TEST(SpawnerTest, BuildRouteLinearChain) {
    LinearNet ln;
    VehicleSpawner spawner(42);

    auto route = spawner.build_route(ln.net, ln.a, ln.c);
    ASSERT_EQ(route.size(), 2u);
    EXPECT_EQ(route[0], ln.road_ab);
    EXPECT_EQ(route[1], ln.road_bc);
}

TEST(SpawnerTest, BuildRouteDirectConnection) {
    LinearNet ln;
    VehicleSpawner spawner(42);

    auto route = spawner.build_route(ln.net, ln.a, ln.b);
    ASSERT_EQ(route.size(), 1u);
    EXPECT_EQ(route[0], ln.road_ab);
}

TEST(SpawnerTest, BuildRouteReturnsEmptyForUnreachable) {
    // A -> B, C is isolated — no path from A to C
    RoadNetwork net;
    auto a = net.add_intersection({ 0, 0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 200, 0 });
    net.add_road(a, b, 100.f);
    // No road from b to c

    VehicleSpawner spawner(42);
    auto route = spawner.build_route(net, a, c);
    EXPECT_TRUE(route.empty());
}

TEST(SpawnerTest, BuildRouteReturnsEmptyForSelfDestination) {
    LinearNet ln;
    VehicleSpawner spawner(42);

    auto route = spawner.build_route(ln.net, ln.a, ln.a);
    EXPECT_TRUE(route.empty());
}

// ═══════════════════════════════════════════════════════════════════
// VehicleSpawner — try_spawn
// ═══════════════════════════════════════════════════════════════════

TEST(SpawnerTest, SpawnsFirstVehicleImmediately) {
    auto net = make_test_map();
    VehicleSpawner spawner(42, 2.0f, 50);

    auto opt = spawner.try_spawn(net, 0, 0.f);
    EXPECT_TRUE(opt.has_value());
    if (opt) {
        EXPECT_TRUE(opt->is_valid());
        EXPECT_EQ(opt->state, VehicleState::Moving);
        EXPECT_FLOAT_EQ(opt->progress_m, 0.f);
        EXPECT_FLOAT_EQ(opt->speed_mps,  0.f);
    }
}

TEST(SpawnerTest, DoesNotSpawnBeforeInterval) {
    auto net = make_test_map();
    VehicleSpawner spawner(42, 2.0f, 50);

    spawner.try_spawn(net, 0, 0.f);                    // first spawn at t=0
    auto opt = spawner.try_spawn(net, 1, 1.0f);        // only 1 s later
    EXPECT_FALSE(opt.has_value());
}

TEST(SpawnerTest, SpawnsAfterInterval) {
    // Use a small deterministic network (A→B→C) so routes always exist.
    RoadNetwork net;
    auto a = net.add_intersection({ 0,   0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 200, 0 });
    net.add_road(a, b, 100.f);
    net.add_road(b, a, 100.f);
    net.add_road(b, c, 100.f);
    net.add_road(c, b, 100.f);

    VehicleSpawner spawner(42, 2.0f, 50);
    spawner.try_spawn(net, 0, 0.f);                     // first spawn
    auto opt = spawner.try_spawn(net, 1, 2.1f);         // past interval
    EXPECT_TRUE(opt.has_value());
}

TEST(SpawnerTest, DoesNotExceedMaxVehicles) {
    auto net = make_test_map();
    VehicleSpawner spawner(42, 2.0f, 5);

    // Pretend 5 vehicles are already active
    auto opt = spawner.try_spawn(net, 5, 0.f);
    EXPECT_FALSE(opt.has_value());
}

TEST(SpawnerTest, SpawnedVehicleHasValidRoute) {
    auto net = make_test_map();
    VehicleSpawner spawner(42, 2.0f, 50);

    auto opt = spawner.try_spawn(net, 0, 0.f);
    ASSERT_TRUE(opt.has_value());
    EXPECT_FALSE(opt->route.empty());
    EXPECT_NE(opt->current_road, kInvalidRoadId);
    EXPECT_EQ(opt->current_road, opt->route[0]);
}

TEST(SpawnerTest, MultipleSpawnsGetDistinctIds) {
    // 3-node fully-connected ring so any src!=dst pair is reachable.
    RoadNetwork net;
    auto a = net.add_intersection({ 0,   0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 100, 100 });
    net.add_road(a, b, 100.f);
    net.add_road(b, a, 100.f);
    net.add_road(b, c, 100.f);
    net.add_road(c, b, 100.f);
    net.add_road(a, c, 141.f);
    net.add_road(c, a, 141.f);

    // Use a large negative interval so consecutive spawns both succeed.
    VehicleSpawner spawner(42, -100.f, 50);

    auto v0 = spawner.try_spawn(net, 0, 0.f);
    auto v1 = spawner.try_spawn(net, 1, 1.f);
    ASSERT_TRUE(v0.has_value());
    ASSERT_TRUE(v1.has_value());
    EXPECT_NE(v0->id, v1->id);
}

// ═══════════════════════════════════════════════════════════════════
// Acceleration / deceleration boundary cases
// ═══════════════════════════════════════════════════════════════════

TEST(MovementSystemTest, VehicleDeceleratesIfOverSpeedLimit) {
    SingleRoadNet sn;
    VehicleMovementSystem sys;

    Vehicle v;
    v.id = static_cast<VehicleId>(0);
    v.route = { sn.road };
    v.route_index = 0;
    v.destination = sn.b;
    v.current_road = sn.road;
    v.progress_m = 0.f;
    v.speed_mps  = 15.f;   // Over the road's 10 m/s limit
    v.max_speed_mps = 15.f;
    v.accel_mps2 = 3.f;
    v.decel_mps2 = 5.f;
    v.state = VehicleState::Moving;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, sn.net, 1.0f);

    // Road limit is 10 m/s → decelerate: 15 - 5*1 = 10
    EXPECT_FLOAT_EQ(vehicles[0].speed_mps, 10.f);
}

TEST(MovementSystemTest, StoppedVehicleDoesNotMove) {
    SingleRoadNet sn;
    VehicleMovementSystem sys;

    Vehicle v;
    v.id = static_cast<VehicleId>(0);
    v.route = { sn.road };
    v.route_index = 0;
    v.destination = sn.b;
    v.current_road = sn.road;
    v.progress_m = 50.f;
    v.speed_mps  = 0.f;
    v.max_speed_mps = 10.f;
    v.accel_mps2 = 0.f;  // No acceleration
    v.decel_mps2 = 5.f;
    v.state = VehicleState::Stopped;

    std::vector<Vehicle> vehicles{ v };
    sys.update(vehicles, sn.net, 5.0f);

    EXPECT_FLOAT_EQ(vehicles[0].progress_m, 50.f);
    EXPECT_EQ(vehicles[0].state, VehicleState::Stopped);
}

// ═══════════════════════════════════════════════════════════════════
// Route Provider abstraction tests
// ═══════════════════════════════════════════════════════════════════

TEST(RouteProviderTest, GreedyRouteProviderDirect) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0,   0 });
    auto b = net.add_intersection({ 100, 0 });
    auto c = net.add_intersection({ 200, 0 });
    auto rab = net.add_road(a, b, 100.f);
    auto rbc = net.add_road(b, c, 100.f);

    GreedyRouteProvider provider;
    auto route = provider.find_route(net, a, c);
    ASSERT_EQ(route.size(), 2u);
    EXPECT_EQ(route[0], rab);
    EXPECT_EQ(route[1], rbc);
}

TEST(RouteProviderTest, GreedyRouteProviderUnreachableReturnsEmpty) {
    RoadNetwork net;
    auto a = net.add_intersection({ 0,   0 });
    auto b = net.add_intersection({ 100, 0 });
    // No road between a and b
    GreedyRouteProvider provider;
    auto route = provider.find_route(net, a, b);
    EXPECT_TRUE(route.empty());
}

} // namespace syntraq

