import math

import pytest

from skysim import Command, MAX_OBJECTS, Mode, Status, Vec3, World, WorldFull, heading_vector
from skysim import physics
from skysim.world import Bounds


def test_free_fall_matches_kinematics_in_vacuum():
    """No drag, no lift: the integrator must reproduce h = h0 - 1/2 g t^2."""
    world = World(dt=0.01, gravity_falloff=False, collisions=False)
    from skysim.profiles import Profile
    vacuum = Profile(name="test_mass", kind="projectile", mass_kg=10.0,
                     ref_area_m2=0.0, cd0=0.0, radius_m=0.0)
    ball = world.spawn(vacuum, Vec3(0.0, 0.0, 5000.0), Vec3())
    world.run(10.0)
    expected_z = 5000.0 - 0.5 * physics.G0 * 100.0
    assert ball.position.z == pytest.approx(expected_z, rel=1e-9)
    assert ball.velocity.z == pytest.approx(-physics.G0 * 10.0, rel=1e-9)


def test_projectile_range_matches_ballistic_formula_in_vacuum():
    world = World(dt=0.005, gravity_falloff=False, collisions=False)
    from skysim.profiles import Profile
    vacuum = Profile(name="test_shell", kind="projectile", mass_kg=5.0,
                     ref_area_m2=0.0, cd0=0.0, radius_m=0.0)
    speed, angle = 300.0, math.radians(35.0)
    shot = world.spawn(vacuum, Vec3(0.0, 0.0, 0.1),
                       Vec3(speed * math.cos(angle), 0.0, speed * math.sin(angle)))
    while shot.active and world.time_s < 120.0:
        world.step()
    expected = speed * speed * math.sin(2 * angle) / physics.G0
    assert shot.position.x == pytest.approx(expected, rel=2e-3)


def test_drag_makes_a_falling_body_reach_terminal_velocity():
    world = World(dt=0.02, collisions=False)
    sonde = world.spawn("dropsonde", Vec3(0.0, 0.0, 3000.0), Vec3())
    world.run(120.0)
    profile = sonde.profile
    v_term = physics.terminal_velocity(profile.mass_kg, profile.cd0,
                                       profile.ref_area_m2, sonde.altitude_m)
    assert abs(sonde.velocity.z) == pytest.approx(v_term, rel=0.1)
    assert abs(sonde.velocity.z) < 100.0


def test_airliner_holds_altitude_heading_and_burns_fuel():
    world = World(dt=0.05, collisions=False)
    plane = world.spawn("airliner", Vec3(0.0, 0.0, 9000.0), heading_vector(0.0) * 230.0,
                        command=Command(mode=Mode.HEADING, heading_deg=90.0,
                                        altitude_m=11000.0, speed_mps=240.0))
    start_fuel = plane.fuel_kg
    world.run(300.0)
    assert plane.altitude_m == pytest.approx(11000.0, abs=50.0)
    assert plane.heading_deg() == pytest.approx(90.0, abs=2.0)
    assert plane.speed_mps == pytest.approx(240.0, rel=0.1)
    assert plane.fuel_kg < start_fuel
    assert plane.load_factor_g < plane.profile.max_g
    assert plane.position.x > 50000.0        # it actually went east


def test_waypoint_capture():
    world = World(dt=0.05, collisions=False)
    plane = world.spawn("light_aircraft", Vec3(0.0, 0.0, 1500.0), heading_vector(0.0) * 60.0,
                        command=Command(mode=Mode.WAYPOINT,
                                        waypoint=Vec3(8000.0, 8000.0, 2000.0),
                                        speed_mps=65.0))
    world.run(400.0)
    horizontal = math.hypot(plane.position.x - 8000.0, plane.position.y - 8000.0)
    assert horizontal < 2000.0
    assert plane.altitude_m == pytest.approx(2000.0, abs=100.0)


def test_rotorcraft_hovers_over_its_waypoint():
    world = World(dt=0.02, collisions=False)
    drone = world.spawn("survey_drone", Vec3(0.0, 0.0, 5.0), Vec3(),
                        command=Command(mode=Mode.HOVER, waypoint=Vec3(500.0, 250.0, 0.0),
                                        altitude_m=120.0))
    world.run(180.0)
    assert drone.position.x == pytest.approx(500.0, abs=5.0)
    assert drone.position.y == pytest.approx(250.0, abs=5.0)
    assert drone.altitude_m == pytest.approx(120.0, abs=2.0)
    assert 0.0 < drone.throttle <= 1.0


def test_rocket_climbs_burns_out_then_falls_back():
    world = World(dt=0.02, collisions=False)
    rocket = world.spawn("sounding_rocket", Vec3(0.0, 0.0, 5.0), Vec3(0.0, 0.0, 1.0),
                         command=Command(mode=Mode.ASCENT, pitch_deg=80.0,
                                         heading_deg=90.0, pitch_over_s=10.0))
    apogee = 0.0
    while rocket.active and world.time_s < 600.0:
        world.step()
        apogee = max(apogee, rocket.altitude_m)
    assert apogee > 20000.0
    assert rocket.fuel_kg == 0.0
    assert rocket.status in (Status.DESTROYED, Status.EXITED)
    kinds = {e["kind"] for e in world.drain_events()}
    assert "fuel_exhausted" in kinds


def test_balloon_rises_bursts_and_comes_down():
    world = World(dt=0.2, collisions=False)
    balloon = world.spawn("weather_balloon", Vec3(0.0, 0.0, 50.0), Vec3(0.0, 0.0, 1.0))
    events = []
    while balloon.active and world.time_s < 20000.0:
        world.step()
        events.extend(world.drain_events())
    kinds = [e["kind"] for e in events]
    assert "balloon_burst" in kinds
    assert balloon.burst
    assert balloon.status is Status.LANDED
    burst = next(e for e in events if e["kind"] == "balloon_burst")
    assert burst["altitude_m"] == pytest.approx(30000.0, abs=100.0)


def test_wind_pushes_a_drifting_object_downwind():
    world = World(dt=0.2, wind=Vec3(20.0, 0.0, 0.0), collisions=False)
    balloon = world.spawn("weather_balloon", Vec3(0.0, 0.0, 500.0), Vec3(0.0, 0.0, 2.0))
    world.run(600.0)
    assert balloon.position.x > 5000.0


def test_proximity_alert_then_clear():
    world = World(dt=0.05, separation_m=2000.0, collisions=False)
    world.spawn("airliner", Vec3(-20000.0, 0.0, 10000.0), heading_vector(90.0) * 240.0,
                command=Command(mode=Mode.HOLD, altitude_m=10000.0, speed_mps=240.0))
    world.spawn("airliner", Vec3(20000.0, 500.0, 10000.0), heading_vector(270.0) * 240.0,
                command=Command(mode=Mode.HOLD, altitude_m=10000.0, speed_mps=240.0))
    kinds = []
    for _ in range(4000):
        world.step()
        kinds.extend(e["kind"] for e in world.drain_events())
    assert "proximity_alert" in kinds
    assert "proximity_clear" in kinds


def test_collision_destroys_both_objects():
    world = World(dt=0.02, separation_m=5000.0)
    a = world.spawn("light_aircraft", Vec3(-3000.0, 0.0, 1000.0), heading_vector(90.0) * 70.0,
                    command=Command(mode=Mode.HOLD, altitude_m=1000.0, speed_mps=70.0))
    b = world.spawn("light_aircraft", Vec3(3000.0, 0.0, 1000.0), heading_vector(270.0) * 70.0,
                    command=Command(mode=Mode.HOLD, altitude_m=1000.0, speed_mps=70.0))
    hit = None
    for _ in range(3000):
        world.step()
        for event in world.drain_events():
            if event["kind"] == "collision":
                hit = event
        if hit:
            break
    assert hit is not None
    assert a.status is Status.DESTROYED and b.status is Status.DESTROYED


def test_follow_mode_closes_on_the_leader():
    world = World(dt=0.05, collisions=False, separation_m=0.0)
    lead = world.spawn("light_aircraft", Vec3(0.0, 0.0, 1200.0), heading_vector(90.0) * 60.0,
                       command=Command(mode=Mode.HOLD, altitude_m=1200.0, speed_mps=60.0))
    chase = world.spawn("light_aircraft", Vec3(-4000.0, -3000.0, 900.0),
                        heading_vector(45.0) * 60.0,
                        command=Command(mode=Mode.FOLLOW, target_id=lead.id, standoff_m=300.0))
    world.run(700.0)
    gap = (lead.position - chase.position).mag
    assert gap < 800.0                      # settled into trail, near the standoff


def test_capacity_is_32_objects():
    world = World(collisions=False)
    for _ in range(MAX_OBJECTS):
        world.spawn("cargo_capsule", Vec3(0.0, 0.0, 20000.0), Vec3())
    assert world.count == MAX_OBJECTS
    with pytest.raises(WorldFull):
        world.spawn("cargo_capsule", Vec3(0.0, 0.0, 20000.0), Vec3())
    removed = next(iter(world.bodies))
    assert world.remove(removed)
    world.spawn("cargo_capsule", Vec3(0.0, 0.0, 20000.0), Vec3())
    assert world.count == MAX_OBJECTS


def test_objects_leaving_the_volume_are_marked_exited():
    world = World(dt=0.1, bounds=Bounds(x_min=-5000, x_max=5000, y_min=-5000, y_max=5000,
                                        ceiling_m=60000.0), collisions=False)
    plane = world.spawn("airliner", Vec3(0.0, 0.0, 10000.0), heading_vector(90.0) * 240.0,
                        command=Command(mode=Mode.HOLD, altitude_m=10000.0, speed_mps=240.0))
    world.run(120.0)
    assert plane.status is Status.EXITED
    assert any(e["kind"] == "exited_volume" for e in world.drain_events())


def test_gentle_touchdown_lands_rather_than_crashes():
    world = World(dt=0.05, collisions=False)
    glider = world.spawn("glider", Vec3(0.0, 0.0, 30.0), heading_vector(0.0) * 30.0,
                         command=Command(mode=Mode.HOLD, altitude_m=0.0, speed_mps=30.0))
    world.run(120.0)
    assert glider.status is Status.LANDED


def test_snapshot_shape_is_stable():
    world = World(dt=0.02)
    world.spawn("airliner", Vec3(0.0, 0.0, 10000.0), heading_vector(90.0) * 230.0)
    snap = world.snapshot()
    assert snap["capacity"] == MAX_OBJECTS
    assert set(snap) >= {"time_s", "dt", "count", "active", "bounds", "wind", "objects"}
    obj = snap["objects"][0]
    assert set(obj) >= {"id", "name", "kind", "status", "position", "velocity",
                        "altitude_m", "speed_mps", "mach", "command"}


def test_determinism_same_inputs_same_trajectory():
    def fly():
        world = World(dt=0.02, collisions=False)
        plane = world.spawn("light_aircraft", Vec3(0.0, 0.0, 1000.0), heading_vector(10.0) * 60.0,
                            command=Command(mode=Mode.WAYPOINT,
                                            waypoint=Vec3(5000.0, 5000.0, 1500.0)))
        world.run(200.0)
        return plane.position.as_tuple()

    assert fly() == fly()


def test_sar_drone_releases_chases_and_attaches_a_beacon():
    world = World(dt=0.05, collisions=False, separation_m=0.0)
    target = world.spawn("light_aircraft", Vec3(0.0, 0.0, 1500.0), heading_vector(90.0) * 65.0,
                         name="distressed-aircraft",
                         command=Command(mode=Mode.HOLD, altitude_m=1500.0, speed_mps=65.0))
    carrier = world.spawn("airliner", Vec3(-20000.0, -3000.0, 9000.0), heading_vector(70.0) * 230.0,
                          name="sar-mothership",
                          command=Command(mode=Mode.HOLD, altitude_m=9000.0, speed_mps=230.0))
    drone = world.release_sar_drone(carrier.id, target.id, name="beacon-1")
    assert drone.command.mode is Mode.PURSUE
    assert drone.command.target_id == target.id

    attached_event = None
    for _ in range(24000):
        world.step()
        for event in world.drain_events():
            if event["kind"] == "beacon_attached":
                attached_event = event
        if attached_event:
            break

    assert attached_event is not None
    assert attached_event["target_id"] == target.id
    assert drone.attached_to == target.id

    # It should now ride along with the target rather than fly on its own.
    world.run(120.0)
    assert (drone.position - target.position).mag == pytest.approx(0.0, abs=1e-6)
    assert drone.velocity.as_tuple() == pytest.approx(target.velocity.as_tuple())
    assert drone.status is Status.ACTIVE

    # An attached beacon is not a separate collision or traffic hazard.
    assert drone not in world.flying


def test_sar_drone_detaches_if_its_target_is_lost():
    world = World(dt=0.05, collisions=False, separation_m=0.0)
    target = world.spawn("light_aircraft", Vec3(0.0, 0.0, 1500.0), heading_vector(90.0) * 65.0,
                         command=Command(mode=Mode.HOLD, altitude_m=1500.0, speed_mps=65.0))
    carrier = world.spawn("light_aircraft", Vec3(-2000.0, 0.0, 1500.0), heading_vector(90.0) * 65.0,
                          command=Command(mode=Mode.HOLD, altitude_m=1500.0, speed_mps=65.0))
    drone = world.release_sar_drone(carrier.id, target.id)
    for _ in range(4000):
        world.step()
        if drone.attached_to:
            break
    assert drone.attached_to == target.id

    world.remove(target.id)
    world.step()
    assert drone.attached_to is None
    assert drone.command.mode is Mode.BALLISTIC
    assert drone.status is Status.ACTIVE   # keeps flying on its own, doesn't vanish


def test_release_requires_a_capable_profile_and_a_live_target():
    world = World(dt=0.05, collisions=False)
    target = world.spawn("light_aircraft", Vec3(0.0, 0.0, 1500.0), Vec3())
    carrier = world.spawn("airliner", Vec3(-5000.0, 0.0, 9000.0), heading_vector(90.0) * 230.0)

    with pytest.raises(ValueError, match="capture_radius_m"):
        world.release_sar_drone(carrier.id, target.id, profile="cargo_capsule")

    with pytest.raises(ValueError, match="not available"):
        world.release_sar_drone(carrier.id, 9999, profile="sar_drone")


def test_released_drone_does_not_collide_with_its_own_carrier():
    world = World(dt=0.02, separation_m=0.0)
    target = world.spawn("light_aircraft", Vec3(20000.0, 0.0, 1500.0), heading_vector(90.0) * 65.0,
                         command=Command(mode=Mode.HOLD, altitude_m=1500.0, speed_mps=65.0))
    carrier = world.spawn("airliner", Vec3(0.0, 0.0, 9000.0), heading_vector(90.0) * 230.0,
                          command=Command(mode=Mode.HOLD, altitude_m=9000.0, speed_mps=230.0))
    drone = world.release_sar_drone(carrier.id, target.id)
    world.step()
    assert drone.status is Status.ACTIVE
    assert carrier.status is Status.ACTIVE
