import math

import pytest

from skysim import physics
from skysim.vec import UP, Vec3


def test_isa_sea_level():
    density, pressure, sound = physics.atmosphere(0.0)
    assert density == pytest.approx(1.225, rel=1e-3)
    assert pressure == pytest.approx(101325.0, rel=1e-4)
    assert sound == pytest.approx(340.3, rel=1e-3)


def test_isa_tropopause_and_monotonic_density():
    density, pressure, _ = physics.atmosphere(11000.0)
    assert density == pytest.approx(0.3639, rel=2e-3)
    assert pressure == pytest.approx(22632.0, rel=1e-3)
    heights = [0, 1000, 5000, 11000, 20000, 32000, 47000, 60000]
    densities = [physics.atmosphere(h)[0] for h in heights]
    assert all(a > b > 0.0 for a, b in zip(densities, densities[1:]))


def test_gravity_points_down_and_weakens_with_height():
    assert physics.gravity_magnitude(0.0) == pytest.approx(physics.G0)
    assert physics.gravity_magnitude(60000.0) < physics.G0
    assert physics.gravity_magnitude(60000.0) > 0.97 * physics.G0
    g = physics.gravity_accel(Vec3(1000.0, -2000.0, 5000.0))
    assert g.x == 0.0 and g.y == 0.0 and g.z < 0.0


def test_drag_opposes_motion_and_scales_with_v_squared():
    rho = 1.225
    slow = physics.drag_force(Vec3(50.0, 0.0, 0.0), rho, 0.3, 2.0)
    fast = physics.drag_force(Vec3(100.0, 0.0, 0.0), rho, 0.3, 2.0)
    assert slow.x < 0.0 and fast.x < 0.0
    assert fast.mag == pytest.approx(4.0 * slow.mag, rel=1e-9)


def test_lift_is_perpendicular_to_velocity():
    vel = Vec3(200.0, 0.0, 20.0)
    lift = physics.lift_force(vel, 1.0, 1.2, 25.0, up_hint=UP)
    assert lift.dot(vel) == pytest.approx(0.0, abs=1e-6)
    assert lift.z > 0.0


def test_transonic_drag_rise_peaks_near_mach_one():
    subsonic = physics.drag_coefficient(0.02, 0.5)
    transonic = physics.drag_coefficient(0.02, 1.1)
    supersonic = physics.drag_coefficient(0.02, 2.5)
    assert transonic > supersonic > subsonic


def test_terminal_velocity_matches_closed_form():
    rho = physics.atmosphere(0.0)[0]
    v = physics.terminal_velocity(100.0, 1.0, 1.0, 0.0)
    assert v == pytest.approx(math.sqrt(2 * 100.0 * physics.G0 / rho), rel=1e-12)
    assert v == pytest.approx(40.0, rel=1e-3)


def test_balloon_envelope_expands_then_saturates():
    base, ratio = 10.0, 20.0
    low = physics.balloon_volume(base, ratio, physics.atmosphere(0.0)[0])
    mid = physics.balloon_volume(base, ratio, physics.atmosphere(10000.0)[0])
    high = physics.balloon_volume(base, ratio, physics.atmosphere(40000.0)[0])
    assert low == pytest.approx(base, rel=1e-6)
    assert base < mid < high
    assert high == pytest.approx(base * ratio, rel=1e-9)


def test_buoyancy_is_up_and_zero_without_volume():
    assert physics.buoyancy_force(0.0, 1.225).mag == 0.0
    b = physics.buoyancy_force(10.0, 1.225)
    assert b.z == pytest.approx(1.225 * 10.0 * physics.G0, rel=1e-6)
