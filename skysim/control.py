"""Autopilots: turn a Command into a steering demand for this tick.

Each function returns the *acceleration the vehicle should feel* (m/s^2,
world frame, gravity compensation included), a throttle setting, and -- for
thrust-vectoring bodies -- where to point the motor. The world then works out
whether the aerodynamics can actually deliver it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import physics
from .bodies import Body, Command, Mode, heading_vector
from .vec import UP, Vec3

# Gains. Loose enough to be stable at 20-50 Hz, tight enough to look flown.
K_ALT = 0.18          # altitude error -> vertical speed demand
K_VS = 1.3            # vertical speed error -> vertical accel
K_TURN = 1.6          # heading error -> lateral accel
K_SPEED = 0.05        # speed error -> throttle
K_POS = 0.6           # position error -> velocity demand (hover/follow)
K_VEL = 1.4           # velocity error -> accel (hover/follow)


@dataclass(slots=True)
class Demand:
    accel: Vec3 = Vec3()        # desired acceleration, world frame
    throttle: float = 0.0       # 0..1
    thrust_dir: Vec3 = UP       # unit vector, used when thrust_mode == "vector"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _dir(v: Vec3, fallback: Vec3 = UP) -> Vec3:
    """Unit vector, falling back when the input is (near) zero length."""
    return v.unit() if v.mag2 > 1e-18 else fallback


def _forward(body: Body) -> Vec3:
    return _dir(body.velocity)


def vertical_demand(body: Body, target_altitude_m: float,
                    max_climb_mps: float | None = None) -> float:
    """Vertical acceleration to capture and hold an altitude (gravity included).

    The climb rate is capped at a fraction of current speed: chasing a big
    altitude error at a steep angle just bleeds energy and stalls the wing.
    """
    if max_climb_mps is None:
        max_climb_mps = max(2.0, 0.25 * body.speed_mps)
    err = target_altitude_m - body.altitude_m
    vs_want = max(-max_climb_mps, min(max_climb_mps, K_ALT * err))
    g = physics.gravity_magnitude(body.altitude_m)
    return g + K_VS * (vs_want - body.velocity.z)


def turn_demand(body: Body, desired_dir: Vec3) -> Vec3:
    """Horizontal acceleration that swings the flight path toward a direction."""
    speed = body.ground_speed_mps()
    if speed < 1e-3:
        return Vec3()
    want = Vec3(desired_dir.x, desired_dir.y, 0.0).unit()
    have = Vec3(body.velocity.x, body.velocity.y, 0.0).unit()
    err = want - have * want.dot(have)
    return err * (K_TURN * speed)


def throttle_for_speed(body: Body, target_speed_mps: float | None) -> float:
    """Proportional speed hold; full throttle when no target is given."""
    if target_speed_mps is None:
        return 1.0
    cap = min(target_speed_mps, body.profile.max_speed_mps)
    return _clamp01(0.5 + K_SPEED * (cap - body.speed_mps))


def hold_demand(body: Body, cmd: Command) -> Demand:
    """Hold heading, altitude and speed -- defaults taken from current state."""
    alt = cmd.altitude_m if cmd.altitude_m is not None else body.altitude_m
    hdg = cmd.heading_deg if cmd.heading_deg is not None else body.heading_deg()
    accel = turn_demand(body, heading_vector(hdg))
    accel = accel + UP * vertical_demand(body, alt)
    return Demand(accel, throttle_for_speed(body, cmd.speed_mps), _forward(body))


def waypoint_demand(body: Body, cmd: Command) -> Demand:
    """Steer to a point; the waypoint's z is the altitude to hold."""
    wp = cmd.waypoint or body.position
    to_wp = Vec3(wp.x - body.position.x, wp.y - body.position.y, 0.0)
    alt = cmd.altitude_m if cmd.altitude_m is not None else wp.z
    if to_wp.mag < 50.0:                       # captured: just hold station
        accel = UP * vertical_demand(body, alt)
    else:
        accel = turn_demand(body, to_wp) + UP * vertical_demand(body, alt)
    return Demand(accel, throttle_for_speed(body, cmd.speed_mps), _forward(body))


def follow_demand(body: Body, cmd: Command, target: Body | None) -> Demand:
    """Trail another object, holding a standoff distance behind it.

    Plain proportional steering toward the trail point, with a speed hold on
    the leader -- the pattern a camera drone or a formation flight uses.
    """
    if target is None or not target.active:
        return hold_demand(body, cmd)
    behind = target.velocity.unit() * (-max(0.0, cmd.standoff_m))
    want_pos = target.position + behind
    to_go = want_pos - body.position
    alt = cmd.altitude_m if cmd.altitude_m is not None else want_pos.z
    accel = turn_demand(body, to_go) + UP * vertical_demand(body, alt)
    want_speed = target.speed_mps + 0.4 * Vec3(to_go.x, to_go.y, 0.0).mag ** 0.5
    return Demand(accel, throttle_for_speed(body, want_speed), _forward(body))


def _vectored_thrust(body: Body, accel: Vec3) -> Demand:
    """Convert a desired acceleration into a (throttle, thrust direction) pair
    for a thrust-vectoring body -- there is no wing to ask for lift instead."""
    thrust_n = max(1e-6, body.profile.thrust_n)
    need = accel.mag * body.mass_kg
    return Demand(Vec3(), _clamp01(need / thrust_n), _dir(accel))


def hover_demand(body: Body, cmd: Command) -> Demand:
    """Thrust-vectoring station keeping: park over a point at an altitude."""
    wp = cmd.waypoint or body.position
    alt = cmd.altitude_m if cmd.altitude_m is not None else wp.z
    want_pos = Vec3(wp.x, wp.y, alt)
    err = want_pos - body.position
    speed_cap = body.profile.max_speed_mps
    vel_want = (err * K_POS).clamp(speed_cap)
    accel = (vel_want - body.velocity) * K_VEL
    accel = accel.clamp(body.profile.max_g * physics.G0)
    accel = accel + UP * physics.gravity_magnitude(body.altitude_m)
    return _vectored_thrust(body, accel)


def _los_rate_accel(pursuer_pos: Vec3, pursuer_vel: Vec3, target_pos: Vec3,
                    target_vel: Vec3, gain: float) -> Vec3:
    """Steering accel that nulls the line-of-sight rotation rate.

    omega = (r x v_rel) / |r|^2 is how fast the bearing to the target is
    swinging; driving it to zero puts the pursuer on a straight-line
    intercept course. This is the standard convergent-pursuit law behind
    any moving-target rendezvous or tracking task (docking, camera
    gimbals, ball-catching robots) -- aiming at a lead point instead (pure
    pursuit) looks similar but does not actually converge when the pursuer
    is much faster than the target: it overshoots and loops, forever.
    """
    r = target_pos - pursuer_pos
    rng2 = max(1.0, r.mag2)
    v_rel = target_vel - pursuer_vel
    omega = r.cross(v_rel) / rng2
    return omega.cross(pursuer_vel) * gain


def pursuit_demand(body: Body, cmd: Command, target: Body | None) -> Demand:
    """Close on a moving aircraft to attach a search-and-rescue locator beacon.

    Once within the profile's capture_radius_m the world attaches the
    drone and it stops flying itself (see World._check_captures). Steering
    is delivered as wing lift for a winged chase aircraft (line-of-sight
    guidance -- a bank-and-turn airframe cannot brake, so it has to fly a
    converging course rather than chase a point), or as vectored thrust for
    a rotorcraft-style drone, which can simply servo onto the target
    directly.
    """
    vectored = body.profile.thrust_mode == "vector"
    if target is None or not target.active:
        # Nothing to rendezvous with right now: hold position and wait.
        if vectored:
            return hover_demand(body, Command(mode=Mode.HOVER, waypoint=body.position,
                                              altitude_m=body.altitude_m))
        return hold_demand(body, Command(mode=Mode.HOLD))

    speed_cap = max(1.0, body.profile.max_speed_mps)

    if vectored:
        rng = (target.position - body.position).mag
        lead_s = min(6.0, rng / speed_cap)
        aim_point = target.position + target.velocity * lead_s
        vel_want = (aim_point - body.position).clamp(speed_cap)
        accel = (vel_want - body.velocity) * K_VEL
        accel = accel.clamp(body.profile.max_g * physics.G0)
        accel = accel + UP * physics.gravity_magnitude(body.altitude_m)
        return _vectored_thrust(body, accel)

    accel = _los_rate_accel(body.position, body.velocity, target.position,
                            target.velocity, gain=3.5)
    accel = Vec3(accel.x, accel.y, 0.0) + UP * vertical_demand(body, target.altitude_m)
    return Demand(accel, throttle_for_speed(body, speed_cap), _forward(body))


def ascent_demand(body: Body, cmd: Command) -> Demand:
    """Rocket pitch program: straight up, then a constant-attitude climb."""
    if body.age_s < cmd.pitch_over_s:
        direction = UP
    else:
        pitch = math.radians(max(-90.0, min(90.0, cmd.pitch_deg)))
        horiz = heading_vector(cmd.heading_deg if cmd.heading_deg is not None else 0.0)
        direction = (horiz * math.cos(pitch) + UP * math.sin(pitch)).unit()
    lit = body.fuel_kg > 0.0
    throttle = (cmd.throttle if cmd.throttle is not None else 1.0) if lit else 0.0
    # Once the motor is out a rocket is just a projectile; keep it nose-forward.
    if not lit and body.speed_mps > 1e-3:
        direction = body.velocity.unit()
    return Demand(Vec3(), _clamp01(throttle), direction)


def solve(body: Body, target: Body | None = None) -> Demand:
    """Dispatch on the body's command mode."""
    cmd = body.command
    if cmd.mode is Mode.BALLISTIC:
        d = Demand(Vec3(), 0.0, _forward(body))
    elif cmd.mode is Mode.HOVER:
        d = hover_demand(body, cmd)
    elif cmd.mode is Mode.ASCENT:
        d = ascent_demand(body, cmd)
    elif cmd.mode is Mode.WAYPOINT:
        d = waypoint_demand(body, cmd)
    elif cmd.mode is Mode.FOLLOW:
        d = follow_demand(body, cmd, target)
    elif cmd.mode is Mode.PURSUE:
        d = pursuit_demand(body, cmd, target)
    else:                                      # HOLD and HEADING share a law
        d = hold_demand(body, cmd)

    if cmd.throttle is not None and cmd.mode is not Mode.ASCENT:
        d = Demand(d.accel, _clamp01(cmd.throttle), d.thrust_dir)
    if not body.profile.powered:
        d = Demand(d.accel, 0.0, d.thrust_dir)
    return d
