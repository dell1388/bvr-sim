"""The simulation world: fixed-step RK4 over up to 32 objects.

One flat block of airspace (default 400 x 400 km, surface to 60 km), every
object a point mass obeying the same force model. Step it, read the state,
repeat -- that is the whole engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import control, physics, profiles
from .bodies import MAX_OBJECTS, Body, Command, Mode, Status
from .profiles import Profile
from .vec import UP, Vec3

DEFAULT_DT = 0.02             # s, 50 Hz
DEFAULT_CEILING_M = 60000.0
DEFAULT_EXTENT_M = 200000.0   # +/- this in x and y
SURVIVABLE_TOUCHDOWN_MPS = 12.0
MAX_EVENTS = 4096


class WorldFull(Exception):
    """Raised when spawning past MAX_OBJECTS."""


@dataclass(frozen=True, slots=True)
class Bounds:
    """The simulated volume. Anything outside it is dropped as 'exited'."""

    x_min: float = -DEFAULT_EXTENT_M
    x_max: float = DEFAULT_EXTENT_M
    y_min: float = -DEFAULT_EXTENT_M
    y_max: float = DEFAULT_EXTENT_M
    ceiling_m: float = DEFAULT_CEILING_M

    def contains(self, p: Vec3) -> bool:
        return (self.x_min <= p.x <= self.x_max
                and self.y_min <= p.y <= self.y_max
                and p.z <= self.ceiling_m)

    def to_dict(self) -> dict:
        return {"x_min": self.x_min, "x_max": self.x_max, "y_min": self.y_min,
                "y_max": self.y_max, "ceiling_m": self.ceiling_m}


@dataclass(slots=True)
class Event:
    t: float
    kind: str
    data: dict

    def to_dict(self) -> dict:
        return {"t": round(self.t, 3), "kind": self.kind, **self.data}


@dataclass
class World:
    dt: float = DEFAULT_DT
    time_s: float = 0.0
    bounds: Bounds = field(default_factory=Bounds)
    wind: Vec3 = field(default_factory=Vec3)      # steady wind, world frame (m/s)
    separation_m: float = 1000.0                  # proximity-alert threshold
    collisions: bool = True
    gravity_falloff: bool = True
    bodies: dict[int, Body] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    _conflicts: set[tuple[int, int]] = field(default_factory=set, repr=False)

    # -- population ------------------------------------------------------

    def spawn(
        self,
        profile: str | Profile,
        position: Vec3,
        velocity: Vec3 = Vec3(),
        name: str = "",
        label: str = "",
        command: Command | None = None,
        fuel_kg: float | None = None,
    ) -> Body:
        """Add an object. Raises WorldFull past 32 live objects."""
        if len(self.bodies) >= MAX_OBJECTS:
            raise WorldFull(f"world already holds {MAX_OBJECTS} objects")
        prof = profiles.get(profile) if isinstance(profile, str) else profile
        body = Body(
            profile=prof,
            position=position,
            velocity=velocity,
            name=name,
            label=label,
            command=command or Command(),
            fuel_kg=-1.0 if fuel_kg is None else fuel_kg,
        )
        self.bodies[body.id] = body
        self._emit("spawn", id=body.id, name=body.name, profile=prof.name, object_kind=prof.kind)
        return body

    def remove(self, body_id: int) -> bool:
        gone = self.bodies.pop(body_id, None)
        if gone is not None:
            self._conflicts = {p for p in self._conflicts if body_id not in p}
            self._emit("removed", id=body_id, name=gone.name)
        return gone is not None

    def get(self, body_id: int) -> Body | None:
        return self.bodies.get(body_id)

    def set_command(self, body_id: int, command: Command) -> Body:
        body = self.bodies.get(body_id)
        if body is None:
            raise KeyError(f"no object {body_id}")
        body.command = command
        self._emit("command", id=body_id, mode=command.mode.value)
        return body

    def release_sar_drone(
        self, carrier_id: int, target_id: int, profile: str | Profile = "sar_drone",
        name: str = "",
    ) -> Body:
        """Release a search-and-rescue drone from a carrier aircraft.

        The drone inherits the carrier's position and velocity, then flies
        Mode.PURSUE toward the designated aircraft to attach a locator
        beacon -- see World._check_captures.
        """
        carrier = self.bodies.get(carrier_id)
        target = self.bodies.get(target_id)
        if carrier is None or not carrier.active:
            raise ValueError(f"carrier {carrier_id} not available")
        if target is None or not target.active:
            raise ValueError(f"target {target_id} not available")
        prof = profiles.get(profile) if isinstance(profile, str) else profile
        if prof.capture_radius_m <= 0.0:
            raise ValueError(f"profile {prof.name!r} cannot attach a beacon "
                             "(capture_radius_m is 0)")

        # Release behind and below the carrier -- clear of its airframe and
        # wake, so the two do not immediately register as a collision.
        clearance = carrier.profile.radius_m + prof.radius_m + 20.0
        offset = (carrier.velocity.unit() * -clearance if carrier.velocity.mag > 1e-6
                 else Vec3()) + UP * (-0.25 * clearance)
        drone = self.spawn(
            prof, carrier.position + offset, carrier.velocity, name=name,
            command=Command(mode=Mode.PURSUE, target_id=target_id),
        )
        self._emit("sar_drone_released", id=drone.id, carrier_id=carrier_id,
                   target_id=target_id)
        return drone

    @property
    def active(self) -> list[Body]:
        return [b for b in self.bodies.values() if b.active]

    @property
    def flying(self) -> list[Body]:
        """Active objects still under their own control -- excludes anything
        that has attached to and is now riding with another object."""
        return [b for b in self.bodies.values() if b.active and b.attached_to is None]

    @property
    def count(self) -> int:
        return len(self.bodies)

    @property
    def capacity(self) -> int:
        return MAX_OBJECTS

    def clear(self) -> None:
        self.bodies.clear()
        self.events.clear()
        self._conflicts.clear()
        self.time_s = 0.0

    def _emit(self, kind: str, **data) -> None:
        self.events.append(Event(self.time_s, kind, data))
        if len(self.events) > MAX_EVENTS:
            del self.events[: len(self.events) - MAX_EVENTS]

    # -- forces ----------------------------------------------------------

    def _thrust_magnitude(self, body: Body, throttle: float) -> float:
        p = body.profile
        if not p.powered or throttle <= 0.0:
            return 0.0
        if p.fuel_kg > 0.0 and body.fuel_kg <= 0.0:
            return 0.0          # tanks dry
        return p.thrust_n * throttle

    def acceleration(self, body: Body, pos: Vec3, vel: Vec3,
                     demand: control.Demand) -> Vec3:
        """Net acceleration (m/s^2) from thrust, drag, lift, buoyancy, weight."""
        p = body.profile
        density, _, sound = physics.atmosphere(pos.z)
        airspeed_vec = vel - self.wind
        airspeed = airspeed_vec.mag
        mach = physics.mach_number(airspeed, sound)
        mass = max(1e-6, body.mass_kg)

        # Wings turn the steering demand into lift, inside the Cl and g limits.
        lift = Vec3()
        cl = 0.0
        if p.cl_max > 0.0 and airspeed > 1e-3:
            want = demand.accel.clamp(p.max_g * physics.G0).perp_to(airspeed_vec)
            if want.mag > 1e-9:
                cl = physics.lift_coefficient_for(
                    want.mag, mass, density, airspeed, p.ref_area_m2, p.cl_max)
                lift = physics.lift_force(airspeed_vec, density, cl, p.ref_area_m2,
                                          up_hint=want)

        # Gas envelopes swell as the air thins: more displacement, more drag area.
        volume = physics.balloon_volume(body.volume_m3, p.expansion_ratio, density)
        area = p.ref_area_m2
        if volume > 0.0 and p.volume_m3 > 0.0:
            area *= (volume / p.volume_m3) ** (2.0 / 3.0)

        cdi = physics.induced_drag_coefficient(cl, p.aspect_ratio)
        cd = physics.drag_coefficient(body.drag_cd0, mach, cdi)
        drag = physics.drag_force(airspeed_vec, density, cd, area)

        if p.thrust_mode == "vector":
            thrust_dir = demand.thrust_dir
        elif airspeed > 1e-3:
            thrust_dir = airspeed_vec.unit()
        else:
            thrust_dir = demand.thrust_dir
        thrust = physics.thrust_force(thrust_dir, self._thrust_magnitude(body, demand.throttle))

        buoyancy = physics.buoyancy_force(volume, density, pos.z)
        weight = physics.weight_force(mass, pos, self.gravity_falloff)

        return (thrust + drag + lift + buoyancy + weight) / mass

    # -- integration -----------------------------------------------------

    def _integrate(self, body: Body, demand: control.Demand, dt: float) -> tuple[Vec3, Vec3]:
        """Classical RK4 on (position, velocity) with the tick's demand frozen."""
        p0, v0 = body.position, body.velocity
        a1 = self.acceleration(body, p0, v0, demand)

        p2, v2 = p0 + v0 * (dt / 2), v0 + a1 * (dt / 2)
        a2 = self.acceleration(body, p2, v2, demand)

        p3, v3 = p0 + v2 * (dt / 2), v0 + a2 * (dt / 2)
        a3 = self.acceleration(body, p3, v3, demand)

        p4, v4 = p0 + v3 * dt, v0 + a3 * dt
        a4 = self.acceleration(body, p4, v4, demand)

        pos = p0 + (v0 + v2 * 2 + v3 * 2 + v4) * (dt / 6)
        vel = v0 + (a1 + a2 * 2 + a3 * 2 + a4) * (dt / 6)
        return pos, vel

    # -- per-tick bookkeeping --------------------------------------------

    def _burn_fuel(self, body: Body, throttle: float, dt: float) -> None:
        p = body.profile
        if p.burn_rate_kgs <= 0.0 or throttle <= 0.0 or body.fuel_kg <= 0.0:
            return
        body.fuel_kg = max(0.0, body.fuel_kg - p.burn_rate_kgs * throttle * dt)
        if body.fuel_kg == 0.0:
            self._emit("fuel_exhausted", id=body.id, name=body.name)

    def _post_update(self, body: Body, prev_vel: Vec3, dt: float) -> None:
        p = body.profile
        _, _, sound = physics.atmosphere(body.altitude_m)

        # Soft envelope cap: bleed anything past the profile's limit speed.
        if body.speed_mps > p.max_speed_mps > 0.0:
            body.velocity = body.velocity.unit() * p.max_speed_mps

        body.mach = physics.mach_number((body.velocity - self.wind).mag, sound)
        if dt > 0.0 and body.speed_mps > 1e-6:
            dv = (body.velocity - prev_vel) / dt
            body.load_factor_g = dv.perp_to(body.velocity).mag / physics.G0

        if p.burst_altitude_m > 0.0 and not body.burst and body.altitude_m >= p.burst_altitude_m:
            body.burst = True
            self._emit("balloon_burst", id=body.id, name=body.name,
                       altitude_m=round(body.altitude_m, 1))

    def _check_limits(self, body: Body) -> None:
        if body.altitude_m <= 0.0:
            descent = -body.velocity.z
            gentle = descent <= SURVIVABLE_TOUCHDOWN_MPS and body.kind != "projectile"
            body.status = Status.LANDED if gentle else Status.DESTROYED
            body.position = Vec3(body.position.x, body.position.y, 0.0)
            body.velocity = Vec3()
            self._emit("ground_contact", id=body.id, name=body.name,
                       descent_rate_mps=round(descent, 2), outcome=body.status.value)
            return
        if not self.bounds.contains(body.position):
            body.status = Status.EXITED
            self._emit("exited_volume", id=body.id, name=body.name,
                       position=body.position.to_dict())

    # -- separation and collisions ---------------------------------------

    @staticmethod
    def _closest_approach(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> float:
        """Least distance between two objects over one tick's swept segments."""
        r0 = a0 - b0
        dr = (a1 - b1) - r0
        denom = dr.mag2
        if denom < 1e-12:
            return r0.mag
        s = max(0.0, min(1.0, -r0.dot(dr) / denom))
        return (r0 + dr * s).mag

    def _check_pairs(self, prev: dict[int, Vec3]) -> None:
        # A beacon riding attached to its target isn't a separate collision
        # or traffic hazard -- it's part of that object now.
        live = self.flying
        seen: set[tuple[int, int]] = set()
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if not (a.active and b.active):
                    continue
                key = (min(a.id, b.id), max(a.id, b.id))
                miss = self._closest_approach(
                    prev.get(a.id, a.position), a.position,
                    prev.get(b.id, b.position), b.position)

                hit_radius = a.profile.radius_m + b.profile.radius_m
                if self.collisions and miss <= hit_radius:
                    a.status = b.status = Status.DESTROYED
                    self._conflicts.discard(key)
                    self._emit("collision", ids=[a.id, b.id], names=[a.name, b.name],
                               separation_m=round(miss, 2))
                    continue

                if miss <= self.separation_m:
                    seen.add(key)
                    if key not in self._conflicts:
                        self._conflicts.add(key)
                        self._emit("proximity_alert", ids=[a.id, b.id],
                                   names=[a.name, b.name], separation_m=round(miss, 2))
        for key in self._conflicts - seen:
            self._conflicts.discard(key)
            self._emit("proximity_clear", ids=list(key))

    # -- search-and-rescue beacons -----------------------------------------

    def _slave_attached(self, dt: float) -> None:
        """Carry along any drone that attached in an earlier tick.

        It rides at its target's position and velocity -- a beacon fixed to
        the fuselage -- until the target is no longer active, at which point
        it detaches and goes back to flying (ballistic) on its own.
        """
        for body in self.bodies.values():
            if not body.active or body.attached_to is None:
                continue
            target = self.bodies.get(body.attached_to)
            if target is None or not target.active:
                body.attached_to = None
                body.command = Command(mode=Mode.BALLISTIC)
                self._emit("beacon_detached", id=body.id, name=body.name,
                           reason="target_lost")
                continue
            body.position = target.position
            body.velocity = target.velocity
            body.age_s += dt

    def _check_captures(self, prev: dict[int, Vec3]) -> None:
        """Attach any PURSUE drone that has closed to its capture radius."""
        for body in self.flying:
            if body.command.mode is not Mode.PURSUE or body.command.target_id is None:
                continue
            if body.profile.capture_radius_m <= 0.0:
                continue
            target = self.bodies.get(body.command.target_id)
            if target is None or not target.active:
                continue
            miss = self._closest_approach(
                prev.get(body.id, body.position), body.position,
                prev.get(target.id, target.position), target.position)
            if miss <= body.profile.capture_radius_m:
                body.attached_to = target.id
                body.position = target.position
                body.velocity = target.velocity
                self._emit("beacon_attached", id=body.id, name=body.name,
                           target_id=target.id, target_name=target.name,
                           position=target.position.to_dict())

    # -- the step --------------------------------------------------------

    def step(self, dt: float | None = None) -> None:
        """Advance the world one fixed tick."""
        h = self.dt if dt is None else dt
        prev_positions = {b.id: b.position for b in self.bodies.values()}
        flying = self.flying

        demands: dict[int, control.Demand] = {}
        for body in flying:
            target = self.bodies.get(body.command.target_id) if body.command.target_id else None
            demands[body.id] = control.solve(body, target)

        for body in flying:
            demand = demands[body.id]
            prev_vel = body.velocity
            body.position, body.velocity = self._integrate(body, demand, h)
            body.age_s += h
            body.throttle = demand.throttle
            body.thrust_dir = demand.thrust_dir
            body.accel_cmd = demand.accel
            self._burn_fuel(body, demand.throttle, h)
            self._post_update(body, prev_vel, h)

        self._slave_attached(h)
        self._check_captures(prev_positions)
        self._check_pairs(prev_positions)

        for body in self.active:
            self._check_limits(body)

        self.time_s += h

    def run(self, duration_s: float, dt: float | None = None) -> int:
        """Step for `duration_s` of sim time. Returns ticks executed."""
        h = self.dt if dt is None else dt
        ticks = max(0, int(round(duration_s / h)))
        for _ in range(ticks):
            self.step(h)
        return ticks

    def run_until_quiet(self, max_time_s: float = 600.0) -> float:
        """Step until nothing is still flying, or the time cap is hit."""
        end = self.time_s + max_time_s
        while self.time_s < end and self.active:
            self.step()
        return self.time_s

    # -- reporting -------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "time_s": round(self.time_s, 3),
            "dt": self.dt,
            "count": self.count,
            "active": len(self.active),
            "capacity": MAX_OBJECTS,
            "bounds": self.bounds.to_dict(),
            "wind": self.wind.to_dict(),
            "separation_m": self.separation_m,
            "objects": [b.to_dict() for b in self.bodies.values()],
        }

    def drain_events(self) -> list[dict]:
        out = [e.to_dict() for e in self.events]
        self.events.clear()
        return out
