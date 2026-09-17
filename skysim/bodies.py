"""Objects the world tracks, and the command each one is flying."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum

from .profiles import Profile
from .vec import UP, Vec3

MAX_OBJECTS = 32


class Status(str, Enum):
    ACTIVE = "active"
    LANDED = "landed"        # touched down / came to rest on the surface
    DESTROYED = "destroyed"  # collision, or ground impact above survivable speed
    EXITED = "exited"        # left the simulated volume
    SPENT = "spent"          # balloon burst, payload finished, nothing left to do


class Mode(str, Enum):
    BALLISTIC = "ballistic"  # unguided: gravity, drag, buoyancy only
    HOLD = "hold"            # keep the altitude/heading/speed it has now
    HEADING = "heading"      # fly a commanded heading + altitude + speed
    WAYPOINT = "waypoint"    # steer to a point, then hold there
    HOVER = "hover"          # rotorcraft: park over a point
    FOLLOW = "follow"        # trail another object at a standoff distance
    ASCENT = "ascent"        # rocket pitch program


@dataclass(slots=True)
class Command:
    """What the autopilot is being asked to do. All fields optional."""

    mode: Mode = Mode.BALLISTIC
    heading_deg: float | None = None
    altitude_m: float | None = None
    speed_mps: float | None = None
    waypoint: Vec3 | None = None
    target_id: int | None = None
    standoff_m: float = 500.0
    pitch_deg: float = 90.0      # ASCENT: 90 = straight up
    pitch_over_s: float = 0.0    # ASCENT: seconds before pitching to pitch_deg
    throttle: float | None = None  # None = autopilot decides

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "heading_deg": self.heading_deg,
            "altitude_m": self.altitude_m,
            "speed_mps": self.speed_mps,
            "waypoint": self.waypoint.to_dict() if self.waypoint else None,
            "target_id": self.target_id,
            "standoff_m": self.standoff_m,
            "pitch_deg": self.pitch_deg,
            "pitch_over_s": self.pitch_over_s,
            "throttle": self.throttle,
        }


_ids = itertools.count(1)


def reset_ids() -> None:
    """Restart id numbering (tests, fresh processes)."""
    global _ids
    _ids = itertools.count(1)


@dataclass(slots=True)
class Body:
    """A point mass flying under the force model in world.py."""

    profile: Profile
    position: Vec3
    velocity: Vec3
    id: int = field(default_factory=lambda: next(_ids))
    name: str = ""
    label: str = ""                       # free-form tag: callsign, operator, ...
    status: Status = Status.ACTIVE
    command: Command = field(default_factory=Command)
    fuel_kg: float = -1.0                 # <0 means "fill from the profile"
    throttle: float = 0.0                 # 0..1, resolved each tick
    age_s: float = 0.0
    burst: bool = False                   # balloons: envelope has let go
    # Set each tick by the control loop / integrator, reported in snapshots.
    thrust_dir: Vec3 = field(default_factory=lambda: UP)
    accel_cmd: Vec3 = field(default_factory=Vec3)
    mach: float = 0.0
    load_factor_g: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.profile.name}-{self.id}"
        if self.fuel_kg < 0.0:
            self.fuel_kg = self.profile.fuel_kg
        if self.velocity.mag > 1e-6:
            self.thrust_dir = self.velocity.unit()

    # -- derived state ---------------------------------------------------

    @property
    def kind(self) -> str:
        return self.profile.kind

    @property
    def mass_kg(self) -> float:
        return self.profile.mass_kg + max(0.0, self.fuel_kg)

    @property
    def altitude_m(self) -> float:
        return self.position.z

    @property
    def speed_mps(self) -> float:
        return self.velocity.mag

    @property
    def active(self) -> bool:
        return self.status is Status.ACTIVE

    @property
    def volume_m3(self) -> float:
        """Displaced volume; a burst balloon displaces nothing."""
        return 0.0 if self.burst else self.profile.volume_m3

    @property
    def drag_cd0(self) -> float:
        """A burst envelope trails behind as a crude parachute."""
        return self.profile.cd0 * (2.5 if self.burst else 1.0)

    def heading_deg(self) -> float:
        """Compass heading of the ground track: 0 = north, 90 = east."""
        return math.degrees(math.atan2(self.velocity.x, self.velocity.y)) % 360.0

    def flight_path_angle_deg(self) -> float:
        """Climb angle above the horizon, in degrees."""
        horiz = math.hypot(self.velocity.x, self.velocity.y)
        return math.degrees(math.atan2(self.velocity.z, horiz)) if (horiz or self.velocity.z) else 0.0

    def ground_speed_mps(self) -> float:
        return math.hypot(self.velocity.x, self.velocity.y)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "profile": self.profile.name,
            "kind": self.kind,
            "status": self.status.value,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "altitude_m": round(self.altitude_m, 2),
            "speed_mps": round(self.speed_mps, 3),
            "ground_speed_mps": round(self.ground_speed_mps(), 3),
            "vertical_speed_mps": round(self.velocity.z, 3),
            "heading_deg": round(self.heading_deg(), 2),
            "flight_path_angle_deg": round(self.flight_path_angle_deg(), 2),
            "mach": round(self.mach, 4),
            "load_factor_g": round(self.load_factor_g, 3),
            "mass_kg": round(self.mass_kg, 3),
            "fuel_kg": round(max(0.0, self.fuel_kg), 3),
            "throttle": round(self.throttle, 3),
            "age_s": round(self.age_s, 3),
            "burst": self.burst,
            "command": self.command.to_dict(),
        }


def heading_vector(heading_deg: float) -> Vec3:
    """Unit horizontal vector for a compass heading (0 = north, 90 = east)."""
    rad = math.radians(heading_deg)
    return Vec3(math.sin(rad), math.cos(rad), 0.0)
