"""Vehicle profiles: the fixed parameters of each kind of object.

Numbers are round, generic placeholders chosen so the sim behaves plausibly
(airliners cruise, rockets climb, balloons rise then burst, payloads fall).
They are not taken from any specific make or model. Add your own with
`register()` or by passing a Profile straight to `World.spawn`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# How thrust is pointed.
#   "velocity" -- along the flight path (jets, rockets past the tower)
#   "vector"   -- along a commanded direction (multirotors, vertical lift-off)
#   "none"     -- unpowered (gliders, balloons, dropped payloads)
THRUST_MODES = ("velocity", "vector", "none")

KINDS = ("aircraft", "rotorcraft", "rocket", "balloon", "projectile")


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    kind: str                     # one of KINDS
    mass_kg: float                # dry mass (without fuel/propellant)
    ref_area_m2: float            # aerodynamic reference area
    cd0: float = 0.3              # zero-lift drag coefficient
    cl_max: float = 0.0           # max lift coefficient (0 = no wings)
    aspect_ratio: float = 0.0     # wing aspect ratio (0 = skip induced drag)
    max_g: float = 3.0            # structural / control acceleration limit, in g
    # Propulsion
    thrust_n: float = 0.0         # max thrust at full throttle
    fuel_kg: float = 0.0          # usable fuel / propellant loaded
    burn_rate_kgs: float = 0.0    # consumption at full throttle
    thrust_mode: str = "none"
    # Envelope
    max_speed_mps: float = 1.0e9  # soft envelope cap
    service_ceiling_m: float = 1.0e9
    # Kind-specific
    volume_m3: float = 0.0        # displaced volume at sea level (balloons)
    expansion_ratio: float = 1.0  # how far the envelope can stretch (1 = rigid)
    capture_radius_m: float = 0.0 # PURSUE-mode rendezvous range (0 = not equipped)
    burst_altitude_m: float = 0.0 # balloons: envelope bursts above this
    radius_m: float = 5.0         # collision radius

    @property
    def wet_mass_kg(self) -> float:
        return self.mass_kg + self.fuel_kg

    @property
    def powered(self) -> bool:
        return self.thrust_n > 0.0 and self.thrust_mode != "none"

    def variant(self, name: str, **changes) -> "Profile":
        """A copy with overrides, e.g. AIRLINER.variant('heavy', fuel_kg=30000)."""
        return replace(self, name=name, **changes)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__slots__}


AIRLINER = Profile(
    name="airliner", kind="aircraft",
    mass_kg=42000.0, ref_area_m2=122.0, cd0=0.021, cl_max=1.4, aspect_ratio=9.0,
    max_g=2.5, thrust_n=220000.0, fuel_kg=16000.0, burn_rate_kgs=1.2,
    thrust_mode="velocity", max_speed_mps=265.0, service_ceiling_m=12500.0, radius_m=30.0,
)

LIGHT_AIRCRAFT = Profile(
    name="light_aircraft", kind="aircraft",
    mass_kg=780.0, ref_area_m2=16.2, cd0=0.032, cl_max=1.6, aspect_ratio=7.4,
    max_g=3.8, thrust_n=2400.0, fuel_kg=110.0, burn_rate_kgs=0.011,
    thrust_mode="velocity", max_speed_mps=78.0, service_ceiling_m=4200.0, radius_m=6.0,
)

GLIDER = Profile(
    name="glider", kind="aircraft",
    mass_kg=350.0, ref_area_m2=11.0, cd0=0.012, cl_max=1.5, aspect_ratio=22.0,
    max_g=4.0, thrust_mode="none", max_speed_mps=70.0, service_ceiling_m=8000.0, radius_m=8.0,
)

SURVEY_DRONE = Profile(
    name="survey_drone", kind="rotorcraft",
    mass_kg=6.5, ref_area_m2=0.28, cd0=0.9, max_g=2.0,
    thrust_n=180.0, fuel_kg=0.0, burn_rate_kgs=0.0,
    thrust_mode="vector", max_speed_mps=22.0, service_ceiling_m=4000.0, radius_m=1.0,
)

SOUNDING_ROCKET = Profile(
    name="sounding_rocket", kind="rocket",
    mass_kg=260.0, ref_area_m2=0.13, cd0=0.32, max_g=12.0,
    thrust_n=42000.0, fuel_kg=760.0, burn_rate_kgs=26.0,
    thrust_mode="vector", max_speed_mps=1800.0, radius_m=3.0,
)

WEATHER_BALLOON = Profile(
    name="weather_balloon", kind="balloon",
    mass_kg=3.2, ref_area_m2=7.5, cd0=0.55, max_g=0.5,
    thrust_mode="none", max_speed_mps=60.0,
    volume_m3=14.0, expansion_ratio=80.0, burst_altitude_m=30000.0, radius_m=2.0,
)

DROPSONDE = Profile(
    name="dropsonde", kind="projectile",
    mass_kg=0.4, ref_area_m2=0.16, cd0=1.2, max_g=0.0,
    thrust_mode="none", radius_m=0.5,
)

SAR_DRONE = Profile(
    name="sar_drone", kind="aircraft",
    mass_kg=180.0, ref_area_m2=3.2, cd0=0.028, cl_max=1.3, aspect_ratio=8.0,
    max_g=6.0, thrust_n=9000.0, fuel_kg=120.0, burn_rate_kgs=0.4,
    thrust_mode="velocity", max_speed_mps=260.0, service_ceiling_m=15000.0,
    radius_m=1.5, capture_radius_m=25.0,
)

CARGO_CAPSULE = Profile(
    name="cargo_capsule", kind="projectile",
    mass_kg=120.0, ref_area_m2=1.8, cd0=0.9, max_g=0.0,
    thrust_mode="none", radius_m=1.5,
)

HIGH_ALT_PLATFORM = Profile(
    name="high_alt_platform", kind="balloon",
    mass_kg=95.0, ref_area_m2=40.0, cd0=0.6, max_g=0.3,
    thrust_mode="none", max_speed_mps=40.0,
    volume_m3=2600.0, expansion_ratio=1.0, burst_altitude_m=0.0, radius_m=15.0,
)

REGISTRY: dict[str, Profile] = {}


def register(profile: Profile) -> Profile:
    """Add (or replace) a profile in the lookup table."""
    if profile.kind not in KINDS:
        raise ValueError(f"unknown kind {profile.kind!r}; have {KINDS}")
    if profile.thrust_mode not in THRUST_MODES:
        raise ValueError(f"unknown thrust_mode {profile.thrust_mode!r}; have {THRUST_MODES}")
    REGISTRY[profile.name] = profile
    return profile


for _p in (AIRLINER, LIGHT_AIRCRAFT, GLIDER, SURVEY_DRONE, SOUNDING_ROCKET,
           WEATHER_BALLOON, HIGH_ALT_PLATFORM, DROPSONDE, CARGO_CAPSULE, SAR_DRONE):
    register(_p)


def get(name: str) -> Profile:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; have {sorted(REGISTRY)}") from None


def names() -> list[str]:
    return sorted(REGISTRY)
