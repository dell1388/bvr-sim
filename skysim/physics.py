"""Force models: atmosphere, drag, lift, buoyancy, thrust, gravity.

Textbook point-mass aerodynamics over a flat local frame, valid from the
surface up to the top of the modelled airspace (~60 km). Everything here is
a pure function of state, so the integrator can call it as often as it likes.
"""
from __future__ import annotations

import math

from .vec import UP, Vec3

G0 = 9.80665              # m/s^2, standard gravity at the surface
R_EARTH = 6371000.0       # m, mean radius (only used for the g(h) falloff)
R_AIR = 287.05287         # J/(kg*K), specific gas constant for dry air
GAMMA = 1.4               # ratio of specific heats

# Top of the modelled airspace. The atmosphere table runs higher, but the
# world treats objects above this as having left the simulated volume.
CEILING_M = 60000.0

# International Standard Atmosphere: (base alt m, base temp K, base pressure Pa, lapse K/m)
_ISA_LAYERS = (
    (0.0, 288.15, 101325.0, -0.0065),
    (11000.0, 216.65, 22632.06, 0.0),
    (20000.0, 216.65, 5474.889, 0.001),
    (32000.0, 228.65, 868.0187, 0.0028),
    (47000.0, 270.65, 110.9063, 0.0),
    (51000.0, 270.65, 66.93887, -0.0028),
    (71000.0, 214.65, 3.956420, -0.002),
)
_ISA_TOP_M = 84852.0
_ISA_TOP_T = 186.946
_ISA_TOP_RHO = 6.958e-6

# Above the ISA table, an approximate piecewise-exponential fit to the upper
# atmosphere: (ceiling m, scale height m). Good enough for orbital decay to
# show up at low altitude and to vanish higher up.
_UPPER_LAYERS = ((150000.0, 7990.0), (500000.0, 42200.0), (1.0e9, 60000.0))


def atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """Return (density kg/m^3, pressure Pa, speed of sound m/s) at altitude."""
    h = altitude_m
    if h <= 0.0:
        h = 0.0

    if h <= _ISA_TOP_M:
        base_h, base_t, base_p, lapse = _ISA_LAYERS[0]
        for layer in _ISA_LAYERS:
            if h >= layer[0]:
                base_h, base_t, base_p, lapse = layer
            else:
                break
        dh = h - base_h
        if lapse == 0.0:
            temp = base_t
            pressure = base_p * math.exp(-G0 * dh / (R_AIR * base_t))
        else:
            temp = base_t + lapse * dh
            pressure = base_p * (temp / base_t) ** (-G0 / (lapse * R_AIR))
        density = pressure / (R_AIR * temp)
        return density, pressure, math.sqrt(GAMMA * R_AIR * temp)

    # Thin upper atmosphere: isothermal-ish exponential segments.
    density = _ISA_TOP_RHO
    h_base = _ISA_TOP_M
    for ceiling, scale in _UPPER_LAYERS:
        top = min(h, ceiling)
        density *= math.exp(-(top - h_base) / scale)
        h_base = top
        if h <= ceiling:
            break
    temp = _ISA_TOP_T
    return density, density * R_AIR * temp, math.sqrt(GAMMA * R_AIR * temp)


def gravity_magnitude(altitude_m: float) -> float:
    """g at altitude (m/s^2): g0 scaled by inverse square of radius.

    At 60 km this is ~0.98 g -- small, but free. The frame stays flat: gravity
    always points straight down.
    """
    ratio = R_EARTH / (R_EARTH + max(0.0, altitude_m))
    return G0 * ratio * ratio


def gravity_accel(position: Vec3, altitude_falloff: bool = True) -> Vec3:
    """Gravitational acceleration, straight down in the world frame."""
    g = gravity_magnitude(position.z) if altitude_falloff else G0
    return Vec3(0.0, 0.0, -g)


def dynamic_pressure(density: float, speed: float) -> float:
    """q = 1/2 * rho * V^2 (Pa)."""
    return 0.5 * density * speed * speed


def mach_number(speed: float, sound_speed: float) -> float:
    return 0.0 if sound_speed <= 0.0 else speed / sound_speed


def drag_coefficient(cd0: float, mach: float, induced: float = 0.0) -> float:
    """Zero-lift Cd with a smooth transonic rise, plus an induced term.

    The Mach factor is a shape, not wind-tunnel data: flat subsonic, a peak
    near Mach 1, settling to a higher-than-subsonic supersonic value.
    """
    if mach < 0.8:
        factor = 1.0
    elif mach < 1.2:
        factor = 1.0 + 2.5 * (mach - 0.8) / 0.4
    else:
        factor = max(1.4, 3.5 / math.sqrt(mach / 1.2))
    return cd0 * factor + induced


def induced_drag_coefficient(cl: float, aspect_ratio: float, efficiency: float = 0.8) -> float:
    """Cdi = Cl^2 / (pi * AR * e)."""
    if aspect_ratio <= 0.0:
        return 0.0
    return (cl * cl) / (math.pi * aspect_ratio * efficiency)


def drag_force(velocity: Vec3, density: float, cd: float, area: float) -> Vec3:
    """D = -q * Cd * S * v_hat (N), always opposing motion."""
    speed = velocity.mag
    if speed < 1e-6 or density <= 0.0:
        return Vec3()
    return velocity.unit() * (-dynamic_pressure(density, speed) * cd * area)


def lift_force(velocity: Vec3, density: float, cl: float, area: float,
               up_hint: Vec3 = UP) -> Vec3:
    """L = q * Cl * S (N), perpendicular to velocity, toward `up_hint`."""
    speed = velocity.mag
    if speed < 1e-6 or abs(cl) < 1e-12:
        return Vec3()
    lift_dir = up_hint.perp_to(velocity)
    if lift_dir.mag2 < 1e-12:
        return Vec3()
    return lift_dir.unit() * (dynamic_pressure(density, speed) * cl * area)


def lift_coefficient_for(accel_mps2: float, mass_kg: float, density: float,
                         speed: float, area: float, cl_max: float) -> float:
    """Cl needed to pull `accel_mps2` of turn, saturated at cl_max."""
    q = dynamic_pressure(density, speed)
    if q <= 1e-9 or area <= 0.0:
        return 0.0
    return min(abs(accel_mps2) * mass_kg / (q * area), cl_max)


def buoyancy_force(volume_m3: float, density: float, altitude_m: float = 0.0) -> Vec3:
    """Archimedes: B = rho_air * V * g, straight up. Zero for non-lifting bodies."""
    if volume_m3 <= 0.0 or density <= 0.0:
        return Vec3()
    return UP * (density * volume_m3 * gravity_magnitude(altitude_m))


def balloon_volume(base_volume_m3: float, expansion_ratio: float, density: float,
                   sea_level_density: float = 1.225) -> float:
    """Displaced volume of a gas envelope at altitude.

    A fixed mass of lifting gas expands as the air thins (V ~ 1/rho), until
    the envelope reaches its stretch limit and volume stops growing --
    which is why a free balloon climbs at a near-constant rate, then floats.
    """
    if base_volume_m3 <= 0.0:
        return 0.0
    if density <= 1e-12:
        return base_volume_m3 * max(1.0, expansion_ratio)
    grown = base_volume_m3 * (sea_level_density / density)
    return min(grown, base_volume_m3 * max(1.0, expansion_ratio))


def thrust_force(direction: Vec3, magnitude: float) -> Vec3:
    """Thrust along a unit pointing direction (N)."""
    if magnitude <= 0.0:
        return Vec3()
    return direction.unit() * magnitude


def weight_force(mass_kg: float, position: Vec3, altitude_falloff: bool = True) -> Vec3:
    return gravity_accel(position, altitude_falloff) * mass_kg


def terminal_velocity(mass_kg: float, cd: float, area: float, altitude_m: float = 0.0) -> float:
    """Steady-state fall speed where drag balances weight (m/s)."""
    density, _, _ = atmosphere(altitude_m)
    if density <= 0.0 or cd <= 0.0 or area <= 0.0:
        return float("inf")
    return math.sqrt(2.0 * mass_kg * G0 / (density * cd * area))
