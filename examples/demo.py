"""A busy block of airspace: run with `python examples/demo.py`.

Two airliners on crossing tracks, a light aircraft flying a waypoint, a camera
drone trailing it, a weather balloon on its way up, a sounding rocket climbing
out through the lot, and a search-and-rescue drone released to tag the light
aircraft with a locator beacon.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skysim import Command, Mode, Vec3, World, heading_vector

TRACK = "{:<20} {:>9} {:>8} {:>7} {:>6} {:>8}  {}"


def build() -> tuple[World, object]:
    world = World(dt=0.05, separation_m=3000.0, wind=Vec3(12.0, 4.0, 0.0))

    world.spawn("airliner", Vec3(-60000, 0, 10500), heading_vector(90) * 240,
                name="cruise-east", label="FL345",
                command=Command(mode=Mode.HOLD, altitude_m=10500, speed_mps=240))
    world.spawn("airliner", Vec3(0, -60000, 10200), heading_vector(0) * 235,
                name="cruise-north", label="FL335",
                command=Command(mode=Mode.HOLD, altitude_m=10200, speed_mps=235))

    survey = world.spawn("light_aircraft", Vec3(-8000, -8000, 1200), heading_vector(45) * 60,
                         name="survey-lead",
                         command=Command(mode=Mode.WAYPOINT, waypoint=Vec3(9000, 9000, 1600),
                                         speed_mps=65))
    world.spawn("survey_drone", Vec3(-8600, -8600, 1100), heading_vector(45) * 18,
                name="camera-drone",
                command=Command(mode=Mode.HOVER, waypoint=Vec3(-8000, -8000, 0),
                                altitude_m=400))

    carrier = world.spawn("airliner", Vec3(-40000, -20000, 9500), heading_vector(60) * 235,
                          name="sar-mothership",
                          command=Command(mode=Mode.HOLD, altitude_m=9500, speed_mps=235))
    world.release_sar_drone(carrier.id, survey.id, name="beacon-1")

    world.spawn("weather_balloon", Vec3(4000, -2000, 200), Vec3(0, 0, 2), name="sonde-lift")
    world.spawn("sounding_rocket", Vec3(25000, 25000, 10), Vec3(0, 0, 1), name="research-shot",
                command=Command(mode=Mode.ASCENT, pitch_deg=85, heading_deg=120,
                                pitch_over_s=12))
    world.spawn("cargo_capsule", Vec3(-20000, 15000, 8000), heading_vector(180) * 40,
                name="airdrop")
    return world, survey


def main() -> None:
    world, survey = build()
    print(f"objects: {world.count}/{world.capacity}, dt={world.dt}s, "
          f"wind={world.wind.as_tuple()} m/s\n")

    for mark in (30.0, 120.0, 300.0, 900.0):
        while world.time_s < mark:
            world.step()
            for event in world.drain_events():
                if event["kind"] != "spawn":
                    print(f"  t={event['t']:>7.1f}s  {event['kind']}: "
                          + ", ".join(f"{k}={v}" for k, v in event.items()
                                      if k not in ("t", "kind")))
        print(f"\n--- t = {world.time_s:.0f} s ---")
        print(TRACK.format("object", "alt (m)", "spd", "hdg", "mach", "fuel", "status"))
        for body in world.bodies.values():
            print(TRACK.format(body.name, f"{body.altitude_m:.0f}", f"{body.speed_mps:.0f}",
                               f"{body.heading_deg():.0f}", f"{body.mach:.2f}",
                               f"{body.fuel_kg:.0f}", body.status.value))
        print()

    print(f"survey aircraft finished at {survey.position.as_tuple()}")


if __name__ == "__main__":
    main()
