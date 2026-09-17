"""skysim -- a small point-mass airspace simulator.

One flat block of airspace, surface to ~60 km, up to 32 objects at a time.
Aircraft, rotorcraft, sounding rockets, balloons and dropped payloads all run
through the same force model (thrust, drag, lift, buoyancy, weight) and the
same fixed-step RK4 integrator.

    from skysim import World, Command, Mode, Vec3

    world = World(dt=0.02)
    plane = world.spawn("airliner", Vec3(0, 0, 10000), Vec3(230, 0, 0),
                        command=Command(mode=Mode.HOLD, altitude_m=10000))
    world.run(60.0)
    print(plane.altitude_m, plane.speed_mps)

The HTTP API lives in skysim.api (`python -m skysim`).
"""
from .bodies import MAX_OBJECTS, Body, Command, Mode, Status, heading_vector
from .profiles import Profile, register
from .vec import Vec3
from .world import Bounds, Event, World, WorldFull

__version__ = "1.0.0"

__all__ = [
    "Body", "Bounds", "Command", "Event", "MAX_OBJECTS", "Mode", "Profile",
    "Status", "Vec3", "World", "WorldFull", "heading_vector", "register",
]
