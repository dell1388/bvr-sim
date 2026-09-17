"""HTTP API over the engine.

Run it with:  uvicorn skysim.api:app --reload   (or: python -m skysim)

Simulations live in memory, keyed by id. Each one is stepped only when a
client asks, so the API is deterministic and replayable: same inputs, same
trajectories.
"""
from __future__ import annotations

import math
import uuid
from threading import RLock
from typing import Literal

from fastapi import Body as BodyParam, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import profiles
from .bodies import MAX_OBJECTS, Command, Mode
from .vec import Vec3
from .world import Bounds, World, WorldFull

app = FastAPI(
    title="skysim",
    version="1.0.0",
    summary="Point-mass airspace simulator: flight, ballistics and balloons, "
            "surface to 60 km, up to 32 objects per world.",
)

_sims: dict[str, World] = {}
_lock = RLock()


# --- request/response models -------------------------------------------

class Vec3Model(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_vec(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)


class BoundsModel(BaseModel):
    x_min: float = -200000.0
    x_max: float = 200000.0
    y_min: float = -200000.0
    y_max: float = 200000.0
    ceiling_m: float = 60000.0


class SimCreate(BaseModel):
    dt: float = Field(0.02, gt=0.0, le=1.0, description="fixed timestep, seconds")
    bounds: BoundsModel | None = None
    wind: Vec3Model | None = Field(None, description="steady wind, m/s, world frame")
    separation_m: float = Field(1000.0, ge=0.0, description="proximity-alert threshold")
    collisions: bool = True
    gravity_falloff: bool = Field(True, description="scale g with altitude")


class SimSettings(BaseModel):
    dt: float | None = Field(None, gt=0.0, le=1.0)
    wind: Vec3Model | None = None
    separation_m: float | None = Field(None, ge=0.0)
    collisions: bool | None = None


class LaunchState(BaseModel):
    """Velocity given the way a flight plan gives it, instead of as a vector."""

    heading_deg: float = 0.0
    speed_mps: float = 0.0
    climb_mps: float = 0.0

    def to_vec(self) -> Vec3:
        rad = math.radians(self.heading_deg)
        return Vec3(math.sin(rad) * self.speed_mps, math.cos(rad) * self.speed_mps,
                    self.climb_mps)


class CommandModel(BaseModel):
    mode: Literal["ballistic", "hold", "heading", "waypoint", "hover", "follow",
                 "ascent", "pursue"] = "ballistic"
    heading_deg: float | None = None
    altitude_m: float | None = None
    speed_mps: float | None = None
    waypoint: Vec3Model | None = None
    target_id: int | None = Field(None, description="object to follow, for mode=follow")
    standoff_m: float = 500.0
    pitch_deg: float = 90.0
    pitch_over_s: float = 0.0
    throttle: float | None = Field(None, ge=0.0, le=1.0)

    def to_command(self) -> Command:
        return Command(
            mode=Mode(self.mode),
            heading_deg=self.heading_deg,
            altitude_m=self.altitude_m,
            speed_mps=self.speed_mps,
            waypoint=self.waypoint.to_vec() if self.waypoint else None,
            target_id=self.target_id,
            standoff_m=self.standoff_m,
            pitch_deg=self.pitch_deg,
            pitch_over_s=self.pitch_over_s,
            throttle=self.throttle,
        )


class SpawnRequest(BaseModel):
    profile: str = Field(..., description="profile name, see GET /profiles")
    position: Vec3Model = Vec3Model()
    velocity: Vec3Model | None = None
    launch: LaunchState | None = Field(None, description="alternative to velocity")
    name: str = ""
    label: str = ""
    fuel_kg: float | None = None
    command: CommandModel | None = None


class ReleaseRequest(BaseModel):
    target_id: int = Field(..., description="object to designate for beacon attachment")
    profile: str = Field("sar_drone", description="must have capture_radius_m > 0")
    name: str = ""


class StepRequest(BaseModel):
    seconds: float | None = Field(None, gt=0.0, le=36000.0)
    ticks: int | None = Field(None, gt=0, le=2000000)
    dt: float | None = Field(None, gt=0.0, le=1.0)


# --- helpers -----------------------------------------------------------

def _world(sim_id: str) -> World:
    world = _sims.get(sim_id)
    if world is None:
        raise HTTPException(404, f"no simulation {sim_id!r}")
    return world


def _object(world: World, object_id: int):
    body = world.get(object_id)
    if body is None:
        raise HTTPException(404, f"no object {object_id}")
    return body


def _sim_info(sim_id: str, world: World) -> dict:
    return {"sim_id": sim_id, "time_s": round(world.time_s, 3), "dt": world.dt,
            "count": world.count, "active": len(world.active), "capacity": MAX_OBJECTS}


# --- routes ------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "sims": len(_sims), "capacity_per_sim": MAX_OBJECTS}


@app.get("/profiles", tags=["meta"])
def list_profiles() -> dict:
    return {"profiles": [profiles.get(n).to_dict() for n in profiles.names()]}


@app.post("/sims", status_code=201, tags=["sims"])
def create_sim(req: SimCreate | None = None) -> dict:
    req = req or SimCreate()
    world = World(
        dt=req.dt,
        bounds=Bounds(**req.bounds.model_dump()) if req.bounds else Bounds(),
        wind=req.wind.to_vec() if req.wind else Vec3(),
        separation_m=req.separation_m,
        collisions=req.collisions,
        gravity_falloff=req.gravity_falloff,
    )
    sim_id = uuid.uuid4().hex[:12]
    with _lock:
        _sims[sim_id] = world
    return _sim_info(sim_id, world)


@app.get("/sims", tags=["sims"])
def list_sims() -> dict:
    with _lock:
        return {"sims": [_sim_info(sid, w) for sid, w in _sims.items()]}


@app.get("/sims/{sim_id}", tags=["sims"])
def get_sim(sim_id: str) -> dict:
    return {"sim_id": sim_id, **_world(sim_id).snapshot()}


@app.patch("/sims/{sim_id}", tags=["sims"])
def update_sim(sim_id: str, settings: SimSettings) -> dict:
    world = _world(sim_id)
    with _lock:
        if settings.dt is not None:
            world.dt = settings.dt
        if settings.wind is not None:
            world.wind = settings.wind.to_vec()
        if settings.separation_m is not None:
            world.separation_m = settings.separation_m
        if settings.collisions is not None:
            world.collisions = settings.collisions
    return {"sim_id": sim_id, **world.snapshot()}


@app.delete("/sims/{sim_id}", tags=["sims"])
def delete_sim(sim_id: str) -> dict:
    with _lock:
        if _sims.pop(sim_id, None) is None:
            raise HTTPException(404, f"no simulation {sim_id!r}")
    return {"deleted": sim_id}


@app.post("/sims/{sim_id}/reset", tags=["sims"])
def reset_sim(sim_id: str) -> dict:
    world = _world(sim_id)
    with _lock:
        world.clear()
    return _sim_info(sim_id, world)


@app.post("/sims/{sim_id}/objects", status_code=201, tags=["objects"])
def spawn_object(sim_id: str, req: SpawnRequest) -> dict:
    world = _world(sim_id)
    try:
        profile = profiles.get(req.profile)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from None
    velocity = req.velocity.to_vec() if req.velocity else (
        req.launch.to_vec() if req.launch else Vec3())
    with _lock:
        try:
            body = world.spawn(
                profile, req.position.to_vec(), velocity,
                name=req.name, label=req.label,
                command=req.command.to_command() if req.command else None,
                fuel_kg=req.fuel_kg,
            )
        except WorldFull as exc:
            raise HTTPException(409, str(exc)) from None
    return body.to_dict()


@app.get("/sims/{sim_id}/objects", tags=["objects"])
def list_objects(sim_id: str) -> dict:
    world = _world(sim_id)
    return {"time_s": round(world.time_s, 3),
            "objects": [b.to_dict() for b in world.bodies.values()]}


@app.get("/sims/{sim_id}/objects/{object_id}", tags=["objects"])
def get_object(sim_id: str, object_id: int) -> dict:
    return _object(_world(sim_id), object_id).to_dict()


@app.delete("/sims/{sim_id}/objects/{object_id}", tags=["objects"])
def delete_object(sim_id: str, object_id: int) -> dict:
    world = _world(sim_id)
    with _lock:
        if not world.remove(object_id):
            raise HTTPException(404, f"no object {object_id}")
    return {"deleted": object_id}


@app.put("/sims/{sim_id}/objects/{object_id}/command", tags=["objects"])
def set_command(sim_id: str, object_id: int, command: CommandModel) -> dict:
    world = _world(sim_id)
    _object(world, object_id)
    if command.mode == "follow" and command.target_id is not None:
        _object(world, command.target_id)
    with _lock:
        body = world.set_command(object_id, command.to_command())
    return body.to_dict()


@app.post("/sims/{sim_id}/objects/{object_id}/release", status_code=201, tags=["objects"])
def release_sar_drone(sim_id: str, object_id: int, req: ReleaseRequest) -> dict:
    """Release a search-and-rescue drone from a carrier aircraft.

    The drone inherits the carrier's position and velocity and flies
    Mode.PURSUE toward `target_id`; once within its capture radius it
    attaches a locator beacon and rides along (see GET .../objects for
    `attached_to`).
    """
    world = _world(sim_id)
    _object(world, object_id)
    _object(world, req.target_id)
    with _lock:
        try:
            drone = world.release_sar_drone(object_id, req.target_id, req.profile, req.name)
        except WorldFull as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    return drone.to_dict()


@app.post("/sims/{sim_id}/step", tags=["run"])
def step_sim(sim_id: str, req: StepRequest | None = BodyParam(None)) -> dict:
    """Advance the world by `seconds`, or by `ticks` steps (default: one tick)."""
    world = _world(sim_id)
    req = req or StepRequest()
    dt = req.dt or world.dt
    with _lock:
        if req.seconds is not None:
            ticks = world.run(req.seconds, dt)
        else:
            ticks = req.ticks or 1
            for _ in range(ticks):
                world.step(dt)
        snapshot = world.snapshot()
        events = world.drain_events()
    return {"sim_id": sim_id, "ticks": ticks, **snapshot, "events": events}


@app.get("/sims/{sim_id}/events", tags=["run"])
def get_events(sim_id: str, drain: bool = True) -> dict:
    world = _world(sim_id)
    with _lock:
        events = world.drain_events() if drain else [e.to_dict() for e in world.events]
    return {"sim_id": sim_id, "time_s": round(world.time_s, 3), "events": events}
