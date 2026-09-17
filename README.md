# skysim

A small airspace simulator with an HTTP API. One flat block of airspace,
surface to ~60 km, up to **32 objects** at a time, all point masses obeying the
same force model: **thrust, drag, lift, buoyancy and weight**, integrated with
fixed-step RK4.

It is meant to be the engine behind something else — a traffic display, a
training tool, a game, a planner. Nothing renders, nothing guesses: you spawn
objects, step time, read state.

## What it models

| Piece | Model |
|---|---|
| Atmosphere | International Standard Atmosphere to 84.9 km (density, pressure, speed of sound), with an approximate exponential fit above |
| Gravity | Straight down, `g(h) = g0 · (R/(R+h))²` (~0.98 g at 60 km) |
| Drag | `D = ½ρV²·Cd·S`, with a transonic Cd rise and induced drag `Cl²/(π·AR·e)` |
| Lift | `L = ½ρV²·Cl·S` perpendicular to the flight path, capped by `Cl_max` and the airframe's g limit |
| Buoyancy | `B = ρ·V·g`, with gas envelopes expanding as `V ∝ 1/ρ` until the envelope's stretch limit — so balloons climb, float, then burst |
| Propulsion | Thrust along the flight path (jets) or along a commanded direction (rotorcraft, rockets), throttled, burning propellant at a fixed rate |
| Wind | A steady world-frame wind vector; all aerodynamics use airspeed, not ground speed |
| Integration | Classical RK4 at a fixed timestep (default 50 Hz), deterministic — same inputs, same trajectories |

Frame: local ENU, metres. **x = east, y = north, z = up**, `z = 0` is the
surface. Headings are compass degrees (0 = north, 90 = east).

## Install

```bash
pip install fastapi uvicorn pydantic          # API only; the engine is stdlib
python -m pytest tests -q                      # 38 tests
python examples/demo.py                        # busy-airspace scenario
```

## Use it as a library

```python
from skysim import World, Command, Mode, Vec3, heading_vector

world = World(dt=0.02, separation_m=1000.0, wind=Vec3(10, 0, 0))

plane = world.spawn("airliner", Vec3(0, 0, 9000), heading_vector(90) * 230,
                    command=Command(mode=Mode.HOLD, altitude_m=11000, speed_mps=240))
balloon = world.spawn("weather_balloon", Vec3(4000, 0, 200), Vec3(0, 0, 2))

world.run(300.0)                    # 300 s of sim time
print(plane.altitude_m, plane.mach, plane.fuel_kg)
print(world.drain_events())         # spawns, proximity alerts, bursts, impacts
```

## Use it over HTTP

```bash
python -m skysim --port 8000        # or: uvicorn skysim.api:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | liveness, sim count, per-sim capacity |
| `GET` | `/profiles` | every vehicle profile and its parameters |
| `POST` | `/sims` | new world (`dt`, `bounds`, `wind`, `separation_m`, `collisions`) |
| `GET` | `/sims` | list worlds |
| `GET` | `/sims/{sim}` | full snapshot: time, bounds, wind, every object |
| `PATCH` | `/sims/{sim}` | change `dt`, wind, separation threshold, collisions |
| `POST` | `/sims/{sim}/reset` | empty the world, reset the clock |
| `DELETE` | `/sims/{sim}` | drop the world |
| `POST` | `/sims/{sim}/objects` | spawn (profile, position, velocity **or** heading/speed/climb, command) |
| `GET` | `/sims/{sim}/objects` | all objects |
| `GET`/`DELETE` | `/sims/{sim}/objects/{id}` | one object |
| `PUT` | `/sims/{sim}/objects/{id}/command` | re-task in flight |
| `POST` | `/sims/{sim}/objects/{id}/release` | release a search-and-rescue drone from a carrier aircraft (see below) |
| `POST` | `/sims/{sim}/step` | advance by `seconds`, or `ticks` (default one tick); returns snapshot + events |
| `GET` | `/sims/{sim}/events` | drain the event queue |

```bash
SIM=$(curl -sX POST localhost:8000/sims -H 'content-type: application/json' \
      -d '{"dt":0.05}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["sim_id"])')

curl -sX POST localhost:8000/sims/$SIM/objects -H 'content-type: application/json' -d '{
  "profile": "airliner",
  "position": {"x": 0, "y": 0, "z": 9000},
  "launch": {"heading_deg": 90, "speed_mps": 230},
  "command": {"mode": "heading", "heading_deg": 45, "altitude_m": 11000, "speed_mps": 240}}'

curl -sX POST localhost:8000/sims/$SIM/step -H 'content-type: application/json' \
     -d '{"seconds": 120}'
```

Spawning past 32 objects returns **409**; an unknown profile returns **400**.

## Command modes

Every object flies one command, swappable at any time.

| Mode | Behaviour | Fields used |
|---|---|---|
| `ballistic` | unguided — gravity, drag, buoyancy only | — |
| `hold` | keep current heading/altitude, or the ones given | `heading_deg`, `altitude_m`, `speed_mps` |
| `heading` | fly a commanded heading, altitude and speed | same as `hold` |
| `waypoint` | steer to a point, then hold station over it | `waypoint`, `altitude_m`, `speed_mps` |
| `hover` | thrust-vectoring station keeping (rotorcraft) | `waypoint`, `altitude_m` |
| `follow` | trail another object at a standoff distance | `target_id`, `standoff_m` |
| `ascent` | rocket pitch program: vertical, then a set attitude | `pitch_deg`, `pitch_over_s`, `heading_deg` |
| `pursue` | close on a designated aircraft to attach a locator beacon | `target_id` (set via `release`, below) |

`throttle` (0–1) overrides the autopilot's own setting in any mode.

## Search and rescue

An aircraft can release a small chase drone that flies itself onto a
designated aircraft and attaches a locator beacon — for tracking a distressed
or lost aircraft, not for anything destructive: there is no payload beyond the
beacon, and the drone rides along afterward instead of doing anything to its
target.

```
POST /sims/{sim}/objects/{carrier_id}/release
{"target_id": 7, "profile": "sar_drone", "name": "beacon-1"}
```

The drone spawns clear of the carrier, inheriting its position and velocity,
and flies `Mode.PURSUE` toward `target_id`. Once within the profile's
`capture_radius_m` it attaches (event `beacon_attached`) and from then on
rides at the target's exact position and velocity every tick — it stops
flying itself and drops out of collision/proximity checks, the same as a
beacon fixed to the fuselage. If the target is later removed from the sim,
the drone detaches (`beacon_detached`) and goes back to flying on its own.

Steering differs by airframe: a winged chase drone (`sar_drone`) cannot brake
by reversing thrust, so it flies a converging pursuit course — proportional
navigation, the standard convergent-tracking law behind any moving-target
rendezvous, not something specific to weapons — rather than aiming at the
target and being unable to stop in time. A thrust-vectoring (rotorcraft)
drone servos straight onto the target instead, since it can brake in any
direction.

`sar_drone` catches anything flying well below its own ~260 m/s top speed —
light aircraft, gliders, a descending or slowed airliner — but won't run down
a jet airliner at cruise; only about 20 m/s faster, it cannot close the
distance before its fuel runs out. That's a real fuel/speed budget, not a
special case.

## Profiles

Generic placeholders, not any specific make or model — tune them freely, or
register your own with `skysim.register(Profile(...))`.

| Profile | Kind | Notes |
|---|---|---|
| `airliner` | aircraft | ~250 m/s cruise, 12.5 km ceiling, 16 t fuel |
| `light_aircraft` | aircraft | ~70 m/s, small piston-class airframe |
| `glider` | aircraft | unpowered, L/D around 28 |
| `survey_drone` | rotorcraft | thrust-vectoring, hovers, ~22 m/s |
| `sounding_rocket` | rocket | ~30 s burn, climbs past 50 km |
| `weather_balloon` | balloon | ~10 m/s climb, bursts at 30 km, parachutes down |
| `high_alt_platform` | balloon | superpressure, floats around 25 km |
| `dropsonde` | projectile | light, high drag, quick terminal velocity |
| `cargo_capsule` | projectile | heavy unguided payload |
| `sar_drone` | aircraft | fast fixed-wing chase drone, releases from a carrier to attach a SAR beacon |

## Events

Drained per step: `spawn`, `command`, `removed`, `fuel_exhausted`,
`balloon_burst`, `proximity_alert` / `proximity_clear` (pairs closer than
`separation_m`, using closest approach across the tick, not just endpoints),
`collision` (radii overlap — both objects destroyed), `ground_contact`
(`landed` under 12 m/s descent, otherwise `destroyed`), `exited_volume`,
`sar_drone_released`, `beacon_attached`, `beacon_detached`.

## Layout

```
skysim/vec.py        3D vector math
skysim/physics.py    atmosphere, drag, lift, buoyancy, thrust, gravity
skysim/profiles.py   vehicle parameters
skysim/bodies.py     object state and commands
skysim/control.py    autopilots: altitude, heading, waypoint, hover, follow, ascent
skysim/world.py      integrator, events, separation and collision checks
skysim/api.py        FastAPI layer
tests/               physics, flight behaviour and API tests
examples/demo.py     seven objects sharing one block of airspace
```

## Limits worth knowing

- Point masses only: no attitude dynamics, no control-surface lag, no
  structural or thermal modelling.
- Flat frame — no earth curvature, rotation or orbital mechanics. Above about
  60 km the atmosphere is thin enough that the model stops being interesting,
  which is where the default ceiling sits.
- Profile numbers are plausible, not authoritative. Treat trajectories as
  qualitative.
- Sims live in process memory; restarting the server drops them.
