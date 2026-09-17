import pytest
from fastapi.testclient import TestClient

from skysim.api import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sim(client):
    sim_id = client.post("/sims", json={"dt": 0.05, "separation_m": 1500.0}).json()["sim_id"]
    yield sim_id
    client.delete(f"/sims/{sim_id}")


def test_health_and_profiles(client):
    assert client.get("/health").json()["capacity_per_sim"] == 32
    names = [p["name"] for p in client.get("/profiles").json()["profiles"]]
    assert {"airliner", "weather_balloon", "sounding_rocket", "survey_drone"} <= set(names)


def test_create_inspect_delete_sim(client):
    created = client.post("/sims", json={"dt": 0.1}).json()
    sim_id = created["sim_id"]
    assert created["capacity"] == 32
    assert client.get(f"/sims/{sim_id}").json()["dt"] == 0.1
    assert any(s["sim_id"] == sim_id for s in client.get("/sims").json()["sims"])
    assert client.delete(f"/sims/{sim_id}").status_code == 200
    assert client.get(f"/sims/{sim_id}").status_code == 404


def test_spawn_and_fly_an_airliner(client, sim):
    spawned = client.post(f"/sims/{sim}/objects", json={
        "profile": "airliner",
        "position": {"x": 0, "y": 0, "z": 9000},
        "launch": {"heading_deg": 0, "speed_mps": 230},
        "command": {"mode": "heading", "heading_deg": 90, "altitude_m": 11000,
                    "speed_mps": 240},
    }).json()
    assert spawned["kind"] == "aircraft"

    result = client.post(f"/sims/{sim}/step", json={"seconds": 300}).json()
    assert result["ticks"] == 6000
    plane = result["objects"][0]
    assert plane["altitude_m"] == pytest.approx(11000.0, abs=50.0)
    assert plane["heading_deg"] == pytest.approx(90.0, abs=2.0)
    assert plane["fuel_kg"] < 16000.0
    assert any(e["kind"] == "spawn" for e in result["events"])


def test_step_defaults_to_one_tick(client, sim):
    client.post(f"/sims/{sim}/objects", json={
        "profile": "cargo_capsule", "position": {"x": 0, "y": 0, "z": 5000}})
    before = client.get(f"/sims/{sim}").json()["time_s"]
    stepped = client.post(f"/sims/{sim}/step").json()
    assert stepped["ticks"] == 1
    assert stepped["time_s"] == pytest.approx(before + 0.05)


def test_unknown_profile_is_rejected(client, sim):
    resp = client.post(f"/sims/{sim}/objects",
                       json={"profile": "flying_saucer", "position": {"z": 1000}})
    assert resp.status_code == 400
    assert "unknown profile" in resp.json()["detail"]


def test_capacity_returns_409(client, sim):
    for _ in range(32):
        assert client.post(f"/sims/{sim}/objects", json={
            "profile": "cargo_capsule", "position": {"x": 0, "y": 0, "z": 20000}
        }).status_code == 201
    over = client.post(f"/sims/{sim}/objects", json={
        "profile": "cargo_capsule", "position": {"x": 0, "y": 0, "z": 20000}})
    assert over.status_code == 409
    assert "32" in over.json()["detail"]


def test_update_command_mid_flight(client, sim):
    obj = client.post(f"/sims/{sim}/objects", json={
        "profile": "survey_drone", "position": {"x": 0, "y": 0, "z": 10},
        "command": {"mode": "hover", "waypoint": {"x": 0, "y": 0, "z": 0},
                    "altitude_m": 100},
    }).json()
    client.post(f"/sims/{sim}/step", json={"seconds": 60})
    updated = client.put(f"/sims/{sim}/objects/{obj['id']}/command", json={
        "mode": "hover", "waypoint": {"x": 400, "y": 0, "z": 0}, "altitude_m": 150}).json()
    assert updated["command"]["altitude_m"] == 150
    client.post(f"/sims/{sim}/step", json={"seconds": 120})
    final = client.get(f"/sims/{sim}/objects/{obj['id']}").json()
    assert final["position"]["x"] == pytest.approx(400.0, abs=10.0)
    assert final["altitude_m"] == pytest.approx(150.0, abs=3.0)


def test_events_drain_once(client, sim):
    client.post(f"/sims/{sim}/objects", json={
        "profile": "dropsonde", "position": {"x": 0, "y": 0, "z": 300}})
    first = client.get(f"/sims/{sim}/events").json()["events"]
    assert any(e["kind"] == "spawn" for e in first)
    assert client.get(f"/sims/{sim}/events").json()["events"] == []


def test_proximity_alert_surfaces_through_the_api(client, sim):
    for x, heading in ((-15000, 90), (15000, 270)):
        client.post(f"/sims/{sim}/objects", json={
            "profile": "airliner", "position": {"x": x, "y": 300, "z": 10000},
            "launch": {"heading_deg": heading, "speed_mps": 240},
            "command": {"mode": "hold", "altitude_m": 10000, "speed_mps": 240},
        })
    kinds = set()
    for _ in range(6):
        kinds |= {e["kind"] for e in
                  client.post(f"/sims/{sim}/step", json={"seconds": 20}).json()["events"]}
    assert "proximity_alert" in kinds or "collision" in kinds


def test_wind_and_settings_patch(client, sim):
    patched = client.patch(f"/sims/{sim}", json={"wind": {"x": 15, "y": 0, "z": 0},
                                                 "separation_m": 2500}).json()
    assert patched["wind"]["x"] == 15
    assert patched["separation_m"] == 2500


def test_reset_clears_objects(client, sim):
    client.post(f"/sims/{sim}/objects", json={
        "profile": "glider", "position": {"x": 0, "y": 0, "z": 2000}})
    client.post(f"/sims/{sim}/step", json={"seconds": 5})
    after = client.post(f"/sims/{sim}/reset").json()
    assert after["count"] == 0 and after["time_s"] == 0.0


def test_delete_object_and_404s(client, sim):
    obj = client.post(f"/sims/{sim}/objects", json={
        "profile": "glider", "position": {"x": 0, "y": 0, "z": 2000}}).json()
    assert client.delete(f"/sims/{sim}/objects/{obj['id']}").status_code == 200
    assert client.get(f"/sims/{sim}/objects/{obj['id']}").status_code == 404
    assert client.put(f"/sims/{sim}/objects/999/command",
                      json={"mode": "hold"}).status_code == 404


def test_release_sar_drone_via_api(client, sim):
    target = client.post(f"/sims/{sim}/objects", json={
        "profile": "light_aircraft", "position": {"x": 0, "y": 0, "z": 1500},
        "launch": {"heading_deg": 90, "speed_mps": 65},
        "command": {"mode": "hold", "altitude_m": 1500, "speed_mps": 65},
    }).json()
    carrier = client.post(f"/sims/{sim}/objects", json={
        "profile": "airliner", "position": {"x": -20000, "y": -3000, "z": 9000},
        "launch": {"heading_deg": 70, "speed_mps": 230},
        "command": {"mode": "hold", "altitude_m": 9000, "speed_mps": 230},
    }).json()

    drone = client.post(f"/sims/{sim}/objects/{carrier['id']}/release",
                        json={"target_id": target["id"], "name": "beacon-1"}).json()
    assert drone["command"]["mode"] == "pursue"
    assert drone["command"]["target_id"] == target["id"]

    attached = False
    for _ in range(40):
        result = client.post(f"/sims/{sim}/step", json={"seconds": 20}).json()
        if any(e["kind"] == "beacon_attached" for e in result["events"]):
            attached = True
            break
    assert attached

    snap = client.get(f"/sims/{sim}/objects/{drone['id']}").json()
    assert snap["attached_to"] == target["id"]


def test_release_rejects_unequipped_profile_and_unknown_ids(client, sim):
    target = client.post(f"/sims/{sim}/objects", json={
        "profile": "light_aircraft", "position": {"x": 0, "y": 0, "z": 1500}}).json()
    carrier = client.post(f"/sims/{sim}/objects", json={
        "profile": "airliner", "position": {"x": -5000, "y": 0, "z": 9000}}).json()

    bad_profile = client.post(f"/sims/{sim}/objects/{carrier['id']}/release",
                              json={"target_id": target["id"], "profile": "cargo_capsule"})
    assert bad_profile.status_code == 400

    unknown_target = client.post(f"/sims/{sim}/objects/{carrier['id']}/release",
                                 json={"target_id": 99999})
    assert unknown_target.status_code == 404

    unknown_carrier = client.post(f"/sims/{sim}/objects/99999/release",
                                  json={"target_id": target["id"]})
    assert unknown_carrier.status_code == 404
