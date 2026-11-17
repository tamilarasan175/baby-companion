import base64

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture()
def client(db):
    with TestClient(main.app) as c:
        yield c


def test_log_and_list_events(client):
    r = client.post("/events", json={"type": "feed", "data": {"type": "formula", "amount_ml": 60}})
    assert r.status_code == 200
    events = client.get("/events").json()
    assert events[0]["type"] == "feed"
    assert events[0]["timestamp"].endswith("Z") or "+00:00" in events[0]["timestamp"]


def test_profile_and_validation(client):
    r = client.put("/api/profile", json={"name": "Maya", "birth_date": "2026-10-27", "birth_weight_g": "3400"})
    assert r.status_code == 200
    assert r.json()["profile"]["birth_weight_g"] == 3400
    assert client.put("/api/profile", json={"birth_date": "not-a-date"}).status_code == 400


def test_immunization_record(client):
    r = client.post("/api/immunizations", json={"name": "Vitamin K", "date": "2026-10-27", "tz": "America/Edmonton"})
    assert r.status_code == 200
    item = client.get("/api/immunizations").json()[0]
    assert item["name"] == "Vitamin K" and item["date"] == "2026-10-27"
    assert client.delete(f"/api/immunizations/{item['id']}").status_code == 200
    assert client.get("/api/immunizations").json() == []
    assert client.post("/api/immunizations", json={"name": "X", "date": "bad"}).status_code == 400


def test_visit_brief_includes_profile_and_vaccines(client):
    client.put("/api/profile", json={"name": "Maya", "birth_date": "2026-10-27", "birth_weight_g": 3400})
    client.post("/api/immunizations", json={"name": "Vitamin K", "date": "2026-10-27"})
    text = client.get("/api/visit-brief?tz=America/Edmonton").json()["copy_text"]
    assert "Maya" in text and "3400 g" in text and "Vitamin K" in text


def test_upload_rejects_unsupported_files(client):
    r = client.post("/documents", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_password_protects_everything(client, monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "s3cret")
    assert client.get("/events").status_code == 401
    assert client.get("/mcp/tools").status_code == 401
    good = base64.b64encode(b"parent:s3cret").decode()
    bad = base64.b64encode(b"parent:wrong").decode()
    assert client.get("/events", headers={"Authorization": f"Basic {good}"}).status_code == 200
    assert client.get("/events", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    # The web-app manifest must stay reachable (browsers fetch it without credentials).
    assert client.get("/static/manifest.json").status_code == 200
