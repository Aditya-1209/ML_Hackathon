from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pathsnap_ml import api


ROOT = Path(__file__).resolve().parents[1]


def test_preview_returns_ten_trips_map_and_aggregate_metrics():
    client = TestClient(api.create_app(ROOT, origins=("http://ui-laptop:8080",)))
    assert client.get("/").status_code == 200
    listing = client.get("/api/trips").json()
    assert listing["trip_count"] == 10
    assert listing["mode"] == "placeholder"
    trip = listing["trips"][0]["trip_id"]
    result = client.post("/api/evaluate", json={"trip_id": trip})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["trip_id"] == trip and body["mode"] == "placeholder"
    assert body["map"]["gps"] and body["map"]["roads"] and body["map"]["runs"]
    assert 0 <= body["metrics"]["accuracy"] <= 1
    assert "target_arc_id" not in result.text
    assert client.post("/api/evaluate", json={"trip_id": "../outside"}).status_code == 404
    preflight = client.options("/api/evaluate", headers={
        "Origin": "http://ui-laptop:8080", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert preflight.headers["access-control-allow-origin"] == "http://ui-laptop:8080"


@pytest.fixture
def evaluator_root(tmp_path):
    for i in range(10):
        tid = f"trial_{i}"
        folder = tmp_path / "evaluator_test" / tid
        folder.mkdir(parents=True)
        (folder / f"{tid}.track").write_text("8.00010 49.0 0\n8.00019 49.0 1\n")
        (folder / f"{tid}.nodes").write_text("8.0 49.0\n8.00030 49.0\n")
        (folder / f"{tid}.arcs").write_text("0 1\n")
        (folder / f"{tid}.route").write_text("0\n")
    return tmp_path


def test_evaluator_requires_configured_model(evaluator_root):
    client = TestClient(api.create_app(ROOT, evaluator_root=evaluator_root))
    assert client.get("/api/trips").json()["partition"] == "evaluator_test"
    assert client.post("/api/evaluate", json={"trip_id": "trial_0"}).status_code == 503


def test_evaluator_loads_reference_after_model_and_decoder(evaluator_root, monkeypatch):
    bundle = evaluator_root / "bundle"
    bundle.mkdir()
    phases = []
    def predict(bundle, features, feature_config_hash):
        phases.append("predict")
        assert "target_arc_id" not in features
        return api._preview_scores(features)

    decode = api.decode_trip
    def trace_decode(*args, **kwargs):
        phases.append("decode")
        return decode(*args, **kwargs)

    reference = api.read_raw_reference
    def trace_reference(*args, **kwargs):
        phases.append("reference")
        return reference(*args, **kwargs)

    monkeypatch.setattr(api, "predict_candidates", predict)
    monkeypatch.setattr(api, "decode_trip", trace_decode)
    monkeypatch.setattr(api, "read_raw_reference", trace_reference)
    client = TestClient(api.create_app(ROOT, evaluator_root=evaluator_root, model_bundle=bundle))
    result = client.post("/api/evaluate", json={"trip_id": "trial_0"})
    assert result.status_code == 200, result.text
    assert phases == ["predict", "decode", "reference"]
    body = result.json()
    assert body["decoder"] == "hmm"
    assert body["metrics"]["correct"] == 2
    assert "target_arc_id" not in result.text


def test_no_label_and_missing_prediction_denominators():
    obs = pd.DataFrame({"trip_id": ["a", "a"], "obs_id": [0, 1], "valid": [True, True]})
    pred = pd.DataFrame({"trip_id": ["a"], "obs_id": [0], "predicted_arc_id": [7]})
    labels = pd.DataFrame({"trip_id": ["a", "a"], "obs_id": [0, 1],
                           "target_arc_id": [7, 9], "label_status": ["assigned", "assigned"]})
    assert api._accuracy(obs, pred, labels) == {"accuracy": .5, "correct": 1, "assigned": 2, "valid": 2}
    assert api._accuracy(obs, pred, labels.iloc[:0])["accuracy"] is None
