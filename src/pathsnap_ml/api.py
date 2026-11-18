"""FastAPI entry point for the PathSnap evaluator workflow."""
import json
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from pydantic import BaseModel
import yaml

from .candidates import make_candidates
from .contracts import KEY, fingerprint, manifest
from .data_io import load_processed_trip, prepare_trip, read_raw_reference
from .decoder import decode_trip
from .features import make_features
from .inference import predict_candidates
from .reference_labels import align_reference
from .route_metrics import compute_continuity

LOG = logging.getLogger("uvicorn.error")
WORKSPACE = Path(__file__).resolve().parents[2]


class EvaluationRequest(BaseModel):
    trip_id: str


def _yaml(workspace, name):
    return yaml.safe_load((workspace / "configs" / name).read_text(encoding="utf-8"))


def _decoder_config(workspace, bundle, override):
    defaults = _yaml(workspace, "decoder.yaml")
    if override is not None:
        payload = yaml.safe_load(Path(override).read_text(encoding="utf-8"))
    elif bundle is not None and (bundle / "decoder_config.json").is_file():
        payload = json.loads((bundle / "decoder_config.json").read_text(encoding="utf-8"))
    else:
        return defaults
    config = payload.get("config", payload) if isinstance(payload, dict) else None
    if not isinstance(config, dict) or set(defaults) - set(config):
        raise ValueError("The model bundle needs a decoder config exported by cli_decode tune")
    return config


def _preview_scores(features):
    distance = features.distance_m.to_numpy()
    heading = features.heading_difference_deg.fillna(0).to_numpy()
    scores = features[KEY].copy()
    scores["p_match"] = np.exp(-distance / 25 - heading / 90).clip(1e-9, 1)
    return scores


def _accuracy(obs, predictions, labels):
    valid = obs.loc[obs.valid, ["trip_id", "obs_id"]]
    assigned = valid.merge(labels.loc[labels.label_status.eq("assigned"),
                                      ["trip_id", "obs_id", "target_arc_id"]],
                           on=["trip_id", "obs_id"], validate="one_to_one")
    checked = assigned.merge(predictions[["trip_id", "obs_id", "predicted_arc_id"]],
                             on=["trip_id", "obs_id"], how="left", validate="one_to_one")
    correct = int(checked.target_arc_id.eq(checked.predicted_arc_id).fillna(False).sum())
    return {"accuracy": correct / len(checked) if len(checked) else None,
            "correct": correct, "assigned": len(checked), "valid": len(valid)}


def _map_data(obs, nodes, arcs, route):
    valid = obs.loc[obs.valid].sort_values("order_index")
    node_xy = nodes.set_index("node_id")[["x_m", "y_m"]]
    node_geo = nodes.set_index("node_id")[["lat", "lon"]]
    roads = arcs.set_index("arc_id")[["from_node_id", "to_node_id"]]
    gps = valid[["x_m", "y_m"]].to_numpy()
    if len(gps):
        lo, hi = gps.min(axis=0), gps.max(axis=0)
    else:
        lo, hi = node_xy.min().to_numpy(), node_xy.max().to_numpy()
    if not np.isfinite([lo, hi]).all():
        lo, hi = np.array([-1., -1.]), np.array([1., 1.])
    pad = max(120., float(max(hi - lo)) * .15)
    start = node_xy.loc[arcs.from_node_id].to_numpy()
    end = node_xy.loc[arcs.to_node_id].to_numpy()
    within = ((np.maximum(start, end) >= lo - pad) & (np.minimum(start, end) <= hi + pad)).all(axis=1)
    context_edges = arcs.loc[within]

    def lat_lon_edges(edges):
        return np.stack([node_geo.loc[edges.from_node_id].to_numpy(),
                         node_geo.loc[edges.to_node_id].to_numpy()], axis=1).round(7).tolist()

    segments = []
    for run in route["runs"]:
        if not run["arc_ids"]:
            continue
        edges = roads.loc[run["arc_ids"]]
        segments.append(lat_lon_edges(edges))
    return {"gps": valid[["lat", "lon"]].to_numpy().round(7).tolist(),
            "roads": lat_lon_edges(context_edges), "runs": segments}


def create_app(workspace=WORKSPACE, evaluator_root=None, model_bundle=None, origins=(), decoder_config=None):
    workspace = Path(workspace).resolve()
    evaluator_root = Path(evaluator_root).resolve() if evaluator_root else None
    bundle = Path(model_bundle).resolve() if model_bundle else None
    feature_config = _yaml(workspace, "features.yaml")
    alignment_config = _yaml(workspace, "reference_alignment.yaml")
    decoder = _decoder_config(workspace, bundle, decoder_config)
    clean_config = _yaml(workspace, "data.yaml")
    feature_hash = fingerprint(feature_config)
    if evaluator_root is None:
        entries = {tid: None for tid in manifest(workspace).query("split == 'student_test'").trip_id}
    else:
        partition = evaluator_root / "evaluator_test"
        entries = {}
        for folder in sorted(partition.iterdir()):
            tid = folder.name
            if not folder.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]+", tid):
                continue
            paths = {f"{kind}_path": Path("evaluator_test", tid, f"{tid}.{kind}")
                     for kind in ("track", "nodes", "arcs", "route")}
            if not all((evaluator_root / path).is_file() for path in paths.values()):
                raise ValueError(f"Evaluator trip {tid} is missing a required input")
            entries[tid] = SimpleNamespace(trip_id=tid, split="evaluator_test", **paths)
        if len(entries) != 10:
            raise ValueError(f"Expected ten evaluator trips, found {len(entries)}")

    app = FastAPI(title="PathSnap Evaluator API", version="1.0")
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(origins),
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "placeholder" if bundle is None else "model",
                "partition": "student_test" if evaluator_root is None else "evaluator_test",
                "trip_count": len(entries)}

    @app.get("/api/trips")
    def trips():
        return {"trips": [{"trip_id": tid} for tid in entries], **health()}

    @app.post("/api/evaluate")
    def evaluate(request: EvaluationRequest):
        tid = request.trip_id
        if tid not in entries:
            raise HTTPException(404, "Trip is outside the configured evaluation set")
        if evaluator_root is not None and bundle is None:
            raise HTTPException(503, "Configure the trained model bundle before evaluator scoring")
        LOG.info("PathSnap inference start trip_id=%s mode=%s", tid, health()["mode"])
        try:
            row = entries[tid]
            trip = (load_processed_trip(workspace / "data" / "processed", tid) if row is None
                    else prepare_trip(evaluator_root, row, clean_config, allow_evaluator=True))
            obs, nodes, arcs, meta = (trip[key] for key in ("observations", "nodes", "arcs", "metadata"))
            candidates = make_candidates(obs, nodes, arcs, meta, feature_config)
            features = make_features(obs, nodes, arcs, candidates, meta, feature_config)
            scores = (_preview_scores(features) if bundle is None else
                      predict_candidates(bundle, features, feature_hash))
            mode = "classifier_only" if bundle is None else "hmm"
            predictions, route = decode_trip(obs, nodes, arcs, candidates, scores, meta, decoder, mode)
            continuity = compute_continuity(obs, nodes, arcs, candidates, predictions, meta, decoder)

            # Reference information is loaded only after inference and used only for scoring.
            if row is None:
                labels = pd.read_parquet(workspace / "data" / "labels" / f"{tid}.parquet")
            else:
                reference = read_raw_reference(evaluator_root, row, len(arcs), allow_evaluator=True)
                labels = align_reference(obs, nodes, arcs, reference, alignment_config)
            metrics = _accuracy(obs, predictions, labels)
            metrics["continuity"] = (int(continuity["feasible"]) / int(continuity["eligible"])
                                     if continuity["eligible"] else None)
            result = {"trip_id": tid, "mode": health()["mode"], "decoder": mode,
                      "metrics": metrics, "route_runs": len(route["runs"]),
                      "map": _map_data(obs, nodes, arcs, route)}
        except Exception:
            LOG.exception("PathSnap inference failed trip_id=%s", tid)
            raise HTTPException(500, "Trip evaluation failed. Check the API console.")
        LOG.info("PathSnap inference complete trip_id=%s correct=%s assigned=%s",
                 tid, metrics["correct"], metrics["assigned"])
        return result

    app.mount("/", StaticFiles(directory=WORKSPACE / "frontend", html=True), name="frontend")
    return app


def app_factory():
    return create_app(
        workspace=os.environ.get("PATHSNAP_WORKSPACE", WORKSPACE),
        evaluator_root=os.environ.get("PATHSNAP_EVALUATOR_ROOT"),
        model_bundle=os.environ.get("PATHSNAP_MODEL_BUNDLE"),
        decoder_config=os.environ.get("PATHSNAP_DECODER_CONFIG"),
        origins=tuple(value.strip() for value in os.environ.get("PATHSNAP_UI_ORIGINS", "").split(",")
                      if value.strip()),
    )
