"""Random Forest experiment for completed GPS trips. Run with python -m pathsnap_ml.context_forest."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .contracts import KEY, manifest, score_keys
from .evaluation import evaluate, top_predictions
from .experiments import estimator_for

MODEL_ID = "random_forest_gps_context"
DISTANCES = (12, 40, 90)
WORKERS = 5


def continuations(arcs, reverse):
    # At either end, continue on an incident directed road. Omit immediate U-turns.
    start, end = ("to_node_id", "from_node_id") if reverse else ("from_node_id", "to_node_id")
    outgoing = arcs.groupby(start).indices
    ends = arcs[end].to_numpy()
    options = [[j for j in outgoing.get(b, ()) if ends[j] != a]
               for a, b in zip(arcs[start], arcs[end])]
    result = np.full((len(arcs), max(1, max(map(len, options), default=0))), -1, dtype=int)
    for i, successors in enumerate(options):
        result[i, :len(successors)] = successors
    return result


def trip_features(root, tid):
    folder = root / "data/processed" / tid
    if not json.loads((folder / "metadata.json").read_text())["graph_directed"]:
        raise ValueError("These context features require a directed road graph")
    obs = pd.read_parquet(folder / "observations.parquet").sort_values("order_index").reset_index(drop=True)
    arcs = pd.read_parquet(folder / "arcs.parquet").reset_index(drop=True)
    nodes = pd.read_parquet(folder / "nodes.parquet").set_index("node_id")[["x_m", "y_m"]]
    frame = pd.read_parquet(root / "data/features" / f"{tid}.parquet")
    a = pd.Index(arcs.arc_id).get_indexer(frame.candidate_arc_id)
    i = pd.Index(obs.obs_id).get_indexer(frame.obs_id)
    if (a < 0).any() or (i < 0).any():
        raise ValueError("Unknown candidate road or observation")

    start = nodes.loc[arcs.from_node_id].to_numpy()
    delta = nodes.loc[arcs.to_node_id].to_numpy() - start
    length = np.linalg.norm(delta, axis=1)
    unit = delta / np.where(length > 0, length, 1)[:, None]
    fraction = frame.projection_fraction.to_numpy()
    projected = start[a] + fraction[:, None] * delta[a]

    gps = obs[["x_m", "y_m"]].to_numpy()
    times = obs.timestamp_s.to_numpy()
    valid = obs.valid.to_numpy(dtype=bool)
    dt = np.diff(times)
    connected = valid[1:] & valid[:-1] & (dt > 0) & (dt <= 60)
    run = np.r_[0, np.cumsum(~connected)]
    step = np.linalg.norm(np.diff(gps, axis=0), axis=1)
    progress = np.r_[0., np.cumsum(np.where(connected, step, 0.))]
    nearest = frame.groupby("obs_id").distance_m.transform("min")
    frame["excess_distance_m"] = frame.distance_m - nearest

    for sign, name in [(-1, "approach"), (1, "departure")]:
        links = continuations(arcs, reverse=(sign < 0))
        remaining = length[a] * (fraction if sign < 0 else 1 - fraction)
        endpoint = projected + sign * unit[a] * remaining[:, None]
        for distance in DISTANCES:
            # Select a nearby GPS sample by travelled distance, avoiding noisy one-step headings.
            goal = progress + sign * distance
            idx = (np.searchsorted(progress, goal, side="right") - 1 if sign < 0
                   else np.searchsorted(progress, goal, side="left"))
            available = (idx >= 0) & (idx < len(obs))
            idx = np.clip(idx, 0, len(obs) - 1)
            available &= (run == run[idx]) & (np.abs(times - times[idx]) <= 120)
            other = gps[idx[i]]
            movement = other - gps[i]
            displacement = np.linalg.norm(movement, axis=1)
            similarity = sign * np.einsum("ij,ij->i", movement, unit[a]) / np.maximum(displacement, 1e-6)
            usable = available[i] & (displacement >= 2) & (length[a] > 0)
            frame[f"{name}_{distance}_alignment"] = np.where(usable, similarity, np.nan)

            # Compare observed GPS position with travel of equal length along this road
            # and, when necessary, one directed connecting road.
            travelled = np.abs(progress[idx[i]] - progress[i])
            predicted = projected + sign * unit[a] * np.minimum(travelled, remaining)[:, None]
            error = np.linalg.norm(predicted - other, axis=1)
            overflow = travelled - remaining
            for column in links.T:
                nxt = column[a]
                use = (overflow > 0) & (nxt >= 0) & available[i]
                k = nxt[use]
                extra = np.minimum(overflow[use], length[k])
                extended = endpoint[use] + sign * unit[k] * extra[:, None]
                error[use] = np.minimum(error[use], np.linalg.norm(extended - other[use], axis=1))
            key = f"{name}_{distance}_error_m"
            frame[key] = np.where(available[i], error, np.nan)
            frame[key + "_above_best"] = frame[key] - frame.groupby("obs_id")[key].transform("min")
    return frame


def load_partition(root, trips):
    features = pd.concat([trip_features(root, tid) for tid in trips], ignore_index=True)
    labels = pd.concat([pd.read_parquet(root / "data/labels" / f"{tid}.parquet") for tid in trips])
    obs = pd.concat([pd.read_parquet(root / "data/processed" / tid / "observations.parquet",
                                    columns=["trip_id", "obs_id", "valid"]) for tid in trips])
    return features, labels, obs


def fit_model(features, labels, columns):
    assigned = labels.loc[labels.label_status.eq("assigned"), ["trip_id", "obs_id", "target_arc_id"]]
    rows = features.merge(assigned, on=["trip_id", "obs_id"], validate="many_to_one")
    model = estimator_for("random_forest", {"model__n_jobs": WORKERS})
    model.fit(rows[columns], rows.candidate_arc_id.eq(rows.target_arc_id).astype(int))
    return model


def predict(model, frame, columns, fold=None):
    scores = frame[KEY].copy()
    scores["p_match"] = model.predict_proba(frame[columns])[:, list(model.classes_).index(1)]
    scores["model_id"] = MODEL_ID
    scores["fold_id"] = pd.array([fold] * len(frame), dtype="Int64")
    score_keys(frame, scores)
    return scores


def run(root):
    out = root / "results/experiments" / MODEL_ID
    out.mkdir(parents=True, exist_ok=True)
    rows = manifest(root)
    baseline_dir = root / "results/baseline/models/random_forest"
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    folds = pd.read_csv(root / "data/folds.csv", dtype={"trip_id": str}).set_index("trip_id").fold_id
    train_ids = rows.query("split == 'train'").trip_id
    if not folds.index.is_unique or len(folds) != 80 or set(folds.index) != set(train_ids):
        raise ValueError("Expected published folds for all 80 training trips")
    x, labels, obs = load_partition(root, train_ids)
    columns = [c for c in x if c not in KEY]
    x[columns] = x[columns].astype("float32")
    fold_ids = x.trip_id.map(folds)
    metrics, predictions = [], []
    for fold in range(5):
        held = fold_ids.eq(fold)
        tic = time.monotonic()
        model = fit_model(x.loc[~held], labels, columns)
        scores = predict(model, x.loc[held], columns, fold)
        m = evaluate(scores, labels, obs[obs.trip_id.isin(folds[folds.eq(fold)].index)])
        metrics.append(m)
        predictions.append(scores)
        print(f"Fold {fold}: {m['top1_accuracy']:.4%} ({time.monotonic()-tic:.1f}s)", flush=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "cv_scores.parquet", index=False)
    cv = float(np.mean([m["top1_accuracy"] for m in metrics]))
    record = dict(model_id=MODEL_ID, feature_columns=columns, seed=42, sklearn=sklearn.__version__,
                  cv_mean_top1=cv, folds=metrics, baseline_cv_mean_top1=baseline["cv_mean_top1"],
                  selection_rule="Refit and open student test once only if mean grouped CV exceeds original forest.")
    (out / "validation.json").write_text(json.dumps(record, indent=2))
    if cv <= baseline["cv_mean_top1"]:
        print("Validation did not improve; stopped before student test.", flush=True)
        return
    tic = time.monotonic()
    model = fit_model(x, labels, columns)
    fit_s = time.monotonic() - tic
    joblib.dump(model, out / "model.joblib", compress=3)
    test_ids = rows.query("split == 'student_test'").trip_id
    if len(test_ids) != 10 or set(test_ids) & set(train_ids):
        raise ValueError("Unexpected student-test split")
    tic = time.monotonic()
    test, truth, test_obs = load_partition(root, test_ids)
    scores = predict(model, test, columns)
    inference_s = time.monotonic() - tic
    original = pd.read_parquet(baseline_dir / "student_test_scores.parquet")
    score_keys(scores, original)
    result = evaluate(scores, truth, test_obs)
    points = truth.query("label_status == 'assigned'").merge(test_obs.query("valid"), on=KEY[:2])
    for name, table in [("baseline", original), ("context", scores)]:
        choices = top_predictions(table)[KEY].rename(columns={"candidate_arc_id": name})
        points = points.merge(choices, on=KEY[:2], how="left", validate="one_to_one")
        points[name + "_correct"] = points[name].eq(points.target_arc_id).fillna(False)
    assert len(points) == result["eligible_observations"]
    assert np.isclose(points.baseline_correct.mean(), baseline["student_test"]["top1_accuracy"])
    assert np.isclose(points.context_correct.mean(), result["top1_accuracy"])
    paired = points.groupby("trip_id").agg(labelled=("obs_id", "count"), baseline_correct=("baseline_correct", "sum"),
                                          context_correct=("context_correct", "sum"))
    paired.to_csv(out / "paired_trips.csv")
    scores.to_parquet(out / "student_test_scores.parquet", index=False)
    summary = dict(student_test=result, baseline_top1=baseline["student_test"]["top1_accuracy"],
                   fit_seconds=fit_s, student_load_features_and_score_seconds=inference_s,
                   corrected=int((~points.baseline_correct & points.context_correct).sum()),
                   newly_wrong=int((points.baseline_correct & ~points.context_correct).sum()))
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(f"Student-test Top-1: {result['top1_accuracy']:.4%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    run(parser.parse_args().workspace.resolve())
