"""Select the untuned forest's decoder on OOF training scores, then evaluate both modes."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from .contracts import manifest
from .data_io import load_processed_trip
from .decoder import decode_trip
from .decoder_evaluation import evaluate_decoder

MODEL = "random_forest"
BETAS = (20, 50, 100)


def score_training_trip(item, root, cv_file, base_config, model_id):
    trip_id, fold_id = item
    scores = pd.read_parquet(cv_file, filters=[("trip_id", "==", trip_id)])
    if scores.empty or not scores.model_id.eq(model_id).all() or not scores.fold_id.eq(fold_id).all():
        raise ValueError(f"Unexpected training score provenance for {trip_id}")
    trip = load_processed_trip(root / "data/processed", trip_id)
    obs = trip["observations"]
    candidates = pd.read_parquet(root / "data/candidates" / f"{trip_id}.parquet")
    labels = pd.read_parquet(root / "data/labels" / f"{trip_id}.parquet")
    eligible = obs.loc[obs.valid, ["trip_id", "obs_id"]].merge(
        labels[labels.label_status.eq("assigned")], on=["trip_id", "obs_id"], validate="one_to_one")
    records = []
    for beta in BETAS:
        predictions, _ = decode_trip(obs, trip["nodes"], trip["arcs"], candidates, scores,
                                     trip["metadata"], {**base_config, "beta_m": beta})
        checked = eligible.merge(predictions, on=["trip_id", "obs_id"], how="left", validate="one_to_one")
        correct = checked.target_arc_id.eq(checked.predicted_arc_id).fillna(False)
        records.append(dict(trip_id=trip_id, fold_id=fold_id, beta_m=beta,
                            correct=int(correct.sum()), labelled=len(eligible)))
    return records


def evaluate_routes(root, model_id, profile, workers):
    original = root / "results" / profile / "models" / model_id
    cv_file = original / "cv_scores.parquet"
    base_config = yaml.safe_load((root / "configs/decoder.yaml").read_text())
    rows = manifest(root)
    folds = pd.read_csv(root / "data/folds.csv", dtype={"trip_id": str})
    train_ids = set(rows[rows.split.eq("train")].trip_id)
    if len(folds) != 80 or set(folds.trip_id) != train_ids:
        raise ValueError("Unexpected training fold assignment")
    out = root / "results/decoder" / profile / model_id
    out.mkdir(parents=True, exist_ok=True)
    process = partial(score_training_trip, root=root, cv_file=cv_file, base_config=base_config, model_id=model_id)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, trial_rows in enumerate(pool.map(process, folds.itertuples(index=False, name=None)), 1):
            records.extend(trial_rows)
            if i % 10 == 0:
                print(f"Decoded {i}/80 training trips at each beta", flush=True)
    trials = pd.DataFrame(records)
    trials.to_csv(out / "training_trials_by_trip.csv", index=False)
    totals = trials.groupby("beta_m", as_index=False)[["correct", "labelled"]].sum()
    totals["accuracy"] = totals.correct / totals.labelled
    beta = int(totals.sort_values(["accuracy", "beta_m"], ascending=[False, True]).beta_m.iloc[0])
    config = {**base_config, "beta_m": beta}
    record = dict(model_id=model_id, config=config, objective="pooled training OOF observation Top-1",
                  trials=totals.to_dict("records"), cv_scores_sha256=hashlib.sha256(cv_file.read_bytes()).hexdigest())
    (out / "decoder_config.json").write_text(json.dumps(record, indent=2))
    test_scores = pd.read_parquet(original / "student_test_scores.parquet", columns=["trip_id", "model_id", "fold_id"])
    if (set(test_scores.trip_id) != set(rows[rows.split.eq("student_test")].trip_id)
            or not test_scores.model_id.eq(model_id).all() or test_scores.fold_id.notna().any()):
        raise ValueError("Unexpected student-test score provenance")
    comparison = {mode: evaluate_decoder(root, model_id, profile, mode, config, "student_test")
                  for mode in ("classifier_only", "hmm")}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    for mode, metric in comparison.items():
        print(mode, metric["pointwise_accuracy"], metric["continuity"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    evaluate_routes(args.workspace.resolve(), MODEL, "baseline", args.workers)


if __name__ == "__main__":
    main()
