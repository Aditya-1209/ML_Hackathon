"""Tune the Viterbi segment-break cost on held-out training-trip forest probabilities."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
import yaml

from .contracts import manifest
from .decoder_evaluation import evaluate_decoder
from .evaluate_forest_routes import score_training_trip
from .tune_context_routes import TUNED, PROFILE


def run(root, workers):
    rows = manifest(root)
    folds = pd.read_csv(root / "data/folds.csv", dtype={"trip_id": str})
    train_ids = rows.query("split == 'train'").trip_id
    if (len(train_ids) != 80 or set(train_ids) != set(folds.trip_id)
            or folds.fold_id.value_counts().to_dict() != dict.fromkeys(range(5), 16)):
        raise ValueError("Expected the 80 published training trips and five grouped folds")
    items = list(folds.itertuples(index=False, name=None))
    model_dir = root / "results" / PROFILE / "models" / TUNED
    cv = model_dir / "cv_scores.parquet"
    base = yaml.safe_load((root / "configs/decoder.yaml").read_text())
    output_name = TUNED + "_segment_breaks"
    out = root / "results/decoder" / PROFILE / output_name
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def objective(trial):
            beta = trial.suggest_float("beta_m", 10, 150, log=True)
            penalty = trial.suggest_float("break_penalty", 0.5, 8, log=True)
            worker = partial(score_training_trip, root=root, cv_file=cv, model_id=TUNED,
                             base_config={**base, "break_penalty": penalty}, betas=(beta,))
            per_trip = pd.DataFrame([r for result in pool.map(worker, items) for r in result])
            grouped = per_trip.groupby("fold_id")[["correct", "labelled"]].sum()
            scores = grouped.correct / grouped.labelled
            trial.set_user_attr("fold_top1", scores.to_list())
            records.append(per_trip.assign(trial=trial.number, break_penalty=penalty))
            pd.concat(records, ignore_index=True).to_csv(out / "training_trials_by_trip.csv", index=False)
            return scores.mean()

        study = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
        study.optimize(objective, n_trials=10, callbacks=[
            lambda s, t: s.trials_dataframe().to_csv(out / "trials.csv", index=False)])
    best = study.best_trial
    config = {**base, **best.params}
    selection = dict(model_id=TUNED, config=config, trial=best.number, mean_fold_top1=best.value,
                     fold_top1=best.user_attrs["fold_top1"], search_seconds=time.monotonic()-started,
                     cv_sha256=hashlib.sha256(cv.read_bytes()).hexdigest(),
                     objective="mean of five trip-grouped fold decoded Top-1 accuracies", seed=42)
    (out / "decoder_config.json").write_text(json.dumps(selection, indent=2))
    test_ids = rows.query("split == 'student_test'").trip_id
    provenance = pd.read_parquet(model_dir / "student_test_scores.parquet", columns=["trip_id", "model_id", "fold_id"])
    if (len(test_ids) != 10 or set(test_ids) & set(train_ids) or set(provenance.trip_id) != set(test_ids)
            or not provenance.model_id.eq(TUNED).all() or provenance.fold_id.notna().any()):
        raise ValueError("Unexpected student-test scores")
    result = evaluate_decoder(root, TUNED, PROFILE, "hmm", config, "student_test", output_name=output_name)
    print("Student-test decoded accuracy", result["pointwise_accuracy"], "continuity", result["continuity"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run(args.workspace.resolve(), args.workers)
