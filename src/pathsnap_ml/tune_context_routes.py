"""Bayesian search and HMM/Viterbi evaluation for the completed-trip context forest."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from .context_forest import MODEL_ID, WORKERS, load_partition, predict
from .contracts import KEY, manifest
from .evaluation import evaluate, top_predictions
from .experiments import estimator_for
from .evaluate_forest_routes import evaluate_routes

TUNED = MODEL_ID + "_tuned"
PROFILE = "experiments"


def train(root):
    started = time.monotonic()
    rows = manifest(root)
    tids = rows.query("split == 'train'").trip_id
    folds = pd.read_csv(root / "data/folds.csv", dtype={"trip_id": str}).set_index("trip_id").fold_id
    if len(tids) != 80 or set(tids) != set(folds.index) or folds.value_counts().to_dict() != dict.fromkeys(range(5), 16):
        raise ValueError("Expected the published five folds of 16 training trips")
    out = root / "results" / PROFILE / "models" / TUNED
    out.mkdir(exist_ok=True, parents=True)
    x, labels, obs = load_partition(root, tids)
    columns = [c for c in x if c not in KEY]
    x[columns] = x[columns].astype("float32")
    truth = obs.query("valid")[KEY[:2]].merge(labels.query("label_status == 'assigned'")[KEY[:2] + ["target_arc_id"]],
                                              on=KEY[:2], validate="one_to_one")
    labelled = x.merge(truth, on=KEY[:2], validate="many_to_one")
    # Only assigned observations enter the search. The denominator includes candidate misses.
    y = labelled.candidate_arc_id.eq(labelled.target_arc_id).astype(int)
    fold_ids = labelled.trip_id.map(folds)
    array = labelled[columns].to_numpy(dtype=np.float32)
    groups = []
    for fold in range(5):
        valid = fold_ids.eq(fold)
        keys = labelled.loc[valid, KEY + ["target_arc_id"]]
        groups.append((array[~valid], y.loc[~valid], array[valid], keys,
                       int(truth.trip_id.map(folds).eq(fold).sum())))

    def estimator(params):
        return estimator_for("random_forest", {"model__n_jobs": WORKERS,
                             **{"model__" + k: v for k, v in params.items()}})

    def objective(trial):
        params = dict(n_estimators=trial.suggest_categorical("n_estimators", [32, 48, 64]),
                      max_depth=trial.suggest_int("max_depth", 14, 26),
                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12, log=True),
                      max_samples=trial.suggest_float("max_samples", 0.15, 0.45, step=0.05))
        scores = []
        for fit, target, held, keys, n in groups:
            forest = estimator(params).fit(fit, target)
            choices = top_predictions(keys.assign(p_match=forest.predict_proba(held)[:, 1]))
            scores.append(float(choices.candidate_arc_id.eq(choices.target_arc_id).sum()/n))
        trial.set_user_attr("fold_top1", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
    search_start = time.monotonic()
    study.optimize(objective, n_trials=10, callbacks=[
        lambda s, t: s.trials_dataframe().to_csv(out / "trials.csv", index=False)])
    search_seconds = time.monotonic() - search_start
    best = study.best_trial
    record = dict(model_id=TUNED, params=best.params, cv_mean_top1=best.value, fold_top1=best.user_attrs["fold_top1"],
                  trial=best.number, trials=len(study.trials), search_seconds=search_seconds, seed=42,
                  sampler="Optuna TPE", startup_trials=3, objective="equal-weight mean of five trip-grouped fold Top-1 accuracies",
                  feature_columns=columns)
    (out/"best.json").write_text(json.dumps(record, indent=2))
    del groups, array
    predictions = []
    for fold in range(5):
        held = x.trip_id.map(folds).eq(fold)
        fit = fold_ids.ne(fold)
        forest = estimator(best.params).fit(labelled.loc[fit, columns], y[fit])
        scored = predict(forest, x.loc[held], columns, fold).assign(model_id=TUNED)
        predictions.append(scored)
        metric = evaluate(scored, labels, obs.loc[obs.trip_id.map(folds).eq(fold)])
        if not np.isclose(metric["top1_accuracy"], record["fold_top1"][fold], rtol=0, atol=1e-12):
            raise ValueError("Winning trial failed to reproduce")
    pd.concat(predictions, ignore_index=True).to_parquet(out/"cv_scores.parquet", index=False)
    tic = time.monotonic()
    fitted = estimator(best.params).fit(labelled[columns], y)
    record["final_fit_seconds"] = time.monotonic()-tic
    joblib.dump(fitted, out/"model.joblib", compress=3)
    test_ids = rows.query("split == 'student_test'").trip_id
    if len(test_ids) != 10 or set(test_ids) & set(tids):
        raise ValueError("Unexpected student-test trips")
    test, targets, observations = load_partition(root, test_ids)
    scored = predict(fitted, test, columns).assign(model_id=TUNED)
    scored.to_parquet(out/"student_test_scores.parquet", index=False)
    record["student_test"] = evaluate(scored, targets, observations)
    record["training_experiment_seconds"] = time.monotonic()-started
    (out/"metrics.json").write_text(json.dumps(record, indent=2))
    print("Tuned forest student Top-1", record["student_test"]["top1_accuracy"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=4, help="Decoder worker processes")
    parser.add_argument("--decoder-only", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    if not args.decoder_only:
        train(root)
    evaluate_routes(root, TUNED, PROFILE, args.workers)
