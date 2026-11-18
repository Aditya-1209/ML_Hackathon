"""Reproduce the fixed-budget Section 7 experiment from the repository root."""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import sklearn

from .contracts import FEATURES, KEY, manifest
from .evaluation import top_predictions
from .experiments import estimator_for, make_folds, positive_scores, run_experiment


def prepare_search(root):
    trips = manifest(root).query("split == 'train'").trip_id
    folds = make_folds(root)
    assert len(trips) == 80 and set(trips) == set(folds.trip_id)
    def read(kind):
        return pd.concat([pd.read_parquet(root/"data"/kind/f"{tid}.parquet") for tid in trips],
                         ignore_index=True)
    features, targets = read("features"), read("labels")
    assigned = targets[targets.label_status.eq("assigned")][["trip_id","obs_id","target_arc_id"]]
    merged = features.merge(assigned, on=["trip_id","obs_id"], validate="many_to_one")
    merged["fold_id"] = merged.trip_id.map(folds.set_index("trip_id").fold_id)
    groups = []
    for fold in range(5):
        fit, valid = merged[merged.fold_id.ne(fold)], merged[merged.fold_id.eq(fold)]
        expected = assigned[assigned.trip_id.isin(folds[folds.fold_id.eq(fold)].trip_id)]
        groups.append((fit[FEATURES], fit.candidate_arc_id.eq(fit.target_arc_id).astype(int),
                       valid[FEATURES], valid[KEY+["target_arc_id"]], len(expected)))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.workspace.resolve()
    out = root/"results/tuning/random_forest"
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    groups = prepare_search(root)
    workers = 5
    fixed = {"model__n_jobs": workers}
    search = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=3))
    def objective(trial):
        params = {
            "model__n_estimators":trial.suggest_int("model__n_estimators", 32, 96, step=16),
            "model__max_depth":trial.suggest_int("model__max_depth", 12, 26),
            "model__min_samples_leaf":trial.suggest_int("model__min_samples_leaf", 1, 12, log=True),
            "model__max_samples":trial.suggest_float("model__max_samples", .20, .80, step=.10),
        }
        scores = []
        for X,y,Xval,keys,denominator in groups:
            est = estimator_for("random_forest", {**fixed,**params})
            est.fit(X,y)
            val = keys.assign(p_match=positive_scores(est,Xval))
            best = top_predictions(val)
            scores.append(float(best.candidate_arc_id.eq(best.target_arc_id).sum()/denominator))
        trial.set_user_attr("fold_top1",scores)
        return float(np.mean(scores))
    def record(study,trial):
        study.trials_dataframe().to_csv(out/"trials.csv",index=False)
    search.optimize(objective,n_trials=10,callbacks=[record])
    search_seconds = time.perf_counter()-start
    best = search.best_trial
    spec = {"trial":best.number,"params":best.params,"objective":best.value,"fold_top1":best.user_attrs["fold_top1"],
            "trials":len(search.trials),"startup_trials":3,"search_seconds":search_seconds,"workers":workers,
            "sampler":"TPESampler","seed":42,"sklearn":sklearn.__version__,"optuna":optuna.__version__,
            "platform":platform.platform()}
    (out/"best.json").write_text(json.dumps(spec,indent=2))
    del groups
    # Reproduce the original hyperparameters on the same software and hardware.
    original = run_experiment(root,"random_forest",params=fixed,output_id="rf_untuned_control")
    tuned = run_experiment(root,"random_forest",params={**fixed,**best.params},output_id="final")
    if not np.isclose(best.value,tuned["cv_mean_top1"],atol=1e-6):
        raise ValueError("Refitted CV score differs from search objective")
    final = root/"artifacts/final"
    shutil.copytree(root/"artifacts/standard/final", final, dirs_exist_ok=True)
    def summarize(m):
        return dict(cv_mean_top1=m["cv_mean_top1"],cv_macro_f1=m["cv_pooled"]["candidate_macro_f1"],
                    student_test=m["student_test"],fit_seconds=m["fit_seconds"],inference_seconds=m["inference_seconds"],
                    data_hash=m["data_hash"])
    (out/"comparison.json").write_text(json.dumps({"untuned":summarize(original),"tuned":summarize(tuned),
        "total_seconds":time.perf_counter()-start},indent=2))


if __name__=="__main__":
    main()
