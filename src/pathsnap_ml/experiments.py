import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from .contracts import FEATURES, KEY, MODEL_IDS, fingerprint, manifest, validate_features
from .evaluation import evaluate


def make_folds(workspace):
    trips = manifest(workspace).query("split == 'train'").trip_id.sort_values().to_numpy()
    if len(trips) != 80:
        raise ValueError("The supplied development manifest must contain 80 training trips")
    frame = pd.DataFrame({"trip_id":trips,"fold_id":-1})
    for i, (_, held) in enumerate(GroupKFold(5).split(trips,groups=trips)):
        frame.loc[held,"fold_id"] = i
    out = workspace/"data/folds.csv"
    if out.exists():
        old = pd.read_csv(out,dtype={"trip_id":str})
        if not old.equals(frame):
            raise ValueError("Existing folds differ from the deterministic shared assignment")
    frame.to_csv(out,index=False)
    return frame


def load_data(workspace, smoke=False):
    rows = manifest(workspace)
    if smoke:
        rows = pd.concat([rows[rows.split.eq(s)].head(n) for s,n in [("train",10),("student_test",2)]])
    store = {}
    digest = hashlib.sha256()
    for kind in ["features","labels"]:
        frames = []
        for tid in rows.trip_id:
            path = workspace/"data"/kind/(tid+".parquet")
            digest.update(path.read_bytes())
            frames.append(pd.read_parquet(path))
        store[kind] = pd.concat(frames,ignore_index=True)
    validate_features(store["features"])
    store["observations"] = pd.concat([pd.read_parquet(workspace/"data/processed"/tid/"observations.parquet") for tid in rows.trip_id],ignore_index=True)
    digest.update((workspace/"data/features/schema.json").read_bytes())
    digest.update((workspace/"data/folds.csv").read_bytes())
    for tid in rows.trip_id:
        digest.update((workspace/"data/processed"/tid/"observations.parquet").read_bytes())
    store["data_hash"] = digest.hexdigest()
    store["manifest"] = rows
    store["schema"] = json.loads((workspace/"data/features/schema.json").read_text())
    return store


def estimator_for(model_id,params=None):
    if model_id not in MODEL_IDS:
        raise ValueError(model_id)
    module = importlib.import_module("pathsnap_ml.models."+model_id)
    return module.build_estimator(params=params,seed=42)


def positive_scores(estimator,features):
    positive = np.flatnonzero(estimator.classes_ == 1)
    if len(positive) != 1 or len(estimator.classes_) != 2:
        raise ValueError("Both classes must occur in every training fold")
    return estimator.predict_proba(features[FEATURES])[:,positive[0]]


def labelled(features, labels):
    frame = features.merge(labels[labels.label_status.eq("assigned")][["trip_id","obs_id","target_arc_id"]],
                           on=["trip_id","obs_id"],validate="many_to_one")
    y = frame.candidate_arc_id.eq(frame.target_arc_id).astype(int)
    if y.nunique() != 2:
        raise ValueError("Training fold needs both candidate classes")
    return frame[FEATURES], y


def score_table(estimator,features,model_id,fold=None):
    scores = features[KEY].copy()
    scores["p_match"] = positive_scores(estimator,features) if len(features) else np.array([],dtype=float)
    scores["model_id"] = model_id
    scores["fold_id"] = pd.array([fold]*len(scores),dtype="Int64")
    return scores


def cross_validate(workspace,model_id,data,params=None):
    folds = pd.read_csv(workspace/"data/folds.csv",dtype={"trip_id":str})
    train_ids = data["manifest"].query("split == 'train'").trip_id
    folds = folds[folds.trip_id.isin(train_ids)]
    all_scores, metrics = [], []
    for fold in range(5):
        held = folds[folds.fold_id.eq(fold)].trip_id
        fit_ids = folds[~folds.fold_id.eq(fold)].trip_id
        fit = data["features"][data["features"].trip_id.isin(fit_ids)]
        valid = data["features"][data["features"].trip_id.isin(held)]
        X,y = labelled(fit,data["labels"])
        estimator = estimator_for(model_id,params)
        started = time.perf_counter()
        estimator.fit(X,y)
        fitted = time.perf_counter()-started
        started = time.perf_counter()
        scores = score_table(estimator,valid,model_id,fold)
        seconds = time.perf_counter()-started
        metric = evaluate(scores,data["labels"],data["observations"][data["observations"].trip_id.isin(held)])
        metric.update(fold_id=fold,fit_seconds=fitted,inference_seconds=seconds)
        metrics.append(metric)
        all_scores.append(scores)
        print(model_id,"fold",fold,"top1",metric["top1_accuracy"],flush=True)
    return pd.concat(all_scores,ignore_index=True),metrics


def run_experiment(workspace: Path, model_id: str, smoke=False, params=None, output_id=None):
    workspace = Path(workspace)
    make_folds(workspace)
    data = load_data(workspace,smoke)
    profile = "smoke" if smoke else "standard"
    result_root = workspace/"results"/("smoke" if smoke else "baseline")
    output_id = output_id or model_id
    out = result_root/"models"/output_id
    out.mkdir(parents=True,exist_ok=True)
    completed = out/"metrics.json"
    if completed.exists():
        completed.unlink()
    scores, folds = cross_validate(workspace,model_id,data,params)
    scores["model_id"] = output_id
    train_ids = data["manifest"].query("split == 'train'").trip_id
    test_ids = data["manifest"].query("split == 'student_test'").trip_id
    train = data["features"][data["features"].trip_id.isin(train_ids)]
    test = data["features"][data["features"].trip_id.isin(test_ids)]
    estimator = estimator_for(model_id,params)
    X,y = labelled(train,data["labels"])
    started = time.perf_counter()
    estimator.fit(X,y)
    fit_seconds = time.perf_counter()-started
    started = time.perf_counter()
    test_scores = score_table(estimator,test,output_id)
    inference_seconds = time.perf_counter()-started
    scores.to_parquet(out/"cv_scores.parquet",index=False)
    test_scores.to_parquet(out/"student_test_scores.parquet",index=False)
    metrics = dict(model_id=output_id,estimator_id=model_id,profile=profile,data_hash=data["data_hash"],
                   accuracy_basis=data["schema"]["accuracy_basis"],folds=folds,
                   cv_mean_top1=float(np.mean([m["top1_accuracy"] for m in folds])),
                   cv_pooled=evaluate(scores,data["labels"],data["observations"][data["observations"].trip_id.isin(train_ids)]),
                   student_test=evaluate(test_scores,data["labels"],data["observations"][data["observations"].trip_id.isin(test_ids)]),
                   fit_seconds=fit_seconds,inference_seconds=inference_seconds,
                   machine=platform.platform(),python=sys.version,sklearn=sklearn.__version__,seed=42,
                   parameters=estimator.get_params(deep=False)["steps"][-1][1].get_params(),
                   feature_config_hash=data["schema"]["feature_config_hash"])
    meta = {k:metrics[k] for k in ["model_id","estimator_id","data_hash","feature_config_hash","profile","accuracy_basis","python","sklearn","seed"]}
    meta.update(feature_columns=FEATURES,positive_class=1,classes=estimator.classes_.tolist(),params=params or {})
    artifact = workspace/"artifacts"/profile/output_id
    artifact.mkdir(parents=True,exist_ok=True)
    joblib.dump(estimator,artifact/"model.joblib",compress=3)
    (artifact/"metadata.json").write_text(json.dumps(meta,indent=2))
    (artifact/"feature_config.json").write_text(json.dumps(data["schema"],indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    axes[0].bar(range(5), [m["top1_accuracy"] for m in folds])
    axes[0].set(xlabel="Held-out trip fold", ylabel="Observation Top-1", ylim=(0, 1))
    matrix = np.asarray(metrics["student_test"]["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[1].text(j, i, str(value), ha="center", va="center")
    axes[1].set(xlabel="Predicted class", ylabel="Reference-derived class",
                xticks=[0, 1], yticks=[0, 1], title="Student-test candidate confusion matrix")
    fig.savefig(out / "diagnostics.png", dpi=150)
    plt.close(fig)
    (out / "summary.txt").write_text(
        f"Model {output_id} | {profile} | reference-derived proxy labels\n"
        f"Mean-fold CV Top-1 {metrics['cv_mean_top1']:.6f}\n"
        + json.dumps(metrics["student_test"], indent=2))
    # This completion record is written last, so incomplete runs cannot be compared.
    completed.write_text(json.dumps(metrics,indent=2))
    return metrics
