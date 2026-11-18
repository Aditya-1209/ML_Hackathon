import json
import shutil
from pathlib import Path

import numpy as np
import optuna

from .experiments import load_data,make_folds,cross_validate,run_experiment


def search_params(trial,model):
    if model=="logistic_regression":
        return {"model__C":trial.suggest_float("model__C",.01,100,log=True)}
    if model=="lightgbm":
        return {"model__learning_rate": trial.suggest_float("model__learning_rate", .02, .15, log=True),
                "model__num_leaves": trial.suggest_int("model__num_leaves", 15, 63),
                "model__min_child_samples": trial.suggest_int("model__min_child_samples", 15, 100)}
    if model=="knn":
        return {"model__n_neighbors":trial.suggest_int("model__n_neighbors",3,25),
                "model__weights":trial.suggest_categorical("model__weights",["uniform","distance"])}
    if model in ("decision_tree","random_forest"):
        result={"model__max_depth":trial.suggest_int("model__max_depth",6,24),
                "model__min_samples_leaf":trial.suggest_int("model__min_samples_leaf",2,20)}
        if model=="random_forest":
            result["model__n_estimators"]=trial.suggest_int("model__n_estimators",80,200,step=40)
        return result
    return {"model__learning_rate":trial.suggest_float("model__learning_rate",.02,.2,log=True),
            "model__max_leaf_nodes":trial.suggest_int("model__max_leaf_nodes",10,40),
            "model__l2_regularization":trial.suggest_float("model__l2_regularization",1e-4,10,log=True)}


def tune(workspace,trials=30):
    selection=json.loads((workspace/"results/baseline/selection.json").read_text())
    if selection.get("provisional"):
        raise ValueError("Run complete baseline experiments before final tuning")
    model=selection["model_id"]
    make_folds(workspace)
    data=load_data(workspace)
    if selection["data_hash"]!=data["data_hash"]:
        raise ValueError("Selection and prepared data do not match")
    out=workspace/"results/tuning"
    out.mkdir(parents=True,exist_ok=True)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42),
                              storage="sqlite:///"+str((out/"study.sqlite3").resolve()),
                              study_name=model+"-"+data["data_hash"][:16],load_if_exists=True)
    def objective(trial):
        params=search_params(trial,model)
        _,folds=cross_validate(workspace,model,data,params)
        return float(np.mean([m["top1_accuracy"] for m in folds]))
    try:
        study.optimize(objective,n_trials=trials)
    finally:
        study.trials_dataframe().to_csv(out/"trials.csv",index=False)
    run_experiment(workspace,model,params=study.best_params,output_id="final")
    final=workspace/"artifacts/final"
    shutil.copytree(workspace/"artifacts/standard/final",final,dirs_exist_ok=True)
    (out/"best.json").write_text(json.dumps({"model_id":model,"params":study.best_params,
        "best_cv_objective":study.best_value,"note":"Selected-configuration development CV; final assessment requires reserved evaluator trips."},indent=2))
    baseline=json.loads((workspace/"results/baseline/models"/model/"metrics.json").read_text())
    tuned=json.loads((workspace/"results/baseline/models/final/metrics.json").read_text())
    (out/"tuned_vs_untuned.json").write_text(json.dumps({"untuned":baseline,"tuned":tuned},indent=2))
