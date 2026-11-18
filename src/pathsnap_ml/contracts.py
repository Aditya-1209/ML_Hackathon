import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "pathsnap-ml-v1"
KEY = ["trip_id", "obs_id", "candidate_arc_id"]
FEATURES = ["distance_m", "heading_difference_deg", "arc_length_m",
            "projection_fraction", "delta_t_s", "gps_step_m", "speed_mps", "heading_available"]
MODEL_IDS = ["logistic_regression", "knn", "random_forest", "hist_gradient_boosting", "decision_tree", "lightgbm"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manifest(workspace):
    result = pd.read_csv(Path(workspace) / "data/manifest.csv", dtype=str, keep_default_na=False)
    if result.trip_id.duplicated().any():
        raise ValueError("Duplicate manifest trip IDs")
    if not set(result.split) <= {"train", "student_test", "evaluator_test"}:
        raise ValueError("Unknown split")
    return result[result.split.isin(["train", "student_test"])].copy()


def validate_features(frame):
    if list(frame.columns) != KEY + FEATURES:
        raise ValueError("Unexpected feature order")
    if frame.duplicated(KEY).any() or np.isinf(frame[FEATURES].to_numpy()).any():
        raise ValueError("Duplicate keys or infinite features")


def score_keys(candidates, scores):
    if scores.duplicated(KEY).any():
        raise ValueError("Duplicate score keys")
    check = candidates[KEY].merge(scores[KEY], on=KEY, how="outer", indicator=True)
    if not check["_merge"].eq("both").all():
        raise ValueError("Scores and candidates have different keys")
    if not np.isfinite(scores.p_match).all() or not scores.p_match.between(0, 1).all():
        raise ValueError("Invalid probability")
