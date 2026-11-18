import json
from pathlib import Path
import joblib
from .contracts import FEATURES, KEY, validate_features
from .experiments import score_table


def predict_candidates(bundle_dir: Path, features, feature_config_hash):
    validate_features(features)
    meta = json.loads((bundle_dir/"metadata.json").read_text())
    if meta["feature_columns"] != FEATURES or meta["feature_config_hash"] != feature_config_hash:
        raise ValueError("Inference feature schema/configuration does not match the fitted model")
    estimator = joblib.load(bundle_dir/"model.joblib")
    return score_table(estimator,features,meta["model_id"])
