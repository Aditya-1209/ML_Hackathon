"""Unfitted Random Forest pipeline for the shared experiment runner."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_estimator(params: dict | None = None, seed: int = 42) -> Pipeline:
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        )),
    ])
    return estimator.set_params(**(params or {}))

