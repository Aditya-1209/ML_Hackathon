"""Decision Tree classifier pipeline for candidate road segment matching.

Adheres to PathSnap ML model contract:
- Returns an unfitted sklearn Pipeline with steps [('impute', ...), ('model', ...)].
- Step 'model' is a DecisionTreeClassifier with reproducible seed.
- Handles missing values and entirely empty columns via SimpleImputer(keep_empty_features=True).
- Supports predict_proba and extracts positive class probabilities using estimator.classes_.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def build_estimator(params: dict | None = None, seed: int = 42) -> Pipeline:
    """Build an unfitted DecisionTreeClassifier pipeline.

    Args:
        params: Optional dictionary of hyperparameter overrides (e.g. {'model__max_depth': 10}).
        seed: Random seed for reproducibility.

    Returns:
        Unfitted sklearn Pipeline with steps named 'impute' and 'model'.
    """
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", DecisionTreeClassifier(random_state=seed)),
    ])
    return pipeline.set_params(**(params or {}))


def get_positive_class_index(estimator: Pipeline | DecisionTreeClassifier) -> int:
    """Identify the column index of the positive class (label 1) in predict_proba output.

    Args:
        estimator: Fitted Pipeline or DecisionTreeClassifier.

    Returns:
        Index of positive class in classes_.
    """
    model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
    if not hasattr(model, "classes_"):
        raise ValueError("Estimator has not been fitted yet; classes_ attribute is missing.")
    classes = list(model.classes_)
    if 1 in classes:
        return classes.index(1)
    if True in classes:
        return classes.index(True)
    if len(classes) == 2:
        return 1
    return 0


def predict_match_probabilities(estimator: Pipeline, X: Any) -> np.ndarray:
    """Safely obtain match (positive class) probabilities for candidate rows.

    Args:
        estimator: Fitted Pipeline.
        X: Feature matrix or DataFrame.

    Returns:
        1D array of positive class match probabilities in [0.0, 1.0].
    """
    proba = estimator.predict_proba(X)
    pos_idx = get_positive_class_index(estimator)
    return proba[:, pos_idx]
