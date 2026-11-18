from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(params: dict | None = None, seed: int = 42) -> Pipeline:
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
    ])
    return pipeline.set_params(**(params or {}))
