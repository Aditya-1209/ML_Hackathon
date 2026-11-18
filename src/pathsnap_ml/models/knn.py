from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(params: dict | None = None, seed: int = 42) -> Pipeline:
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2)),
    ])
    return pipeline.set_params(**(params or {}))
