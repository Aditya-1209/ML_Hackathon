from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_estimator(params=None, seed=42):
    estimator = LGBMClassifier(
        objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=seed, n_jobs=2,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    pipeline = Pipeline([("impute", imputer.set_output(transform="pandas")), ("model", estimator)])
    return pipeline.set_params(**(params or {}))
