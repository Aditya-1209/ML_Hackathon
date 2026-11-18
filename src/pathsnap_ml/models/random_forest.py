from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def build_estimator(params=None, seed=42):
    estimator = Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                          ("model",RandomForestClassifier(n_estimators=100,max_depth=16,
                                                          min_samples_leaf=3,n_jobs=2,random_state=seed))])
    return estimator.set_params(**(params or {}))
