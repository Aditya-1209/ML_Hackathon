from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


def build_estimator(params=None, seed=42):
    estimator = Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                          ("model",HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,
                                                                  early_stopping=False,random_state=seed))])
    return estimator.set_params(**(params or {}))
