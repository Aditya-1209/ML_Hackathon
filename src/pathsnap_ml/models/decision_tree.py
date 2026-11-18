from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


def build_estimator(params=None, seed=42):
    estimator = Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                          ("model",DecisionTreeClassifier(max_depth=12,min_samples_leaf=5,random_state=seed))])
    return estimator.set_params(**(params or {}))
