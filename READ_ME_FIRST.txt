Person 3 | Fold creation and model registration
Use pandas.read_csv("data/manifest.csv", dtype={"trip_id": str}). Build common trip-grouped folds from the 80 train rows. Reserve the 10 student_test rows for development comparisons. You can build folds from this manifest alone.
Use the included model builders for logistic_regression and knn. Each exposes build_estimator(params=None, seed=42) and returns an unfitted scikit-learn Pipeline with a final step named model. With requirements installed and PYTHONPATH=src, import from pathsnap_ml.models.logistic_regression or pathsnap_ml.models.knn.
Send the shared data/folds.csv and training runner command/version to the model owners. Training also requires Person 2's features and labels.

Person 1 source revision: 7c5733b. All paths below data/ are relative to the extracted folder.
