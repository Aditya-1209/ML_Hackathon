"""From the repository root: python report_sections/figures/plot_model_comparison.py"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap_ml.data_io import load_processed_trip
from pathsnap_ml.evaluation import top_predictions
from pathsnap_ml.route_metrics import compute_continuity

NAMES = {"random_forest": "Random Forest", "hist_gradient_boosting": "Histogram boosting",
         "decision_tree": "Decision Tree", "logistic_regression": "Logistic Regression", "knn": "KNN"}
OUT = Path(__file__).parent


def summarize(model):
    folder = ROOT / "results/baseline/models" / model
    metrics = json.loads((folder / "metrics.json").read_text())
    scores = pd.read_parquet(folder / "student_test_scores.parquet")
    choices = top_predictions(scores).rename(columns={"candidate_arc_id": "predicted_arc_id"})
    config = yaml.safe_load((ROOT / "configs/decoder.yaml").read_text())
    feasible = eligible = 0
    for trip_id, group in choices.groupby("trip_id"):
        trip = load_processed_trip(ROOT / "data/processed", trip_id)
        candidates = pd.read_parquet(ROOT / "data/candidates" / f"{trip_id}.parquet")
        counts = compute_continuity(trip["observations"], trip["nodes"], trip["arcs"],
                                    candidates, group, trip["metadata"], config)
        feasible += counts["feasible"]
        eligible += counts["eligible"]
    return metrics, feasible, eligible


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    records = []
    names = []
    cv, student = [], []
    for model, name in NAMES.items():
        m, feasible, eligible = summarize(model)
        folds = np.array([fold["top1_accuracy"] for fold in m["folds"]])
        axes[0].plot(range(5), folds * 100, marker="o", markersize=4, label=name)
        names.append(name)
        cv.append(100 * folds.mean())
        student.append(100 * m["student_test"]["top1_accuracy"])
        records.append(dict(model=model, feasible_pairs=feasible, eligible_pairs=eligible,
                            classifier_continuity=feasible / eligible))
    axes[0].set(xlabel="Held-out trip fold", ylabel="Observation Top-1 accuracy (%)",
                title="Performance varies by trip fold", xticks=range(5))
    axes[0].legend(fontsize=8, ncol=2)
    positions = np.arange(len(names))
    axes[1].scatter(cv, positions - .1, label="Mean grouped CV", color="#235789")
    axes[1].scatter(student, positions + .1, label="Student test", marker="D", color="#b64b22")
    axes[1].set(xlabel="Observation Top-1 accuracy (%)", title="Development comparison",
                yticks=positions, yticklabels=names, ylim=(len(names)-.45, -.65))
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.22)
    fig.savefig(OUT / "model_comparison.png", dpi=170)
    pd.DataFrame(records).to_csv(OUT / "classifier_continuity.csv", index=False)


if __name__ == "__main__":
    main()
