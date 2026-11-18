"""Build Sections 7 and 8 figures using the committed development results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parents[1] / "results"
MODEL = "random_forest_gps_context_tuned"
DECODERS = RESULTS / "decoder/experiments"
UNTUNED = RESULTS / "experiments/random_forest_gps_context"


def plot_trials():
    trials = pd.read_csv(RESULTS / "experiments/models" / MODEL / "trials.csv")
    untuned = json.loads((UNTUNED / "validation.json").read_text())["cv_mean_top1"]
    best = trials.loc[trials.value.idxmax()]
    fig, ax = plt.subplots(figsize=(8, 3.3), layout="constrained")
    ax.scatter(trials.number, 100 * trials.value, label="Trial", color="#3a6598")
    ax.scatter(best.number, 100 * best.value, marker="*", color="#a34611", s=150, label="Selected trial")
    ax.axhline(100 * untuned, linestyle="--", color="#52604a", label="Untuned GPS-context forest")
    ax.set(xlabel="Trial number", ylabel="Mean fold Top-1 (%)", xticks=trials.number)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "context_forest_trials.png", dpi=160)
    plt.close(fig)


def plot_routes():
    modes = [(DECODERS / MODEL / "classifier_only", "Classifier alone", "o", "#186785"),
             (DECODERS / MODEL / "hmm", "Strict decoder", "x", "#9a6619"),
             (DECODERS / (MODEL + "_segment_breaks") / "hmm", "Selected decoder", "D", "#5d3c92")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.2), sharey=True, layout="constrained")
    for offset, (folder, label, marker, color) in zip((-.19, 0, .19), modes):
        frame = pd.DataFrame(json.loads((folder / "student_test/metrics.json").read_text())["per_trip"]).sort_values("trip_id")
        y = np.arange(len(frame))
        for ax, key, title in zip(axes, ("pointwise_accuracy", "continuity"), ("Road-assignment Top-1 (%)", "Route continuity (%)")):
            ax.scatter(100 * frame[key], y + offset, color=color, marker=marker, label=label, s=23)
            ax.set_xlabel(title)
            ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(y, frame.trip_id)
    axes[0].set_ylabel("Student-test trip")
    axes[0].invert_yaxis()
    axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.savefig(OUT / "context_forest_routes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_trials()
    plot_routes()
