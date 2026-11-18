"""Rebuild the Section 8 per-trip table and figure from the saved decoder results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/decoder/baseline/random_forest"
KEY = ["trip_id", "obs_id"]


def trip_comparison():
    summary = json.loads((RESULTS / "comparison.json").read_text())
    joined = None
    for mode in ("classifier_only", "hmm"):
        frame = pd.read_parquet(RESULTS / mode / "student_test/predictions.parquet")
        predictions = frame[KEY + ["predicted_arc_id"]].rename(columns={"predicted_arc_id": mode})
        if joined is None:
            joined = predictions
        else:
            joined = joined.merge(predictions, on=KEY, how="outer", validate="one_to_one", indicator=True)
            assert joined["_merge"].eq("both").all()
    rows = []
    for trip_id, predictions in joined.groupby("trip_id"):
        labels = pd.read_parquet(ROOT / "data/labels" / f"{trip_id}.parquet")
        obs = pd.read_parquet(ROOT / "data/processed" / trip_id / "observations.parquet")
        keys = obs.loc[obs.valid, KEY].merge(labels[labels.label_status.eq("assigned")], on=KEY, validate="one_to_one")
        paired = keys.merge(predictions, on=KEY, validate="one_to_one")
        assert len(paired) == len(keys) and len(predictions) == len(obs)
        before = paired.classifier_only.eq(paired.target_arc_id).fillna(False)
        after = paired.hmm.eq(paired.target_arc_id).fillna(False)
        b, a = [next(m for m in summary[mode]["per_trip"] if m["trip_id"] == trip_id)
                for mode in ("classifier_only", "hmm")]
        assert (before.sum(), after.sum(), len(keys)) == (b["correct"], a["correct"], a["labelled"])
        assert b["eligible"] == a["eligible"] and b["labelled"] == a["labelled"]
        rows.append(dict(trip_id=trip_id, labelled=len(keys), classifier_correct=int(before.sum()),
                         hmm_correct=int(after.sum()), corrected=int((~before & after).sum()),
                         lost=int((before & ~after).sum()), classifier_accuracy=b["pointwise_accuracy"],
                         hmm_accuracy=a["pointwise_accuracy"], eligible_pairs=b["eligible"],
                         classifier_feasible=b["feasible"], hmm_feasible=a["feasible"],
                         classifier_continuity=b["continuity"], hmm_continuity=a["continuity"]))
    return pd.DataFrame(rows)


def main():
    table = trip_comparison()
    table.to_csv(RESULTS / "paired_trips.csv", index=False)
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, ["accuracy", "continuity"], ["Road-selection accuracy", "Route continuity"]):
        before = table["classifier_" + key] * 100
        after = table["hmm_" + key] * 100
        ax.hlines(y, np.minimum(before, after), np.maximum(before, after), color="#aab4b9", lw=1.5)
        ax.scatter(before, y, color="#2f6683", marker="o", label="Classifier only", zorder=3)
        ax.scatter(after, y, color="#a84e22", marker="D", label="HMM/Viterbi", zorder=4)
        ax.set(title=title, xlabel="Percent", yticks=y, yticklabels=table.trip_id)
        ax.grid(axis="x", alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8)
    axes[0].set_ylabel("Student-test trip")
    axes[0].invert_yaxis()
    fig.savefig(Path(__file__).parent / "untuned_forest_routes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
