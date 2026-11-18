"""Run the candidate/feature and offline label stages separately."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re

import pandas as pd
import shapely
import yaml

from .candidates import KEY_COLUMNS, make_candidates
from .features import FEATURE_COLUMNS, make_features
from .reference_labels import align_reference, coverage_summary, write_alignment_svg


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return config, hashlib.sha256(payload.encode()).hexdigest()


def _save(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def run(workspace, mode):
    from .data_io import load_processed_trip

    manifest = workspace / "data/manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["trip_id"] for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty manifest with unique trip IDs")
    if any(row["split"] not in {"train", "student_test", "evaluator_test"} for row in rows):
        raise ValueError("Unknown manifest partition")
    rows = [row for row in rows if row["split"] in {"train", "student_test"}]
    if not rows:
        raise ValueError("Manifest contains no development trips")
    filename = "features.yaml" if mode == "build" else "reference_alignment.yaml"
    config, config_hash = _config(workspace / "configs" / filename)
    processed = workspace / "data/processed"
    summaries = []
    mapped_splits = set()
    for row in rows:
        trip_id = row["trip_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
            raise ValueError(f"Invalid trip ID: {trip_id!r}")
        trip = load_processed_trip(processed, trip_id)
        metadata = trip["metadata"]
        if metadata["trip_id"] != trip_id or metadata["split"] != row["split"]:
            raise ValueError("Processed trip disagrees with manifest")
        candidate_path = workspace / "data/candidates" / f"{trip_id}.parquet"
        if mode == "build":
            candidates = make_candidates(**trip, config=config)
            features = make_features(**trip, candidates=candidates, config=config)
            _save(candidate_path, candidates)
            _save(workspace / "data/features" / candidate_path.name, features)
            labels = None
        else:
            candidates = pd.read_parquet(candidate_path)
            reference_path = workspace / "data/reference" / trip_id / "route.parquet"
            reference = pd.read_parquet(reference_path) if reference_path.is_file() else None
            labels = align_reference(**trip, reference=reference, config=config)
            _save(workspace / "data/labels" / candidate_path.name, labels)
            if reference is not None and not reference.empty and row["split"] not in mapped_splits:
                path = workspace / f"results/features/{row['split']}_{trip_id}_alignment.svg"
                write_alignment_svg(path, **trip, reference=reference, labels=labels)
                mapped_splits.add(row["split"])
        summary = coverage_summary(trip["observations"], candidates, labels)
        summary.update(trip_id=trip_id, split=row["split"], input_hashes={
            key: value for key, value in metadata["input_hashes"].items()
            if mode == "labels" or "route" not in key.lower()
        })
        if mode == "build":
            summary["feature_distributions"] = {
                name: {
                    "missing": int(features[name].isna().sum()),
                    "median": float(features[name].median()) if features[name].notna().any() else None,
                } for name in FEATURE_COLUMNS
            }
        summaries.append(summary)
    provenance = {
        "contract_version": "pathsnap-ml-v1", "config": config, "config_hash": config_hash,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("candidates.py", "features.py", "reference_labels.py", "cli_features.py")
        },
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "shapely": shapely.__version__},
    }
    if mode == "build":
        _json(workspace / "data/features/schema.json", {
            **provenance, "feature_columns": list(FEATURE_COLUMNS), "key_columns": KEY_COLUMNS,
        })
    _json(workspace / f"results/features/{mode}_summary.json", {**provenance, "trips": summaries})
    print(f"{mode}: processed {len(summaries)} development trips")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "labels"])
    parser.add_argument("--workspace", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.workspace.resolve(), args.mode)


if __name__ == "__main__":
    main()
