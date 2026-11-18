import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .contracts import FEATURES, fingerprint, manifest
from .data_io import load_processed_trip, write_table
from .candidates import make_candidates
from .features import make_features
from .reference_labels import align_reference


def build(workspace, selected_trip=None):
    config = yaml.safe_load((workspace/"configs/features.yaml").read_text())
    align = yaml.safe_load((workspace/"configs/reference_alignment.yaml").read_text())
    summaries = []
    for row in manifest(workspace).itertuples():
        if selected_trip and selected_trip != row.trip_id:
            continue
        trip = load_processed_trip(workspace/"data/processed", row.trip_id)
        obs, nodes, arcs, meta = (trip[k] for k in ("observations","nodes","arcs","metadata"))
        candidates = make_candidates(obs, nodes, arcs, meta, config)
        features = make_features(obs, nodes, arcs, candidates, meta, config)
        reference = pd.read_parquet(workspace/"data/reference"/row.trip_id/"route.parquet")
        labels = align_reference(obs, nodes, arcs, reference, align)
        for folder, frame in [("candidates",candidates),("features",features),("labels",labels)]:
            write_table(frame, workspace/"data"/folder/(row.trip_id+".parquet"))
        assigned = labels[labels.label_status.eq("assigned")]
        hit = assigned.merge(candidates[["trip_id","obs_id","candidate_arc_id"]],
                             left_on=["trip_id","obs_id","target_arc_id"],
                             right_on=["trip_id","obs_id","candidate_arc_id"])
        summaries.append(dict(trip_id=row.trip_id,split=row.split,observations=len(obs),candidate_rows=len(candidates),
                              assigned=len(assigned),ambiguous=int(labels.label_status.eq("ambiguous").sum()),
                              misses=len(assigned)-len(hit),no_candidates=int((~obs.obs_id.isin(candidates.obs_id)&obs.valid).sum())))
        print(row.trip_id, len(candidates), len(assigned), flush=True)
    schema = {"contract_version":"pathsnap-ml-v1","feature_columns":FEATURES,"config":config,
              "feature_config_hash":fingerprint(config),"alignment_config":align,
              "accuracy_basis":"reference-derived proxy; ordered-route alignment, not official observation ground truth"}
    (workspace/"data/features/schema.json").write_text(json.dumps(schema,indent=2))
    out = workspace/"results/features"
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/"coverage.csv",index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace",type=Path,default=Path("."))
    p.add_argument("--trip")
    args = p.parse_args()
    build(args.workspace.resolve(),args.trip)


if __name__ == "__main__":
    main()
