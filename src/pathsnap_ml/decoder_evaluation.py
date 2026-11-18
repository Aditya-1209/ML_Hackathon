import json
from pathlib import Path

import pandas as pd
import yaml

from .contracts import manifest
from .data_io import load_processed_trip
from .decoder import decode_trip
from .route_metrics import compute_continuity


def evaluate_decoder(workspace,model_id,profile="baseline",mode="hmm",config_override=None,split="student_test"):
    scores = pd.read_parquet(workspace/"results"/profile/"models"/model_id/("cv_scores.parquet" if split=="train" else "student_test_scores.parquet"))
    config = config_override or yaml.safe_load((workspace/"configs/decoder.yaml").read_text())
    outputs,routes,trip_metrics = [],[],[]
    for tid in manifest(workspace).query("split == @split").trip_id:
        if tid not in set(scores.trip_id):
            continue
        trip = load_processed_trip(workspace/"data/processed",tid)
        obs,nodes,arcs,meta = (trip[k] for k in ["observations","nodes","arcs","metadata"])
        candidates = pd.read_parquet(workspace/"data/candidates"/(tid+".parquet"))
        predictions,route = decode_trip(obs,nodes,arcs,candidates,scores[scores.trip_id.eq(tid)],meta,config,mode)
        continuity = compute_continuity(obs,nodes,arcs,candidates,predictions,meta,config)
        labels = pd.read_parquet(workspace/"data/labels"/(tid+".parquet"))
        eligible = obs[obs.valid][["trip_id","obs_id"]].merge(labels.query("label_status == 'assigned'"),on=["trip_id","obs_id"])
        checked = eligible.merge(predictions,on=["trip_id","obs_id"],validate="one_to_one")
        correct = checked.target_arc_id.eq(checked.predicted_arc_id).fillna(False)
        trip_metrics.append(dict(trip_id=tid,correct=int(correct.sum()),labelled=len(correct),
                                 pointwise_accuracy=float(correct.mean()) if len(correct) else None,**continuity))
        outputs.append(predictions); routes.append(route)
    if not outputs:
        raise ValueError("No matching decoder evaluation scores")
    feasible=sum(m["feasible"] for m in trip_metrics)
    eligible=sum(m["eligible"] for m in trip_metrics)
    labelled=sum(m["labelled"] for m in trip_metrics)
    evaluable=[m for m in trip_metrics if m["labelled"]]
    summary=dict(model_id=model_id,profile=profile,mode=mode,split=split,config=config,
                 accuracy_basis="reference-derived proxy; coverage restricted to assigned observations",
                 labelled=labelled,pointwise_accuracy=sum(m["correct"] for m in trip_metrics)/max(1,labelled),
                 mean_trip_accuracy=sum(m["pointwise_accuracy"] for m in evaluable)/max(1,len(evaluable)),
                 exact_trip_match_rate=sum(m["correct"]==m["labelled"] for m in evaluable)/max(1,len(evaluable)),
                 continuity=feasible/eligible if eligible else None,feasible=feasible,eligible=eligible,per_trip=trip_metrics)
    out=workspace/"results/decoder"/profile/model_id/mode/split
    out.mkdir(parents=True,exist_ok=True)
    pd.concat(outputs,ignore_index=True).to_parquet(out/"predictions.parquet",index=False)
    (out/"routes.json").write_text(json.dumps({"contract_version":"pathsnap-ml-v1","trips":routes},indent=2))
    (out/"metrics.json").write_text(json.dumps(summary,indent=2))
    return summary
