import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from . import data_io, eda


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare"])
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--raw-root",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,default=Path("data/manifest.csv"))
    parser.add_argument("--include-reference",action="store_true")
    parser.add_argument("--trip",help="Process a single development trip for the first handoff")
    args=parser.parse_args()
    base=args.workspace.resolve()
    raw=args.raw_root.resolve()
    manifest=base/args.manifest
    if manifest.exists():
        frame=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    else:
        frame=data_io.discover(raw)
        manifest.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(manifest,index=False)
    if frame.trip_id.duplicated().any() or not set(frame.split)<=set((*data_io.SPLITS,"evaluator_test")):
        raise ValueError("Invalid manifest IDs or partition names")
    dev=frame[frame.split.isin(data_io.SPLITS)]
    if args.trip:
        dev=dev[dev.trip_id==args.trip]
        if len(dev)!=1:
            parser.error("Trip must belong to the development partitions")
    config=yaml.safe_load((base/"configs/data.yaml").read_text())
    results=base/"results/eda"
    results.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for i,row in enumerate(dev.itertuples(index=False)):
        trip=data_io.prepare_trip(raw,row,config)
        ref=None
        if args.include_reference and row.route_path:
            ref=data_io.read_raw_reference(raw,row,len(trip["arcs"]))
            data_io.write_table(ref,base/"data/reference"/row.trip_id/"route.parquet")
        folder=base/"data/processed"/row.trip_id
        for key in ("observations","arcs","nodes"):
            data_io.write_table(trip[key],folder/f"{key}.parquet")
        (folder/"metadata.json").write_text(json.dumps(trip["metadata"],indent=2))
        summary=eda.summarize(trip,ref,config["gap_threshold_s"])
        summary["reference_available"]=bool(row.route_path)
        for kind in ("track","nodes","arcs"):
            summary[f"{kind}_bytes"]=data_io.raw_file(raw,row,kind).stat().st_size
        summaries.append(summary)
        if i==0:
            eda.sample_map(trip,results/f"{row.split}-{row.trip_id}.html")
    stats=pd.DataFrame(summaries)
    stem=f"trip-{args.trip}" if args.trip else "development"
    stats.to_csv(results/f"{stem}.csv",index=False)
    if not args.trip:
        eda.plot_inventory(stats,results/"development.png")
    print(f"Processed {len(stats)} trips, {stats.observations.sum()} observations")


if __name__=="__main__":
    main()
