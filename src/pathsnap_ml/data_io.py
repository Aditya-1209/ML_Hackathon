"""Read raw trip-local IDs and expose separate inference and offline reference loaders."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from .geometry import local_projection

SPLITS = ("train", "student_test")
KINDS = ("track", "nodes", "arcs", "route")


def discover(root):
    records = []
    for split in SPLITS:
        for folder in sorted((root / split).iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            trip = folder.name
            record = dict(trip_id=trip, split=split)
            for kind in KINDS:
                relative = Path(split, trip, f"{trip}.{kind}")
                if kind != "route" and not (root / relative).is_file():
                    raise FileNotFoundError(relative)
                record[f"{kind}_path"] = str(relative) if (root / relative).is_file() else ""
            records.append(record)
    return pd.DataFrame(records)


def check_trip_id(trip):
    if not trip or Path(trip).name != trip or trip in {".", ".."}:
        raise ValueError(f"Unsafe trip ID {trip!r}")


def raw_file(root, row, kind, *, allow_evaluator=False):
    permitted = (*SPLITS, "evaluator_test") if allow_evaluator else SPLITS
    if row.split not in permitted:
        raise ValueError("Development loaders accept only train and student_test")
    check_trip_id(row.trip_id)
    path = (root / getattr(row, f"{kind}_path")).resolve()
    if not path.is_relative_to((root / row.split).resolve()) or path.suffix != f".{kind}":
        raise ValueError(f"Unexpected {kind} path for {row.trip_id}")
    return path


def numeric_matrix(path, width, dtype):
    # Each physical line defines a node, arc or route-position ID.
    data = np.loadtxt(path, dtype=dtype, comments=None, ndmin=2)
    with path.open("rb") as stream:
        count = sum(1 for _ in stream)
    if not count or data.shape != (count, width):
        raise ValueError(f"{path}: expected {width} fields on every row")
    return data


def observations(path, trip, projection):
    rows = []
    with path.open() as source:
        for idx, line in enumerate(source):
            fields = line.split()
            reason = None
            lon, lat, ts = np.nan, np.nan, np.nan
            if len(fields) != 3:
                reason = "field_count"
            else:
                try:
                    lon, lat, ts = map(float, fields)
                    if not np.isfinite([lon, lat, ts]).all():
                        reason = "nonfinite"
                    elif abs(lon) > 180 or abs(lat) > 90:
                        reason = "coordinate_range"
                except ValueError:
                    reason = "nonnumeric"
            rows.append((trip, idx, ts, lon, lat, reason))
    df = pd.DataFrame(rows, columns=["trip_id","obs_id","timestamp_s","lon","lat","invalid_reason"])
    df = df.astype({"trip_id":"string","obs_id":"int64","timestamp_s":"float64","lon":"float64","lat":"float64","invalid_reason":"string"})
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df["valid"] = df.invalid_reason.isna()
    df["x_m"] = np.nan
    df["y_m"] = np.nan
    good = df.valid
    if good.any():
        x, y = projection.transform(df.loc[good,"lon"].to_numpy(), df.loc[good,"lat"].to_numpy(), errcheck=True)
        df.loc[good,"x_m"], df.loc[good,"y_m"] = x, y
    idx = df[good].sort_values(["timestamp_s","obs_id"]).index
    df["order_index"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[idx,"order_index"] = np.arange(len(idx))
    return df[["trip_id","obs_id","order_index","timestamp_s","lon","lat","x_m","y_m","valid","invalid_reason"]]


def prepare_trip(root, row, config, *, allow_evaluator=False):
    files = {k: raw_file(root, row, k, allow_evaluator=allow_evaluator) for k in ("track","nodes","arcs")}
    coords = numeric_matrix(files["nodes"], 2, float)
    lon, lat = coords.T
    if not np.isfinite(coords).all() or (abs(lon)>180).any() or (abs(lat)>90).any():
        raise ValueError(f"{row.trip_id}: invalid network coordinates")
    crs, projection = local_projection(lon,lat)
    x,y = projection.transform(lon,lat,errcheck=True)
    nodes = pd.DataFrame(dict(trip_id=row.trip_id,node_id=np.arange(len(x)),lon=lon,lat=lat,x_m=x,y_m=y))
    edges = numeric_matrix(files["arcs"], 2, np.int64)
    if (edges<0).any() or (edges>=len(nodes)).any():
        raise ValueError(f"{row.trip_id}: arc endpoint outside node table")
    u,v = edges.T
    arcs = pd.DataFrame(dict(trip_id=row.trip_id,arc_id=np.arange(len(u)),from_node_id=u,to_node_id=v,length_m=np.hypot(x[u]-x[v],y[u]-y[v])))
    obs = observations(files["track"],row.trip_id,projection)
    meta = dict(contract_version="pathsnap-ml-v1",trip_id=row.trip_id,split=row.split,crs_wkt=crs.to_wkt(),
                graph_directed=config["graph_directed"],directionality_basis=config["directionality_basis"],
                input_hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},cleaning_config=config)
    return dict(observations=obs,nodes=nodes,arcs=arcs,metadata=meta)


def read_raw_reference(root,row,arc_count, *, allow_evaluator=False):
    path = raw_file(root, row, "route", allow_evaluator=allow_evaluator)
    ids = numeric_matrix(path,1,np.int64).ravel()
    if (ids<0).any() or (ids>=arc_count).any():
        raise ValueError(f"{row.trip_id}: reference contains unknown arc ID")
    return pd.DataFrame(dict(trip_id=row.trip_id, route_pos=np.arange(len(ids)), arc_id=ids))


def write_table(df, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        name=tmp.name
    try:
        df.to_parquet(name,index=False)
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_processed_trip(processed_root: Path, trip_id: str) -> dict:
    check_trip_id(trip_id)
    folder=processed_root/trip_id
    result={name:pd.read_parquet(folder/f"{name}.parquet") for name in ("observations","nodes","arcs")}
    result["metadata"]=json.loads((folder/"metadata.json").read_text())
    return result


def load_development_reference(reference_root, row):
    if row.split not in SPLITS:
        raise ValueError("Development references only")
    check_trip_id(row.trip_id)
    return pd.read_parquet(reference_root/row.trip_id/"route.parquet")
