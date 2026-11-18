"""Generate road candidates from projected observations and road geometry."""

import numpy as np
import pandas as pd
from shapely import STRtree, box, linestrings, points

KEY_COLUMNS = ["trip_id", "obs_id", "candidate_arc_id"]
CANDIDATE_DTYPES = {
    "trip_id": "string", "obs_id": "int64", "candidate_arc_id": "int64",
    "candidate_rank": "int64", "distance_m": "float64",
    "projected_x_m": "float64", "projected_y_m": "float64",
    "projection_fraction": "float64", "used_expanded_search": "bool",
}


def valid_observations(observations, metadata):
    if observations.obs_id.duplicated().any():
        raise ValueError("Duplicate observation IDs")
    if not observations.trip_id.eq(metadata["trip_id"]).all():
        raise ValueError("Observation trip ID mismatch")
    valid = observations.loc[observations.valid].sort_values("order_index")
    if valid.order_index.tolist() != list(range(len(valid))):
        raise ValueError("Invalid observation order_index")
    if not np.isfinite(valid[["x_m", "y_m", "timestamp_s"]]).all().all():
        raise ValueError("Valid observations require finite projected coordinates and time")
    if valid.obs_id.tolist() != valid.sort_values(["timestamp_s", "obs_id"]).obs_id.tolist():
        raise ValueError("Observation order disagrees with timestamp and original row index")
    if observations.loc[~observations.valid, "order_index"].notna().any():
        raise ValueError("Invalid observations must have no order_index")
    return valid


def segment_geometry(nodes, arcs, metadata):
    if not isinstance(metadata["graph_directed"], bool):
        raise ValueError("Expected saved graph directionality")
    for frame, key in ((nodes, "node_id"), (arcs, "arc_id")):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Missing or duplicate {key}")
        if not frame.trip_id.eq(metadata["trip_id"]).all():
            raise ValueError("Geometry trip ID mismatch")
    xy = nodes.set_index("node_id")[["x_m", "y_m"]]
    if not np.isfinite(xy).all().all():
        raise ValueError("Nonfinite node coordinates")
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy().ravel()
    if not np.isin(endpoints, xy.index).all():
        raise ValueError("Arc references a missing endpoint")
    return (
        xy.loc[arcs.from_node_id].to_numpy(dtype=float),
        xy.loc[arcs.to_node_id].to_numpy(dtype=float),
    )


def project(point, starts, ends):
    vectors = ends - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    numerator = np.einsum("ij,ij->i", point - starts, vectors)
    fractions = np.divide(
        numerator, squared_lengths, out=np.zeros_like(numerator), where=squared_lengths > 0
    ).clip(0, 1)
    projected = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(projected - point, axis=1)
    return distances, projected, fractions


def make_candidates(observations, nodes, arcs, metadata, config):
    radius = config["radius_m"]
    fallback = config["fallback_radius_m"]
    limit = config["max_candidates"]
    if not np.isfinite([radius, fallback]).all() or not 0 < radius <= fallback:
        raise ValueError("Expected 0 < radius_m <= fallback_radius_m")
    if type(limit) is not int or limit < 1:
        raise ValueError("max_candidates must be a positive integer")
    valid = valid_observations(observations, metadata)
    starts, ends = segment_geometry(nodes, arcs, metadata)
    arc_ids = arcs.arc_id.to_numpy()
    geometry = linestrings(np.stack([starts, ends], axis=1))
    zero_length = np.all(starts == ends, axis=1)
    geometry[zero_length] = points(starts[zero_length])
    tree = STRtree(geometry)
    rows = []
    for observation in valid.itertuples(index=False):
        point = np.array([observation.x_m, observation.y_m])
        for expanded, distance_limit in ((False, radius), (True, fallback)):
            x, y = point
            indices = tree.query(box(x - distance_limit, y - distance_limit,
                                     x + distance_limit, y + distance_limit))
            distances, projected, fractions = project(point, starts[indices], ends[indices])
            ordered = np.lexsort((arc_ids[indices], distances))
            selected = ordered[distances[ordered] <= distance_limit][:limit]
            if len(selected):
                for rank, index in enumerate(selected):
                    rows.append((metadata["trip_id"], observation.obs_id, arc_ids[indices[index]],
                                 rank, distances[index], *projected[index],
                                 fractions[index], expanded))
                break
    return pd.DataFrame(rows, columns=list(CANDIDATE_DTYPES)).astype(CANDIDATE_DTYPES)

