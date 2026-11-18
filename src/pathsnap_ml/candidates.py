import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from .contracts import KEY


def segment_geometries(nodes, arcs):
    xy = nodes.set_index("node_id")[["x_m", "y_m"]]
    return shapely.linestrings(np.stack((xy.loc[arcs.from_node_id].to_numpy(),
                                        xy.loc[arcs.to_node_id].to_numpy()), axis=1))


def make_candidates(observations, nodes, arcs, metadata, config):
    geometry = segment_geometries(nodes, arcs)
    tree = STRtree(geometry)
    valid = observations[observations.valid].sort_values("order_index")
    records = []
    for obs in valid.itertuples():
        point = shapely.Point(obs.x_m, obs.y_m)
        indices = tree.query(point, predicate="dwithin", distance=config["radius_m"])
        expanded = len(indices) == 0
        if expanded:
            indices = tree.query(point, predicate="dwithin", distance=config["fallback_radius_m"])
        if not len(indices):
            continue
        distances = shapely.distance(geometry[indices], point)
        ids = arcs.arc_id.to_numpy()[indices]
        order = np.lexsort((ids, distances))[:config["max_candidates"]]
        for rank, j in enumerate(order):
            idx = indices[j]
            line = geometry[idx]
            length = float(shapely.length(line))
            fraction = float(shapely.line_locate_point(line, point) / length) if length else 0.
            projected = shapely.line_interpolate_point(line, fraction * length)
            records.append((obs.trip_id, obs.obs_id, int(arcs.iloc[idx].arc_id), rank,
                            float(distances[j]), projected.x, projected.y, fraction, expanded))
    cols = KEY + ["candidate_rank", "distance_m", "projected_x_m", "projected_y_m",
                  "projection_fraction", "used_expanded_search"]
    return pd.DataFrame(records, columns=cols).astype({
        "trip_id": "string", "obs_id": "int64", "candidate_arc_id": "int64",
        "candidate_rank": "int64", "distance_m": "float64", "projected_x_m": "float64",
        "projected_y_m": "float64", "projection_fraction": "float64", "used_expanded_search": "bool"})
