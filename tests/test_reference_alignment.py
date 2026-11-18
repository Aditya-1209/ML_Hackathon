import numpy as np
import pandas as pd

from pathsnap_ml.candidates import CANDIDATE_DTYPES
from pathsnap_ml.reference_labels import align_reference, coverage_summary


CONFIG = dict(max_reference_distance_m=2, ambiguity_cost_margin_m=0.5,
              alignment_gap_s=30, max_alignment_cells=1000)


def fixture(points):
    coordinates = np.array(points, dtype=float)
    size = len(coordinates)
    observations = pd.DataFrame({
        "trip_id": ["0001"] * size, "obs_id": range(size),
        "order_index": range(size), "timestamp_s": range(size),
        "x_m": coordinates[:, 0], "y_m": coordinates[:, 1],
        "valid": np.isfinite(coordinates).all(axis=1),
    })
    observations["order_index"] = observations.order_index.where(observations.valid).astype("Int64")
    nodes = pd.DataFrame({
        "trip_id": ["0001"] * 4, "node_id": range(4),
        "x_m": [-10., 10., 0., 0.], "y_m": [0., 0., -10., 10.],
    })
    arcs = pd.DataFrame({
        "trip_id": ["0001"] * 2, "arc_id": [40, 99],
        "from_node_id": [0, 2], "to_node_id": [1, 3], "length_m": [20., 20.],
    })
    return dict(observations=observations, nodes=nodes, arcs=arcs,
                metadata={"trip_id": "0001", "graph_directed": True, "split": "train"})


def route(arc_ids):
    return pd.DataFrame({"trip_id": "0001", "route_pos": range(len(arc_ids)), "arc_id": arc_ids})


def test_repeated_arcs_keep_distinct_positions_and_far_rows_are_unavailable():
    trip = fixture([(-8, 0), (-5, 0), (0, 8), (8, 0), (40, 40), (np.nan, np.nan)])
    labels = align_reference(**trip, reference=route([40, 99, 40]), config=CONFIG)
    assert labels.target_route_pos.iloc[:4].tolist() == [0, 0, 1, 2]
    assert labels.target_arc_id.iloc[:4].tolist() == [40, 40, 99, 40]
    assert labels.label_status.tolist() == ["assigned"] * 4 + ["unavailable"] * 2


def test_unresolved_crossing_is_ambiguous():
    trip = fixture([(-8, 0), (0, 0), (0, 8)])
    labels = align_reference(**trip, reference=route([40, 99]), config=CONFIG)
    assert labels.label_status.tolist() == ["assigned", "ambiguous", "assigned"]
    assert pd.isna(labels.target_arc_id.iloc[1])


def test_missing_target_stays_a_miss_including_no_candidate_observation():
    trip = fixture([(-8, 0), (0, 8), (40, 40)])
    labels = align_reference(**trip, reference=route([40, 99]), config=CONFIG)
    rows = [("0001", 0, 99, 0, 8., 0., 0., .5, False)]
    candidates = pd.DataFrame(rows, columns=list(CANDIDATE_DTYPES)).astype(CANDIDATE_DTYPES)
    result = coverage_summary(trip["observations"], candidates, labels)
    assert result["candidate_misses"] == 2
    assert result["candidate_miss_rate"] == 1
    assert result["label_coverage"] == 2 / 3
    assert result["no_candidate_observations"] == 2

