import numpy as np
import pandas as pd

from pathsnap_ml.context_forest import trip_features


def test_context_uses_directed_connection_and_stops_at_time_gap(tmp_path):
    tid = "trip"
    folder = tmp_path / "data/processed" / tid
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text('{"graph_directed":true}')
    pd.DataFrame({"node_id": [0, 1, 2], "x_m": [0., 10., 10.], "y_m": [0., 0., 30.]}).to_parquet(folder / "nodes.parquet")
    pd.DataFrame({"arc_id": [0, 1, 2], "from_node_id": [0, 1, 1], "to_node_id": [1, 0, 2]}).to_parquet(folder / "arcs.parquet")
    pd.DataFrame({"obs_id": [0, 1, 2, 3], "order_index": [0, 1, 2, 3], "timestamp_s": [0., 1., 2., 150.],
                  "valid": True, "x_m": [5., 10., 10., 10.], "y_m": [0., 0., 10., 30.]}).to_parquet(folder / "observations.parquet")
    directory = tmp_path / "data/features"
    directory.mkdir()
    pd.DataFrame({"trip_id": tid, "obs_id": [0, 0, 2, 3], "candidate_arc_id": [0, 1, 2, 2],
                  "distance_m": 0., "projection_fraction": [.5, .5, 1/3, 1.]}).to_parquet(directory / f"{tid}.parquet")
    result = trip_features(tmp_path, tid)
    assert result.iloc[0].departure_12_error_m == 0
    assert result.iloc[1].departure_12_error_m > 0
    assert np.isnan(result.iloc[2].departure_12_error_m)
    assert np.isnan(result.iloc[3].approach_12_error_m)
