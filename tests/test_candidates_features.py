import numpy as np
import pandas as pd
import pytest

from pathsnap_ml.candidates import KEY_COLUMNS, make_candidates
from pathsnap_ml.features import FEATURE_COLUMNS, make_features
from pathsnap_ml.models.random_forest import build_estimator


@pytest.fixture
def trip():
    observations = pd.DataFrame({
        "trip_id": ["0001"] * 6, "obs_id": [0, 1, 2, 3, 4, 5],
        "order_index": pd.array([0, 1, 2, 3, 4, None], dtype="Int64"),
        "timestamp_s": [0., 1., 1., 2., 100., np.nan],
        "x_m": [1., 9., 8., 25., 100., np.nan],
        "y_m": [0., 0., 0., 0., 0., np.nan],
        "valid": [True] * 5 + [False],
    })
    nodes = pd.DataFrame({
        "trip_id": ["0001"] * 2, "node_id": [0, 1],
        "x_m": [0., 10.], "y_m": [0., 0.],
    })
    arcs = pd.DataFrame({
        "trip_id": ["0001"] * 4, "arc_id": [7, 3, 4, 5],
        "from_node_id": [0, 0, 1, 0], "to_node_id": [1, 1, 0, 0],
        "length_m": [10., 10., 10., 0.],
    })
    metadata = {"trip_id": "0001", "graph_directed": True}
    return dict(observations=observations, nodes=nodes, arcs=arcs, metadata=metadata)


CONFIG = dict(radius_m=2, fallback_radius_m=20, max_candidates=8,
              heading_min_step_m=3, motion_max_gap_s=30)


def test_candidates_keep_parallel_arcs_and_expand_only_when_empty(trip):
    candidates = make_candidates(**trip, config=CONFIG)
    pd.testing.assert_frame_equal(candidates, make_candidates(**trip, config=CONFIG))
    assert candidates.groupby("obs_id").candidate_arc_id.apply(list).to_dict() == {
        0: [3, 4, 7, 5], 1: [3, 4, 7], 2: [3, 4, 7], 3: [3, 4, 7]
    }
    assert candidates.used_expanded_search.eq(candidates.obs_id.eq(3)).all()
    assert candidates.loc[candidates.candidate_arc_id.eq(5), "projection_fraction"].eq(0).all()
    assert not candidates.duplicated(KEY_COLUMNS).any()


def test_exact_features_and_motion_policies(trip):
    candidates = make_candidates(**trip, config=CONFIG)
    features = make_features(**trip, candidates=candidates, config=CONFIG)
    assert features.columns.tolist() == KEY_COLUMNS + list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(features[KEY_COLUMNS], candidates[KEY_COLUMNS])
    assert features.loc[features.obs_id.eq(0), "heading_available"].eq(0).all()
    repeated = features.loc[features.obs_id.eq(2)]
    assert repeated.delta_t_s.eq(0).all()
    assert repeated.speed_mps.isna().all()
    assert repeated.heading_difference_deg.isna().all()
    reverse = features.obs_id.eq(1) & features.candidate_arc_id.eq(4)
    assert features.loc[reverse, "heading_difference_deg"].item() == 180
    trip["metadata"]["graph_directed"] = False
    undirected = make_features(**trip, candidates=candidates, config=CONFIG)
    assert undirected.loc[reverse, "heading_difference_deg"].item() == 0


def test_empty_candidates_and_features(trip):
    trip["arcs"] = trip["arcs"].iloc[:0]
    candidates = make_candidates(**trip, config=CONFIG)
    features = make_features(**trip, candidates=candidates, config=CONFIG)
    assert candidates.empty and features.empty
    assert features.columns.tolist() == KEY_COLUMNS + list(FEATURE_COLUMNS)


def test_random_forest_handles_entirely_missing_column():
    features = np.ones((8, 8))
    features[:, 0] = np.arange(8)
    features[:, 1] = np.nan
    estimator = build_estimator({"model__n_estimators": 4})
    assert not hasattr(estimator.named_steps["model"], "classes_")
    estimator.fit(features, np.arange(8) % 2)
    scores = estimator.predict_proba(features)[:, np.flatnonzero(estimator.classes_ == 1)[0]]
    assert estimator.named_steps["imputer"].transform(features).shape == features.shape
    assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()

