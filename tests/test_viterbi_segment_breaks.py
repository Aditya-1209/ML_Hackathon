import numpy as np
import pandas as pd

from pathsnap_ml.candidates import make_candidates
from pathsnap_ml.contracts import KEY
from pathsnap_ml.decoder import decode_trip
from pathsnap_ml.route_metrics import compute_continuity


def test_viterbi_resolves_weak_emission_and_records_disconnected_switch():
    # Parallel one-way roads with no connecting arc.
    nodes = pd.DataFrame({"node_id": [0, 1, 2, 3], "x_m": [0., 100., 0., 100.], "y_m": [0., 0., 2., 2.]})
    arcs = pd.DataFrame({"arc_id": [0, 1], "from_node_id": [0, 2], "to_node_id": [1, 3], "length_m": [100., 100.]})
    obs = pd.DataFrame({"trip_id": "T", "obs_id": range(4), "order_index": range(4), "timestamp_s": [0., 1., 2., 3.],
                        "x_m": [5., 15., 25., 35.], "y_m": 1., "valid": True})
    meta = {"trip_id": "T", "graph_directed": True}
    candidates = make_candidates(obs, nodes, arcs, meta, {"radius_m": 5, "fallback_radius_m": 10, "max_candidates": 2})
    scores = candidates[KEY].copy()
    prob = np.array([.999, .4, .999, .001])[scores.obs_id]
    scores["p_match"] = np.where(scores.candidate_arc_id == 0, prob, 1 - prob)
    cfg = {"beta_m": 50, "epsilon": 1e-9, "max_speed_mps": 55, "slack_m": 30, "gap_s": 60}
    strict, _ = decode_trip(obs, nodes, arcs, candidates, scores, meta, cfg)
    assert strict.predicted_arc_id.tolist() == [0, 0, 0, 0]
    result, route = decode_trip(obs, nodes, arcs, candidates, scores, meta, {**cfg, "break_penalty": 1})
    assert result.predicted_arc_id.tolist() == [0, 0, 0, 1]
    assert route["runs"] == [{"obs_ids": [0, 1, 2], "arc_ids": [0]}, {"obs_ids": [3], "arc_ids": [1]}]
    assert route["breaks"][-1]["break_reason"] == "unreachable_transition"
    counts = compute_continuity(obs, nodes, arcs, candidates, result, meta, cfg)
    assert (counts["feasible"], counts["eligible"]) == (2, 3)
