"""Unit and edge-case tests for road routing, HMM, and Viterbi decoder.

Covers:
- Tiny hand-verifiable road graph.
- Connected and disconnected roads.
- Directed and undirected graph behavior.
- Parallel edges.
- Zero-length edges.
- Same-arc projected positions (forward, reverse, loop cycles).
- Repeated road visits.
- No candidate observations.
- Invalid observations.
- Large GPS gaps (> max_gap_s).
- Zero delta time.
- All-invalid transitions (transparent restarts).
- Shuffled candidate score rows.
- Duplicate score keys and tied scores (deterministic tie-breaking).
- Missing score keys.
- Viterbi backtracking correctness.
- Exactly one output row per original observation in original order.
- Label-free inference.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Add src/ to sys.path so language servers (Pylance/VSCode) and pytest resolve pathsnap_ml
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
import pytest

from pathsnap_ml.decoder import DecoderConfig, decode_classifier_only, viterbi_decode
from pathsnap_ml.road_routing import CandidatePosition, RoadNetworkGraph


@pytest.fixture
def hand_verifiable_network() -> RoadNetworkGraph:
    """A hand-verifiable directed road network:
    Nodes:
        0: (0, 0)
        1: (100, 0)
        2: (200, 0)
        3: (300, 0)
        4: (100, 100) - for loop back to 0
        5: (999, 999) - disconnected island
    Arcs:
        0: 0 -> 1 (len 100)
        1: 1 -> 2 (len 100)
        2: 2 -> 3 (len 100)
        3: 1 -> 2 (len 120, parallel edge with arc 1)
        4: 0 -> 0 (len 0, zero-length loop)
        5: 1 -> 4 (len 100)
        6: 4 -> 0 (len 141.42, completes loop back to 0)
        7: 5 -> 5 (len 0, disconnected island)
    """
    nodes_df = pd.DataFrame([
        {"node_id": 0, "x": 0.0, "y": 0.0},
        {"node_id": 1, "x": 100.0, "y": 0.0},
        {"node_id": 2, "x": 200.0, "y": 0.0},
        {"node_id": 3, "x": 300.0, "y": 0.0},
        {"node_id": 4, "x": 100.0, "y": 100.0},
        {"node_id": 5, "x": 999.0, "y": 999.0},
    ])
    arcs_df = pd.DataFrame([
        {"arc_id": 0, "from_node_id": 0, "to_node_id": 1, "length_m": 100.0},
        {"arc_id": 1, "from_node_id": 1, "to_node_id": 2, "length_m": 100.0},
        {"arc_id": 2, "from_node_id": 2, "to_node_id": 3, "length_m": 100.0},
        {"arc_id": 3, "from_node_id": 1, "to_node_id": 2, "length_m": 120.0},
        {"arc_id": 4, "from_node_id": 0, "to_node_id": 0, "length_m": 0.0},
        {"arc_id": 5, "from_node_id": 1, "to_node_id": 4, "length_m": 100.0},
        {"arc_id": 6, "from_node_id": 4, "to_node_id": 0, "length_m": 141.42},
        {"arc_id": 7, "from_node_id": 5, "to_node_id": 5, "length_m": 0.0},
    ])
    return RoadNetworkGraph(nodes_df, arcs_df, directed=True, network_id="test_hand")


def test_tiny_hand_verifiable_and_connected_routing(hand_verifiable_network: RoadNetworkGraph):
    """Test shortest node path on a hand-verifiable graph."""
    graph = hand_verifiable_network
    dist, path = graph.shortest_node_path(0, 3)
    # 0 -> 1 -> 2 -> 3 = 100 + 100 + 100 = 300
    assert abs(dist - 300.0) < 1e-4
    assert path == [0, 1, 2]


def test_parallel_edges(hand_verifiable_network: RoadNetworkGraph):
    """Test that parallel edges are handled and shortest edge is selected."""
    graph = hand_verifiable_network
    # Arc 1 is length 100, Arc 3 is length 120 (both 1 -> 2)
    dist, path = graph.shortest_node_path(1, 2)
    assert abs(dist - 100.0) < 1e-4
    assert path == [1]


def test_disconnected_roads(hand_verifiable_network: RoadNetworkGraph):
    """Test that disconnected nodes return infinite distance without inventing connections."""
    graph = hand_verifiable_network
    dist, path = graph.shortest_node_path(0, 5)
    assert math.isinf(dist)
    assert path == []


def test_zero_length_edges(hand_verifiable_network: RoadNetworkGraph):
    """Test zero-length edge projection and routing."""
    graph = hand_verifiable_network
    # Project point onto zero length arc 4 (at node 0)
    proj = graph.project_point_to_arc(10.0, 5.0, arc_id=4)
    assert proj.fraction == 0.0
    assert proj.projected_x == 0.0
    assert proj.projected_y == 0.0
    assert abs(proj.distance_to_arc_m - math.hypot(10.0, 5.0)) < 1e-4


def test_same_arc_projected_positions_forward_and_backward(hand_verifiable_network: RoadNetworkGraph):
    """Test same-arc routing: forward along directed arc vs backward loop."""
    graph = hand_verifiable_network

    # Forward on arc 0: frac 0.2 to 0.7 -> (0.7 - 0.2) * 100 = 50m
    c1 = CandidatePosition(arc_id=0, fraction=0.2, projected_x=20.0, projected_y=0.0)
    c2 = CandidatePosition(arc_id=0, fraction=0.7, projected_x=70.0, projected_y=0.0)
    d_fwd, route_fwd = graph.network_distance(c1, c2)
    assert abs(d_fwd - 50.0) < 1e-4
    assert route_fwd == [0]

    # Backward on arc 0 in directed mode: must loop through network
    # From frac 0.7 to end of arc 0 (30m) -> node 1 -> 4 (100m) -> 0 (141.42m) -> arc 0 to frac 0.2 (20m)
    # Total = 30 + 100 + 141.42 + 20 = 291.42m
    d_bwd, route_bwd = graph.network_distance(c2, c1)
    expected_bwd = (1.0 - 0.7) * 100.0 + 100.0 + 141.42 + 0.2 * 100.0
    assert abs(d_bwd - expected_bwd) < 1e-2
    assert route_bwd == [0, 5, 6, 0]


def test_undirected_graph_support():
    """Test routing when directed=False."""
    nodes_df = pd.DataFrame([
        {"node_id": 0, "x": 0.0, "y": 0.0},
        {"node_id": 1, "x": 100.0, "y": 0.0},
    ])
    arcs_df = pd.DataFrame([
        {"arc_id": 0, "from_node_id": 0, "to_node_id": 1, "length_m": 100.0},
    ])
    graph = RoadNetworkGraph(nodes_df, arcs_df, directed=False)
    # Move backwards along arc 0: from frac 0.8 to frac 0.3
    c1 = CandidatePosition(arc_id=0, fraction=0.8, projected_x=80.0, projected_y=0.0)
    c2 = CandidatePosition(arc_id=0, fraction=0.3, projected_x=30.0, projected_y=0.0)
    d, route = graph.network_distance(c1, c2)
    assert abs(d - 50.0) < 1e-4
    assert route == [0]


def test_repeated_road_visits(hand_verifiable_network: RoadNetworkGraph):
    """Test that routes can legally revisit the same road through loop."""
    graph = hand_verifiable_network
    # Path from node 0 through loop back to node 1
    # 0 -> 1 -> 4 -> 0 -> 1: arcs [0, 5, 6, 0]
    c_start = CandidatePosition(arc_id=0, fraction=0.5, projected_x=50.0, projected_y=0.0)
    c_loop = CandidatePosition(arc_id=0, fraction=0.9, projected_x=90.0, projected_y=0.0)
    # With same arc forward: frac 0.5 to 0.9 is 40m
    d, r = graph.network_distance(c_start, c_loop)
    assert abs(d - 40.0) < 1e-4


def test_viterbi_one_output_per_original_observation(hand_verifiable_network: RoadNetworkGraph):
    """Verify that Viterbi decoder outputs exactly one prediction row per original observation."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 101, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 102, "timestamp": 1.0, "x": 90.0, "y": 0.0, "is_valid": True},
        {"obs_id": 103, "timestamp": 2.0, "x": 150.0, "y": 0.0, "is_valid": True},
        {"obs_id": 104, "timestamp": 3.0, "x": 250.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 101, "arc_id": 0, "p_match": 0.9, "fraction": 0.1, "projected_x": 10.0, "projected_y": 0.0},
        {"obs_id": 102, "arc_id": 0, "p_match": 0.9, "fraction": 0.9, "projected_x": 90.0, "projected_y": 0.0},
        {"obs_id": 103, "arc_id": 1, "p_match": 0.8, "fraction": 0.5, "projected_x": 150.0, "projected_y": 0.0},
        {"obs_id": 104, "arc_id": 2, "p_match": 0.95, "fraction": 0.5, "projected_x": 250.0, "projected_y": 0.0},
    ])

    res = viterbi_decode(obs_df, cands_df, graph)
    assert len(res) == len(obs_df)
    assert list(res["obs_id"]) == [101, 102, 103, 104]
    assert list(res["predicted_arc_id"]) == [0, 0, 1, 2]


def test_no_candidate_observations(hand_verifiable_network: RoadNetworkGraph):
    """Test handling of observations that have 0 candidates."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 1.0, "x": 500.0, "y": 500.0, "is_valid": True},  # No candidate
        {"obs_id": 3, "timestamp": 2.0, "x": 110.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.9, "fraction": 0.1, "projected_x": 10.0, "projected_y": 0.0},
        # Obs 2 has no rows
        {"obs_id": 3, "arc_id": 1, "p_match": 0.9, "fraction": 0.1, "projected_x": 110.0, "projected_y": 0.0},
    ])

    res = viterbi_decode(obs_df, cands_df, graph)
    assert len(res) == 3
    assert res.iloc[0]["predicted_arc_id"] == 0
    assert pd.isna(res.iloc[1]["predicted_arc_id"]) or res.iloc[1]["predicted_arc_id"] is None
    assert res.iloc[2]["predicted_arc_id"] == 1


def test_invalid_observations(hand_verifiable_network: RoadNetworkGraph):
    """Test handling of invalid observations (is_valid=False or NaN coordinates)."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 1.0, "x": np.nan, "y": np.nan, "is_valid": False},
        {"obs_id": 3, "timestamp": 2.0, "x": 110.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.9, "fraction": 0.1, "projected_x": 10.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 0, "p_match": 0.5, "fraction": 0.5, "projected_x": 50.0, "projected_y": 0.0},
        {"obs_id": 3, "arc_id": 1, "p_match": 0.9, "fraction": 0.1, "projected_x": 110.0, "projected_y": 0.0},
    ])

    res = viterbi_decode(obs_df, cands_df, graph)
    assert len(res) == 3
    assert pd.isna(res.iloc[1]["predicted_arc_id"])
    assert res.iloc[1]["segment_id"] == -1


def test_large_gps_gaps(hand_verifiable_network: RoadNetworkGraph):
    """Test that time gaps exceeding max_gap_s split the Viterbi segment."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 1.0, "x": 90.0, "y": 0.0, "is_valid": True},
        # Gap of 50 seconds > max_gap_s (15s)
        {"obs_id": 3, "timestamp": 51.0, "x": 110.0, "y": 0.0, "is_valid": True},
        {"obs_id": 4, "timestamp": 52.0, "x": 190.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.9, "fraction": 0.1, "projected_x": 10.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 0, "p_match": 0.9, "fraction": 0.9, "projected_x": 90.0, "projected_y": 0.0},
        {"obs_id": 3, "arc_id": 1, "p_match": 0.9, "fraction": 0.1, "projected_x": 110.0, "projected_y": 0.0},
        {"obs_id": 4, "arc_id": 1, "p_match": 0.9, "fraction": 0.9, "projected_x": 190.0, "projected_y": 0.0},
    ])

    cfg = DecoderConfig(max_gap_s=15.0)
    res = viterbi_decode(obs_df, cands_df, graph, config=cfg)
    assert len(res) == 4
    # The segment_id of obs 3 and 4 must differ from obs 1 and 2
    seg_early = res.loc[res["obs_id"] == 1, "segment_id"].values[0]
    seg_late = res.loc[res["obs_id"] == 3, "segment_id"].values[0]
    assert seg_early != seg_late


def test_zero_delta_time(hand_verifiable_network: RoadNetworkGraph):
    """Test handling of observations with identical timestamps."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 10.0, "x": 50.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 10.0, "x": 50.0, "y": 0.0, "is_valid": True},  # zero dt
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.9, "fraction": 0.5, "projected_x": 50.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 0, "p_match": 0.9, "fraction": 0.5, "projected_x": 50.0, "projected_y": 0.0},
    ])
    res = viterbi_decode(obs_df, cands_df, graph)
    assert len(res) == 2
    assert list(res["predicted_arc_id"]) == [0, 0]


def test_all_invalid_transitions_transparent_restart(hand_verifiable_network: RoadNetworkGraph):
    """Test that when all transitions are impossible (e.g., jump to disconnected island), Viterbi restarts transparently."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 50.0, "y": 0.0, "is_valid": True},
        # Vehicle teleports to disconnected island node 5 in 1 second (impossible)
        {"obs_id": 2, "timestamp": 1.0, "x": 999.0, "y": 999.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.95, "fraction": 0.5, "projected_x": 50.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 7, "p_match": 0.90, "fraction": 0.0, "projected_x": 999.0, "projected_y": 999.0},
    ])

    res = viterbi_decode(obs_df, cands_df, graph)
    assert len(res) == 2
    assert res.iloc[0]["predicted_arc_id"] == 0
    assert res.iloc[1]["predicted_arc_id"] == 7
    # Restarted in a new segment
    assert res.iloc[0]["segment_id"] != res.iloc[1]["segment_id"]


def test_shuffled_score_rows_order_invariance(hand_verifiable_network: RoadNetworkGraph):
    """Test that shuffling candidate rows produces identical predictions."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 20.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 1.0, "x": 120.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.8, "fraction": 0.2, "projected_x": 20.0, "projected_y": 0.0},
        {"obs_id": 1, "arc_id": 1, "p_match": 0.2, "fraction": 0.0, "projected_x": 100.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 0, "p_match": 0.1, "fraction": 1.0, "projected_x": 100.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 1, "p_match": 0.9, "fraction": 0.2, "projected_x": 120.0, "projected_y": 0.0},
    ])

    # Run original
    res1 = viterbi_decode(obs_df, cands_df, graph)
    # Shuffle candidates
    shuffled_cands = cands_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    res2 = viterbi_decode(obs_df, shuffled_cands, graph)

    assert list(res1["predicted_arc_id"]) == list(res2["predicted_arc_id"])


def test_duplicate_score_keys_deterministic_tie_breaking(hand_verifiable_network: RoadNetworkGraph):
    """Test deterministic tie-breaking when candidate scores tie exactly."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 150.0, "y": 0.0, "is_valid": True},
    ])
    # Arc 1 and Arc 3 both connect (1 -> 2); give them identical p_match
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 3, "p_match": 0.5, "fraction": 0.5, "projected_x": 150.0, "projected_y": 0.0},
        {"obs_id": 1, "arc_id": 1, "p_match": 0.5, "fraction": 0.5, "projected_x": 150.0, "projected_y": 0.0},
    ])
    res = viterbi_decode(obs_df, cands_df, graph)
    # Tied scores break deterministically by smaller arc_id (1 < 3)
    assert res.iloc[0]["predicted_arc_id"] == 1


def test_label_free_inference(hand_verifiable_network: RoadNetworkGraph):
    """Verify that inference does not access reference route files."""
    graph = hand_verifiable_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 50.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "p_match": 0.8, "fraction": 0.5, "projected_x": 50.0, "projected_y": 0.0},
    ])
    # Ensure no 'ground_truth' or 'reference' argument exists in viterbi_decode
    import inspect
    sig = inspect.signature(viterbi_decode)
    assert "reference" not in sig.parameters
    assert "ground_truth" not in sig.parameters
    res = viterbi_decode(obs_df, cands_df, graph)
    assert res.iloc[0]["predicted_arc_id"] == 0
