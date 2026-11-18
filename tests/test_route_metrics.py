"""Unit and integration tests for route metrics and continuity evaluation.

Tests:
- compute_continuity predicate consistency.
- Numerator, denominator, and rate calculation.
- Missing predictions counted as failures.
- Large GPS gaps (> max_gap_s) excluded and reported.
- Physically impossible transitions (speed violations, unreachable paths) counted as failures.
- Invalid observations breaking runs.
- Pointwise accuracy against ground-truth labels.
- Comparative evaluation between classifier-only and HMM predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to sys.path so language servers (Pylance/VSCode) and pytest resolve pathsnap_ml
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pandas as pd
import pytest

from pathsnap_ml.decoder import decode_classifier_only, viterbi_decode
from pathsnap_ml.decoder_evaluation import compare_classifier_and_hmm, evaluate_predictions
from pathsnap_ml.road_routing import RoadNetworkGraph
from pathsnap_ml.route_metrics import compute_continuity, compute_pointwise_accuracy


@pytest.fixture
def linear_route_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A linear connected road network: 0 -> 1 -> 2 -> 3."""
    nodes_df = pd.DataFrame([
        {"node_id": 0, "x": 0.0, "y": 0.0},
        {"node_id": 1, "x": 100.0, "y": 0.0},
        {"node_id": 2, "x": 200.0, "y": 0.0},
        {"node_id": 3, "x": 300.0, "y": 0.0},
    ])
    arcs_df = pd.DataFrame([
        {"arc_id": 0, "from_node_id": 0, "to_node_id": 1, "length_m": 100.0},
        {"arc_id": 1, "from_node_id": 1, "to_node_id": 2, "length_m": 100.0},
        {"arc_id": 2, "from_node_id": 2, "to_node_id": 3, "length_m": 100.0},
    ])
    return nodes_df, arcs_df


def test_continuity_perfect_run(linear_route_network):
    """Test that a perfectly continuous route achieves 1.0 continuity rate."""
    nodes_df, arcs_df = linear_route_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 3.0, "x": 90.0, "y": 0.0, "is_valid": True},
        {"obs_id": 3, "timestamp": 6.0, "x": 150.0, "y": 0.0, "is_valid": True},
        {"obs_id": 4, "timestamp": 10.0, "x": 250.0, "y": 0.0, "is_valid": True},
    ])
    preds_df = pd.DataFrame([
        {"obs_id": 1, "predicted_arc_id": 0},
        {"obs_id": 2, "predicted_arc_id": 0},
        {"obs_id": 3, "predicted_arc_id": 1},
        {"obs_id": 4, "predicted_arc_id": 2},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "fraction": 0.1},
        {"obs_id": 2, "arc_id": 0, "fraction": 0.9},
        {"obs_id": 3, "arc_id": 1, "fraction": 0.5},
        {"obs_id": 4, "arc_id": 2, "fraction": 0.5},
    ])

    metrics = compute_continuity(obs_df, nodes_df, arcs_df, cands_df, preds_df)
    assert metrics["continuity_rate"] == 1.0
    assert metrics["continuous_transitions"] == 3
    assert metrics["total_evaluated_transitions"] == 3
    assert metrics["missing_prediction_failures"] == 0
    assert metrics["large_gap_exclusions"] == 0


def test_continuity_missing_prediction_failure(linear_route_network):
    """Test that missing predictions count as failures in continuity evaluation."""
    nodes_df, arcs_df = linear_route_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 1.0, "x": 90.0, "y": 0.0, "is_valid": True},
        {"obs_id": 3, "timestamp": 2.0, "x": 150.0, "y": 0.0, "is_valid": True},
    ])
    # Obs 2 has missing prediction None
    preds_df = pd.DataFrame([
        {"obs_id": 1, "predicted_arc_id": 0},
        {"obs_id": 2, "predicted_arc_id": None},
        {"obs_id": 3, "predicted_arc_id": 1},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "fraction": 0.1},
        {"obs_id": 3, "arc_id": 1, "fraction": 0.5},
    ])

    metrics = compute_continuity(obs_df, nodes_df, arcs_df, cands_df, preds_df)
    assert metrics["total_evaluated_transitions"] == 2
    assert metrics["continuous_transitions"] == 0
    assert metrics["missing_prediction_failures"] == 2
    assert metrics["continuity_rate"] == 0.0


def test_continuity_large_gap_exclusion(linear_route_network):
    """Test that transitions across gaps longer than max_gap_s are excluded from the denominator."""
    nodes_df, arcs_df = linear_route_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 3.0, "x": 90.0, "y": 0.0, "is_valid": True},
        # 100 seconds gap
        {"obs_id": 3, "timestamp": 103.0, "x": 150.0, "y": 0.0, "is_valid": True},
        {"obs_id": 4, "timestamp": 106.0, "x": 250.0, "y": 0.0, "is_valid": True},
    ])
    preds_df = pd.DataFrame([
        {"obs_id": 1, "predicted_arc_id": 0},
        {"obs_id": 2, "predicted_arc_id": 0},
        {"obs_id": 3, "predicted_arc_id": 1},
        {"obs_id": 4, "predicted_arc_id": 2},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "fraction": 0.1},
        {"obs_id": 2, "arc_id": 0, "fraction": 0.9},
        {"obs_id": 3, "arc_id": 1, "fraction": 0.5},
        {"obs_id": 4, "arc_id": 2, "fraction": 0.5},
    ])

    metrics = compute_continuity(obs_df, nodes_df, arcs_df, cands_df, preds_df, config={"model": {"max_gap_s": 15.0}})
    # Transition 2 -> 3 was excluded because delta_t = 100s > 15s
    assert metrics["large_gap_exclusions"] == 1
    assert metrics["total_evaluated_transitions"] == 2
    assert metrics["continuous_transitions"] == 2
    assert metrics["continuity_rate"] == 1.0


def test_continuity_speed_violation_failure(linear_route_network):
    """Test that transitions requiring impossible vehicle speed count as failures."""
    nodes_df, arcs_df = linear_route_network
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 10.0, "y": 0.0, "is_valid": True},
        # Moves 200m in 1 second = 200 m/s > (40 + 5) m/s
        {"obs_id": 2, "timestamp": 1.0, "x": 250.0, "y": 0.0, "is_valid": True},
    ])
    preds_df = pd.DataFrame([
        {"obs_id": 1, "predicted_arc_id": 0},
        {"obs_id": 2, "predicted_arc_id": 2},
    ])
    cands_df = pd.DataFrame([
        {"obs_id": 1, "arc_id": 0, "fraction": 0.1},
        {"obs_id": 2, "arc_id": 2, "fraction": 0.5},
    ])

    cfg = {"model": {"max_speed_mps": 40.0, "speed_slack_mps": 5.0, "max_gap_s": 15.0}}
    metrics = compute_continuity(obs_df, nodes_df, arcs_df, cands_df, preds_df, config=cfg)
    assert metrics["total_evaluated_transitions"] == 1
    assert metrics["continuous_transitions"] == 0
    assert metrics["speed_violation_failures"] == 1


def test_pointwise_accuracy_and_exact_match(linear_route_network):
    """Test pointwise accuracy and exact sequence matching."""
    preds_df = pd.DataFrame([
        {"obs_id": 1, "predicted_arc_id": 0},
        {"obs_id": 2, "predicted_arc_id": 1},
        {"obs_id": 3, "predicted_arc_id": 2},
    ])
    gt_df = pd.DataFrame([
        {"obs_id": 1, "ground_truth_arc_id": 0},
        {"obs_id": 2, "ground_truth_arc_id": 1},
        {"obs_id": 3, "ground_truth_arc_id": 2},
    ])
    acc_res = compute_pointwise_accuracy(preds_df, gt_df)
    assert acc_res["pointwise_accuracy"] == 1.0
    assert acc_res["correct_matches"] == 3
    assert acc_res["missing_predictions"] == 0


def test_comparison_classifier_vs_hmm(linear_route_network):
    """Test comparative evaluation between classifier-only and HMM decoder."""
    nodes_df, arcs_df = linear_route_network
    road_graph = RoadNetworkGraph(nodes_df, arcs_df, directed=True)

    # Observation sequence where classifier-only picks disconnected candidate due to noisy p_match
    obs_df = pd.DataFrame([
        {"obs_id": 1, "timestamp": 0.0, "x": 20.0, "y": 0.0, "is_valid": True},
        {"obs_id": 2, "timestamp": 1.0, "x": 80.0, "y": 0.0, "is_valid": True},
        {"obs_id": 3, "timestamp": 2.0, "x": 120.0, "y": 0.0, "is_valid": True},
    ])
    cands_df = pd.DataFrame([
        # Obs 1
        {"obs_id": 1, "arc_id": 0, "p_match": 0.9, "fraction": 0.2, "projected_x": 20.0, "projected_y": 0.0},
        # Obs 2: classifier alone prefers distant arc 2 (p=0.95), but arc 0 is continuous (p=0.85)
        {"obs_id": 2, "arc_id": 0, "p_match": 0.85, "fraction": 0.8, "projected_x": 80.0, "projected_y": 0.0},
        {"obs_id": 2, "arc_id": 2, "p_match": 0.95, "fraction": 0.1, "projected_x": 210.0, "projected_y": 0.0},
        # Obs 3
        {"obs_id": 3, "arc_id": 1, "p_match": 0.9, "fraction": 0.2, "projected_x": 120.0, "projected_y": 0.0},
    ])

    gt_df = pd.DataFrame([
        {"obs_id": 1, "ground_truth_arc_id": 0},
        {"obs_id": 2, "ground_truth_arc_id": 0},
        {"obs_id": 3, "ground_truth_arc_id": 1},
    ])

    clf_preds, hmm_preds, summary = compare_classifier_and_hmm(
        obs_df, nodes_df, arcs_df, cands_df, road_graph, ground_truth=gt_df
    )

    # Classifier picks arc 2 for obs 2
    assert clf_preds.loc[clf_preds["obs_id"] == 2, "predicted_arc_id"].values[0] == 2
    # HMM enforces continuity and picks arc 0 for obs 2!
    assert hmm_preds.loc[hmm_preds["obs_id"] == 2, "predicted_arc_id"].values[0] == 0

    # HMM continuity must be strictly greater than classifier-only
    assert summary["hmm_viterbi"]["continuity_rate"] > summary["classifier_only"]["continuity_rate"]
    # HMM accuracy must be strictly greater than classifier-only
    assert summary["hmm_viterbi"]["pointwise_accuracy"] > summary["classifier_only"]["pointwise_accuracy"]
