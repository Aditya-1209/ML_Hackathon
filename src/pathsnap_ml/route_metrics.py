"""Route metrics and continuity evaluation for map-matching predictions.

Follows PathSnap ML v1 specification:
- Exact compute_continuity interface for Person 3 and shared evaluations.
- Identical continuity predicate applied uniformly across classifier-only and HMM models.
- Transparent reporting of numerators, denominators, large-gap exclusions, and physical constraints.
- Pointwise accuracy, candidate miss rate, and exact-trip-match rate calculations.
"""
from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np
import pandas as pd

from pathsnap_ml.road_routing import CandidatePosition, RoadNetworkGraph


def compute_continuity(
    observations: pd.DataFrame,
    nodes: pd.DataFrame | dict[str, Any],
    arcs: pd.DataFrame | dict[str, Any],
    candidates: pd.DataFrame | list[Any],
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute route continuity across consecutive GPS observations.

    Guarantees:
    - Same continuity predicate for all classifier-only models and HMM.
    - Missing predictions count as failures.
    - Invalid observations break runs and are not evaluated across.
    - Large gaps (delta_t > max_gap_s) are excluded and reported.
    - Physically impossible transitions (speed limit exceeded or unreachable) count as failures.
    - Returns numerator, denominator, gap exclusions, and configured physical constraints.

    Args:
        observations: Original GPS observations DataFrame.
        nodes: Road network nodes DataFrame or dict.
        arcs: Road network arcs DataFrame or dict.
        candidates: Candidate positions DataFrame or list.
        predictions: Model predictions DataFrame with ['obs_id', 'predicted_arc_id'].
        metadata: Optional trip metadata dictionary.
        config: Optional configuration dictionary.

    Returns:
        Dictionary with:
            continuity_rate: float in [0.0, 1.0]
            continuous_transitions: int (numerator)
            total_evaluated_transitions: int (denominator)
            large_gap_exclusions: int
            missing_prediction_failures: int
            unreachable_failures: int
            speed_violation_failures: int
            configured_max_speed_mps: float
            configured_speed_slack_mps: float
            configured_max_gap_s: float
            configured_directed: bool
    """
    cfg = config or {}
    model_cfg = cfg.get("model", cfg)
    routing_cfg = cfg.get("routing", cfg)

    max_speed_mps = float(model_cfg.get("max_speed_mps", 40.0))
    speed_slack_mps = float(model_cfg.get("speed_slack_mps", 5.0))
    max_gap_s = float(model_cfg.get("max_gap_s", 15.0))
    directed = bool(routing_cfg.get("directed", True))

    # Build road network graph
    road_graph = RoadNetworkGraph(nodes=nodes, arcs=arcs, directed=directed)

    # Prepare observations
    obs_df = observations.copy()
    id_col = "obs_id" if "obs_id" in obs_df.columns else obs_df.index.name or "obs_id"
    if id_col not in obs_df.columns:
        obs_df[id_col] = np.arange(len(obs_df))
    time_col = "timestamp" if "timestamp" in obs_df.columns else obs_df.columns[2]
    x_col = "x" if "x" in obs_df.columns else ("lon" if "lon" in obs_df.columns else obs_df.columns[0])
    y_col = "y" if "y" in obs_df.columns else ("lat" if "lat" in obs_df.columns else obs_df.columns[1])
    valid_col = "is_valid" if "is_valid" in obs_df.columns else None

    # Sort observations chronologically
    sorted_obs = obs_df.sort_values(by=[time_col, id_col]).reset_index(drop=True)

    # Index predictions by obs_id
    pred_arc_col = "predicted_arc_id" if "predicted_arc_id" in predictions.columns else "arc_id"
    pred_id_col = "obs_id" if "obs_id" in predictions.columns else predictions.columns[0]
    preds_by_obs = {}
    for _, prow in predictions.iterrows():
        p_oid = int(prow[pred_id_col])
        val = prow[pred_arc_col]
        preds_by_obs[p_oid] = None if pd.isna(val) else int(val)

    # Index candidate positions by (obs_id, arc_id) to retrieve projection fraction
    cand_frac_map: dict[tuple[int, int], float] = {}
    if isinstance(candidates, pd.DataFrame):
        c_obs_col = "obs_id" if "obs_id" in candidates.columns else candidates.columns[0]
        c_arc_col = "arc_id" if "arc_id" in candidates.columns else candidates.columns[1]
        c_frac_col = "fraction" if "fraction" in candidates.columns else None
        for _, crow in candidates.iterrows():
            c_oid = int(crow[c_obs_col])
            c_aid = int(crow[c_arc_col])
            frac = float(crow[c_frac_col]) if c_frac_col and not pd.isna(crow[c_frac_col]) else 0.5
            cand_frac_map[(c_oid, c_aid)] = frac
    elif isinstance(candidates, Sequence):
        for c in candidates:
            if hasattr(c, "obs_id") and hasattr(c, "arc_id") and hasattr(c, "fraction"):
                cand_frac_map[(c.obs_id, c.arc_id)] = float(c.fraction)

    continuous_transitions = 0
    total_evaluated_transitions = 0
    large_gap_exclusions = 0
    missing_prediction_failures = 0
    unreachable_failures = 0
    speed_violation_failures = 0

    prev_row = None

    for _, curr_row in sorted_obs.iterrows():
        if prev_row is None:
            prev_row = curr_row
            continue

        prev_oid = int(prev_row[id_col])
        curr_oid = int(curr_row[id_col])
        prev_time = float(prev_row[time_col])
        curr_time = float(curr_row[time_col])
        delta_t = curr_time - prev_time

        prev_valid = bool(prev_row[valid_col]) if valid_col else True
        curr_valid = bool(curr_row[valid_col]) if valid_col else True

        # If either observation is invalid, break run without evaluating transition
        if not prev_valid or not curr_valid:
            prev_row = curr_row
            continue

        # Check large GPS gap exclusion
        if delta_t > max_gap_s:
            large_gap_exclusions += 1
            prev_row = curr_row
            continue

        # Evaluated transition
        total_evaluated_transitions += 1
        prev_arc = preds_by_obs.get(prev_oid)
        curr_arc = preds_by_obs.get(curr_oid)

        # Missing prediction counts as continuity failure
        if prev_arc is None or curr_arc is None or prev_arc not in road_graph.arcs or curr_arc not in road_graph.arcs:
            missing_prediction_failures += 1
            prev_row = curr_row
            continue

        # Get projection fractions or project GPS coords if missing
        prev_frac = cand_frac_map.get((prev_oid, prev_arc))
        if prev_frac is None:
            proj = road_graph.project_point_to_arc(float(prev_row[x_col]), float(prev_row[y_col]), prev_arc)
            prev_frac = proj.fraction

        curr_frac = cand_frac_map.get((curr_oid, curr_arc))
        if curr_frac is None:
            proj = road_graph.project_point_to_arc(float(curr_row[x_col]), float(curr_row[y_col]), curr_arc)
            curr_frac = proj.fraction

        c1 = CandidatePosition(
            arc_id=prev_arc,
            fraction=prev_frac,
            projected_x=0.0,
            projected_y=0.0,
            obs_id=prev_oid,
        )
        c2 = CandidatePosition(
            arc_id=curr_arc,
            fraction=curr_frac,
            projected_x=0.0,
            projected_y=0.0,
            obs_id=curr_oid,
        )

        net_dist, _ = road_graph.network_distance(c1, c2)

        if math.isinf(net_dist):
            unreachable_failures += 1
            prev_row = curr_row
            continue

        # Speed constraint check
        if delta_t > 0:
            effective_speed = net_dist / delta_t
            if effective_speed > (max_speed_mps + speed_slack_mps):
                speed_violation_failures += 1
                prev_row = curr_row
                continue
        elif delta_t == 0 and net_dist > 0.01:
            speed_violation_failures += 1
            prev_row = curr_row
            continue

        # Transition passed all physical continuity checks!
        continuous_transitions += 1
        prev_row = curr_row

    rate = (
        continuous_transitions / total_evaluated_transitions
        if total_evaluated_transitions > 0
        else 0.0
    )

    return {
        "continuity_rate": rate,
        "continuous_transitions": continuous_transitions,
        "total_evaluated_transitions": total_evaluated_transitions,
        "large_gap_exclusions": large_gap_exclusions,
        "missing_prediction_failures": missing_prediction_failures,
        "unreachable_failures": unreachable_failures,
        "speed_violation_failures": speed_violation_failures,
        "configured_max_speed_mps": max_speed_mps,
        "configured_speed_slack_mps": speed_slack_mps,
        "configured_max_gap_s": max_gap_s,
        "configured_directed": directed,
    }


def compute_pointwise_accuracy(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame | Sequence[int],
) -> dict[str, float]:
    """Compute observation-level pointwise accuracy and coverage against offline reference labels.

    Args:
        predictions: DataFrame with 'predicted_arc_id' and 'obs_id'.
        ground_truth: DataFrame with 'obs_id' and 'ground_truth_arc_id', or sequence of arc IDs.

    Returns:
        Dict with 'pointwise_accuracy', 'total_observations', 'correct_matches', 'missing_predictions'.
    """
    if isinstance(ground_truth, pd.DataFrame):
        gt_map = dict(zip(ground_truth["obs_id"], ground_truth["ground_truth_arc_id"]))
    else:
        gt_map = dict(enumerate(ground_truth))

    correct = 0
    total = len(predictions)
    missing = 0

    pred_col = "predicted_arc_id" if "predicted_arc_id" in predictions.columns else predictions.columns[1]
    id_col = "obs_id" if "obs_id" in predictions.columns else predictions.columns[0]

    for _, row in predictions.iterrows():
        oid = int(row[id_col])
        pred_val = row[pred_col]
        if pd.isna(pred_val) or pred_val is None:
            missing += 1
            continue
        gt_val = gt_map.get(oid)
        if gt_val is not None and int(pred_val) == int(gt_val):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "pointwise_accuracy": acc,
        "total_observations": total,
        "correct_matches": correct,
        "missing_predictions": missing,
    }
