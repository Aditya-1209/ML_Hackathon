"""Hidden Markov Model (HMM) and Viterbi decoder for map-matching.

Follows PathSnap ML v1 specification:
- Hidden states: road-segment candidates for each GPS observation.
- Emission score: E_t(c) = log(max(p_match, epsilon)) [optional normalization].
- Transition score: T(c_i, c_j) = -abs(network_dist_m - observed_disp_m) / beta_m.
- Physical plausibility constraints: speed limits, unreachable segments, and large GPS gaps.
- Deterministic log-space Viterbi dynamic programming and backtracking.
- Transparent restart on gaps, missing candidates, invalid observations, or disconnected transitions.
- Returns exactly one prediction row per original observation.
- Strictly label-free during inference (no reference files accessed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from pathsnap_ml.road_routing import CandidatePosition, RoadNetworkGraph


@dataclass
class DecoderConfig:
    """Configuration parameters for HMM Viterbi decoding."""
    beta_m: float = 10.0
    epsilon: float = 1e-6
    normalize_emissions: bool = False
    max_speed_mps: float = 40.0
    speed_slack_mps: float = 5.0
    max_gap_s: float = 15.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> DecoderConfig:
        if not d:
            return cls()
        model_cfg = d.get("model", d)
        return cls(
            beta_m=float(model_cfg.get("beta_m", 10.0)),
            epsilon=float(model_cfg.get("epsilon", 1e-6)),
            normalize_emissions=bool(model_cfg.get("normalize_emissions", False)),
            max_speed_mps=float(model_cfg.get("max_speed_mps", 40.0)),
            speed_slack_mps=float(model_cfg.get("speed_slack_mps", 5.0)),
            max_gap_s=float(model_cfg.get("max_gap_s", 15.0)),
        )


def compute_emission_score(
    p_match: float,
    epsilon: float = 1e-6,
    total_candidate_prob: float | None = None,
    normalize: bool = False,
) -> float:
    """Compute log emission score for a candidate state.

    E_t(c) = log(max(p_match, epsilon))

    Args:
        p_match: Classifier match probability in [0.0, 1.0].
        epsilon: Probability floor to prevent log(0).
        total_candidate_prob: Sum of probabilities for candidate set (for normalization).
        normalize: If True, normalizes p_match by total_candidate_prob before taking log.

    Returns:
        Log emission score (float <= 0.0).
    """
    prob = float(p_match)
    if normalize and total_candidate_prob is not None and total_candidate_prob > 0.0:
        prob = prob / total_candidate_prob
    val = max(prob, epsilon)
    return math.log(val)


def compute_transition_score(
    from_cand: CandidatePosition,
    to_cand: CandidatePosition,
    from_time_s: float,
    to_time_s: float,
    obs_disp_m: float,
    road_graph: RoadNetworkGraph,
    config: DecoderConfig,
) -> float:
    """Compute log transition score between two candidate positions.

    T(c_i, c_j) = -abs(network_distance_m - observed_displacement_m) / beta_m
    Returns float('-inf') if unreachable, speed limit exceeded, or large gap present.

    Args:
        from_cand: Candidate state at time t.
        to_cand: Candidate state at time t+1.
        from_time_s: Timestamp in seconds at time t.
        to_time_s: Timestamp in seconds at time t+1.
        obs_disp_m: Euclidean distance between GPS observations in metres.
        road_graph: RoadNetworkGraph instance.
        config: Decoder configuration.

    Returns:
        Log transition score (float <= 0.0) or float('-inf').
    """
    delta_t = to_time_s - from_time_s

    # Large gap: treated as segment boundary, transition not feasible
    if delta_t > config.max_gap_s:
        return float("-inf")

    # Zero or negative delta_t handling
    if delta_t <= 0.0:
        if delta_t < 0.0:
            return float("-inf")
        # delta_t == 0: stationary transition is only allowed if on same arc and exact same position
        if from_cand.arc_id == to_cand.arc_id and abs(from_cand.fraction - to_cand.fraction) < 1e-4:
            return 0.0
        return float("-inf")

    # Compute position-aware road network shortest path distance
    net_dist, _ = road_graph.network_distance(from_cand, to_cand)
    if math.isinf(net_dist):
        return float("-inf")

    # Physical feasibility: speed constraint
    effective_speed = net_dist / delta_t
    max_allowed_speed = config.max_speed_mps + config.speed_slack_mps
    if effective_speed > max_allowed_speed:
        return float("-inf")

    # Exponential transition score in log-space
    score = -abs(net_dist - obs_disp_m) / config.beta_m
    return score


def viterbi_decode(
    observations: pd.DataFrame,
    candidates: pd.DataFrame | list[CandidatePosition],
    road_graph: RoadNetworkGraph,
    config: DecoderConfig | None = None,
) -> pd.DataFrame:
    """Run log-space Viterbi decoding over chronological GPS observations.

    Guarantees:
    - Exactly one output row per original observation in original order.
    - Stable chronological sorting and deterministic tie-breaking.
    - Handles missing candidates, invalid rows, large gaps, and disconnected graphs.
    - Transparently restarts at the next valid observation when continuity is broken.
    - Preserves observation IDs and records intermediate route segments.

    Args:
        observations: DataFrame with columns 'obs_id', 'timestamp', 'x', 'y' (or lon/lat),
                      and optional 'is_valid'.
        candidates: DataFrame or list of CandidatePosition with columns 'obs_id', 'arc_id',
                    'p_match', 'fraction', 'projected_x', 'projected_y'.
        road_graph: RoadNetworkGraph instance.
        config: DecoderConfig instance.

    Returns:
        DataFrame with columns:
            ['obs_id', 'timestamp', 'predicted_arc_id', 'match_probability',
             'method', 'segment_id', 'is_valid', 'candidate_count']
    """
    if config is None:
        config = DecoderConfig()

    obs_df = observations.copy()
    id_col = "obs_id" if "obs_id" in obs_df.columns else obs_df.index.name or "obs_id"
    if id_col not in obs_df.columns:
        obs_df[id_col] = np.arange(len(obs_df))

    time_col = "timestamp" if "timestamp" in obs_df.columns else obs_df.columns[2]
    x_col = "x" if "x" in obs_df.columns else ("lon" if "lon" in obs_df.columns else obs_df.columns[0])
    y_col = "y" if "y" in obs_df.columns else ("lat" if "lat" in obs_df.columns else obs_df.columns[1])
    valid_col = "is_valid" if "is_valid" in obs_df.columns else None

    # Index candidates by obs_id
    cands_by_obs: dict[int, list[CandidatePosition]] = {}
    if isinstance(candidates, pd.DataFrame):
        cand_obs_col = "obs_id" if "obs_id" in candidates.columns else candidates.columns[0]
        cand_arc_col = "arc_id" if "arc_id" in candidates.columns else candidates.columns[1]
        p_col = "p_match" if "p_match" in candidates.columns else ("score" if "score" in candidates.columns else None)
        frac_col = "fraction" if "fraction" in candidates.columns else None
        px_col = "projected_x" if "projected_x" in candidates.columns else None
        py_col = "projected_y" if "projected_y" in candidates.columns else None

        for _, r in candidates.iterrows():
            oid = int(r[cand_obs_col])
            aid = int(r[cand_arc_col])
            pm = float(r[p_col]) if p_col and not pd.isna(r[p_col]) else 1.0
            frac = float(r[frac_col]) if frac_col and not pd.isna(r[frac_col]) else 0.5
            px = float(r[px_col]) if px_col and not pd.isna(r[px_col]) else 0.0
            py = float(r[py_col]) if py_col and not pd.isna(r[py_col]) else 0.0

            cpos = CandidatePosition(
                arc_id=aid,
                fraction=frac,
                projected_x=px,
                projected_y=py,
                obs_id=oid,
                p_match=pm,
            )
            cands_by_obs.setdefault(oid, []).append(cpos)
    else:
        for c in candidates:
            if c.obs_id is not None:
                cands_by_obs.setdefault(c.obs_id, []).append(c)

    # Sort observations chronologically: stable sort by timestamp then original obs_id
    sorted_obs = obs_df.sort_values(by=[time_col, id_col]).reset_index(drop=True)

    # Output storage indexed by obs_id
    results_by_obs_id: dict[int, dict[str, Any]] = {}

    # State tracking across Viterbi segments
    # Current active segment structures:
    #   segment_obs_ids: list of obs_ids in current continuous chain
    #   V: list of dicts: step_idx -> list of Viterbi log-probabilities
    #   B: list of dicts: step_idx -> list of backpointers (int index in previous step)
    #   step_candidates: list of lists of CandidatePosition
    segment_id = 0

    def finalize_segment(
        seg_obs_ids: list[int],
        V_steps: list[list[float]],
        B_steps: list[list[int | None]],
        cand_steps: list[list[CandidatePosition]],
        curr_seg_id: int,
    ) -> None:
        """Backtrack through completed Viterbi segment and record best predictions."""
        if not seg_obs_ids:
            return

        T_steps = len(seg_obs_ids)
        # Find best candidate at final step with deterministic tie-breaking
        last_V = V_steps[-1]
        best_idx = 0
        best_val = float("-inf")

        # Deterministic tie-breaking: prefer higher V, then higher p_match, then smaller arc_id
        for idx, val in enumerate(last_V):
            if val > best_val:
                best_val = val
                best_idx = idx
            elif abs(val - best_val) < 1e-9:
                curr_c = cand_steps[-1][idx]
                best_c = cand_steps[-1][best_idx]
                if (curr_c.p_match, -curr_c.arc_id) > (best_c.p_match, -best_c.arc_id):
                    best_val = val
                    best_idx = idx

        # Backtrack
        chosen_indices = [best_idx]
        curr_ptr = best_idx
        for t in range(T_steps - 1, 0, -1):
            prev_ptr = B_steps[t][curr_ptr]
            if prev_ptr is None:
                # Disconnection restart inside segment
                prev_V = V_steps[t - 1]
                prev_ptr = int(np.argmax(prev_V))
            chosen_indices.append(prev_ptr)
            curr_ptr = prev_ptr

        chosen_indices.reverse()

        for t, (oid, c_idx) in enumerate(zip(seg_obs_ids, chosen_indices)):
            c_pos = cand_steps[t][c_idx]
            results_by_obs_id[oid] = {
                "obs_id": oid,
                "predicted_arc_id": c_pos.arc_id,
                "match_probability": c_pos.p_match,
                "method": "viterbi_hmm",
                "segment_id": curr_seg_id,
                "candidate_count": len(cand_steps[t]),
            }

    curr_seg_obs_ids: list[int] = []
    curr_V: list[list[float]] = []
    curr_B: list[list[int | None]] = []
    curr_cands: list[list[CandidatePosition]] = []
    prev_time: float | None = None
    prev_pos: tuple[float, float] | None = None

    for _, row in sorted_obs.iterrows():
        oid = int(row[id_col])
        t_s = float(row[time_col])
        px = float(row[x_col])
        py = float(row[y_col])
        is_valid = bool(row[valid_col]) if valid_col else True

        # Check validity and candidate presence
        obs_cands = cands_by_obs.get(oid, [])

        if not is_valid or math.isnan(px) or math.isnan(py) or len(obs_cands) == 0:
            # Finalize any ongoing segment
            if curr_seg_obs_ids:
                finalize_segment(curr_seg_obs_ids, curr_V, curr_B, curr_cands, segment_id)
                segment_id += 1
                curr_seg_obs_ids, curr_V, curr_B, curr_cands = [], [], [], []
                prev_time, prev_pos = None, None

            # Record missing / invalid prediction row
            results_by_obs_id[oid] = {
                "obs_id": oid,
                "predicted_arc_id": None,
                "match_probability": 0.0,
                "method": "viterbi_hmm",
                "segment_id": -1,
                "candidate_count": len(obs_cands),
            }
            continue

        # Sort candidates deterministically by arc_id to eliminate any order instability
        obs_cands = sorted(obs_cands, key=lambda c: c.arc_id)
        total_p = sum(c.p_match for c in obs_cands)

        # Calculate emission log scores for all candidates
        emissions = [
            compute_emission_score(c.p_match, config.epsilon, total_p, config.normalize_emissions)
            for c in obs_cands
        ]

        # Check gap condition
        is_gap = (prev_time is not None) and ((t_s - prev_time) > config.max_gap_s)

        if not curr_seg_obs_ids or is_gap:
            # Start new segment
            if curr_seg_obs_ids and is_gap:
                finalize_segment(curr_seg_obs_ids, curr_V, curr_B, curr_cands, segment_id)
                segment_id += 1
                curr_seg_obs_ids, curr_V, curr_B, curr_cands = [], [], [], []

            curr_seg_obs_ids.append(oid)
            curr_V.append(emissions)
            curr_B.append([None] * len(obs_cands))
            curr_cands.append(obs_cands)
            prev_time = t_s
            prev_pos = (px, py)
        else:
            # Transition step from previous candidates
            obs_disp_m = math.hypot(px - prev_pos[0], py - prev_pos[1]) if prev_pos else 0.0
            prev_cands_list = curr_cands[-1]
            prev_V_list = curr_V[-1]

            step_V: list[float] = []
            step_B: list[int | None] = []
            all_transitions_invalid = True

            for k, cand_k in enumerate(obs_cands):
                best_score = float("-inf")
                best_prev = 0

                for j, cand_j in enumerate(prev_cands_list):
                    if math.isinf(prev_V_list[j]):
                        continue
                    t_score = compute_transition_score(
                        from_cand=cand_j,
                        to_cand=cand_k,
                        from_time_s=prev_time,
                        to_time_s=t_s,
                        obs_disp_m=obs_disp_m,
                        road_graph=road_graph,
                        config=config,
                    )
                    if math.isinf(t_score):
                        continue

                    candidate_total = prev_V_list[j] + t_score
                    if candidate_total > best_score:
                        best_score = candidate_total
                        best_prev = j
                        all_transitions_invalid = False
                    elif abs(candidate_total - best_score) < 1e-9:
                        # Deterministic tie-breaking
                        if cand_j.arc_id < prev_cands_list[best_prev].arc_id:
                            best_score = candidate_total
                            best_prev = j

                if math.isinf(best_score):
                    # Unreachable from any previous state
                    step_V.append(float("-inf"))
                    step_B.append(None)
                else:
                    step_V.append(best_score + emissions[k])
                    step_B.append(best_prev)

            # If ALL transitions into this observation are -inf, restart transparently!
            if all_transitions_invalid or all(math.isinf(v) for v in step_V):
                finalize_segment(curr_seg_obs_ids, curr_V, curr_B, curr_cands, segment_id)
                segment_id += 1
                curr_seg_obs_ids = [oid]
                curr_V = [emissions]
                curr_B = [[None] * len(obs_cands)]
                curr_cands = [obs_cands]
            else:
                curr_seg_obs_ids.append(oid)
                curr_V.append(step_V)
                curr_B.append(step_B)
                curr_cands.append(obs_cands)

            prev_time = t_s
            prev_pos = (px, py)

    # Finalize any remaining active segment
    if curr_seg_obs_ids:
        finalize_segment(curr_seg_obs_ids, curr_V, curr_B, curr_cands, segment_id)

    # Assemble output strictly in the original order of observations
    out_rows = []
    for _, row in obs_df.iterrows():
        oid = int(row[id_col])
        res = results_by_obs_id.get(oid, {
            "obs_id": oid,
            "predicted_arc_id": None,
            "match_probability": 0.0,
            "method": "viterbi_hmm",
            "segment_id": -1,
            "candidate_count": 0,
        })
        res["timestamp"] = row[time_col]
        res["is_valid"] = bool(row[valid_col]) if valid_col else True
        out_rows.append(res)

    out_df = pd.DataFrame(out_rows)
    return out_df


def decode_classifier_only(
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Greedy classifier-only top-1 candidate matching per observation.

    Selects the candidate with highest p_match for each observation independently.
    Guarantees exactly one output row per observation in original order.
    """
    obs_df = observations.copy()
    id_col = "obs_id" if "obs_id" in obs_df.columns else obs_df.index.name or "obs_id"
    if id_col not in obs_df.columns:
        obs_df[id_col] = np.arange(len(obs_df))
    time_col = "timestamp" if "timestamp" in obs_df.columns else obs_df.columns[2]
    valid_col = "is_valid" if "is_valid" in obs_df.columns else None

    cand_obs_col = "obs_id" if "obs_id" in candidates.columns else candidates.columns[0]
    cand_arc_col = "arc_id" if "arc_id" in candidates.columns else candidates.columns[1]
    p_col = "p_match" if "p_match" in candidates.columns else ("score" if "score" in candidates.columns else candidates.columns[-1])

    # Find highest scoring candidate per obs_id deterministically
    best_cands: dict[int, tuple[int, float, int]] = {}  # obs_id -> (arc_id, p_match, count)
    cand_counts = candidates.groupby(cand_obs_col).size().to_dict()

    for oid, group in candidates.groupby(cand_obs_col):
        # Sort by p_match descending, then arc_id ascending
        sorted_g = group.sort_values(by=[p_col, cand_arc_col], ascending=[False, True])
        top_row = sorted_g.iloc[0]
        best_cands[int(oid)] = (int(top_row[cand_arc_col]), float(top_row[p_col]), cand_counts.get(oid, len(group)))

    out_rows = []
    for _, row in obs_df.iterrows():
        oid = int(row[id_col])
        is_valid = bool(row[valid_col]) if valid_col else True
        if not is_valid or oid not in best_cands:
            out_rows.append({
                "obs_id": oid,
                "timestamp": row[time_col],
                "predicted_arc_id": None,
                "match_probability": 0.0,
                "method": "classifier_only",
                "segment_id": -1,
                "is_valid": is_valid,
                "candidate_count": cand_counts.get(oid, 0),
            })
        else:
            top_arc, top_p, count = best_cands[oid]
            out_rows.append({
                "obs_id": oid,
                "timestamp": row[time_col],
                "predicted_arc_id": top_arc,
                "match_probability": top_p,
                "method": "classifier_only",
                "segment_id": 0,
                "is_valid": is_valid,
                "candidate_count": count,
            })

    return pd.DataFrame(out_rows)
