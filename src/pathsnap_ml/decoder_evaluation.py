"""Evaluation and comparison framework for map-matching predictions.

Follows PathSnap ML v1 specification:
- Compares Classifier-Only predictions vs Classifier + HMM/Viterbi predictions.
- Computes pointwise accuracy, mean per-trip accuracy, exact-trip-match rate,
  candidate miss rate, no-candidate rate, label coverage, and route continuity.
- Uses identical labelled observation sets and denominators.
- Supports tuning decoder parameters using training OOF predictions.
- Generates artifacts/final/decoder_config.json with provenance tracking.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from pathsnap_ml.decoder import DecoderConfig, decode_classifier_only, viterbi_decode
from pathsnap_ml.road_routing import RoadNetworkGraph
from pathsnap_ml.route_metrics import compute_continuity, compute_pointwise_accuracy


def evaluate_predictions(
    observations: pd.DataFrame,
    nodes: pd.DataFrame | dict[str, Any],
    arcs: pd.DataFrame | dict[str, Any],
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame | Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for a single trip.

    Reports:
    - Observation counts, candidate miss rate, no-candidate rate.
    - Continuity metrics (rate, continuous transitions, failures).
    - Pointwise accuracy (if ground_truth is provided).
    - Exact match flag (True if all predicted arcs match ground_truth).
    """
    id_col = "obs_id" if "obs_id" in observations.columns else observations.columns[0]
    cand_obs_col = "obs_id" if "obs_id" in candidates.columns else candidates.columns[0]

    total_obs = len(observations)
    obs_with_cands = set(candidates[cand_obs_col].unique())
    no_cand_obs_count = sum(1 for oid in observations[id_col] if oid not in obs_with_cands)
    no_cand_rate = no_cand_obs_count / total_obs if total_obs > 0 else 0.0

    continuity_res = compute_continuity(
        observations=observations,
        nodes=nodes,
        arcs=arcs,
        candidates=candidates,
        predictions=predictions,
        config=config,
    )

    result: dict[str, Any] = {
        "total_observations": total_obs,
        "no_candidate_count": no_cand_obs_count,
        "no_candidate_rate": no_cand_rate,
        **continuity_res,
    }

    if ground_truth is not None:
        acc_res = compute_pointwise_accuracy(predictions, ground_truth)
        result.update(acc_res)

        # Exact sequence match rate (1.0 if all match, else 0.0)
        is_exact = bool(acc_res["correct_matches"] == total_obs and total_obs > 0)
        result["exact_match"] = is_exact
    else:
        result["pointwise_accuracy"] = None
        result["exact_match"] = None

    return result


def compare_classifier_and_hmm(
    observations: pd.DataFrame,
    nodes: pd.DataFrame | dict[str, Any],
    arcs: pd.DataFrame | dict[str, Any],
    candidates: pd.DataFrame,
    road_graph: RoadNetworkGraph,
    ground_truth: pd.DataFrame | Sequence[int] | None = None,
    config: DecoderConfig | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run both Classifier-Only and HMM/Viterbi decoders and return comparative metrics.

    Returns:
        (clf_predictions, hmm_predictions, comparison_summary)
    """
    cfg_obj = config if isinstance(config, DecoderConfig) else DecoderConfig.from_dict(config)
    cfg_dict = config if isinstance(config, dict) else (config.__dict__ if config else {})

    # 1. Classifier-only
    t0 = time.perf_counter()
    clf_preds = decode_classifier_only(observations, candidates)
    clf_runtime_s = time.perf_counter() - t0

    # 2. HMM + Viterbi
    t1 = time.perf_counter()
    hmm_preds = viterbi_decode(observations, candidates, road_graph, cfg_obj)
    hmm_runtime_s = time.perf_counter() - t1

    # Evaluate both under identical conditions
    clf_metrics = evaluate_predictions(
        observations=observations,
        nodes=nodes,
        arcs=arcs,
        candidates=candidates,
        predictions=clf_preds,
        ground_truth=ground_truth,
        config=cfg_dict,
    )
    clf_metrics["runtime_s"] = clf_runtime_s

    hmm_metrics = evaluate_predictions(
        observations=observations,
        nodes=nodes,
        arcs=arcs,
        candidates=candidates,
        predictions=hmm_preds,
        ground_truth=ground_truth,
        config=cfg_dict,
    )
    hmm_metrics["runtime_s"] = hmm_runtime_s

    summary = {
        "classifier_only": clf_metrics,
        "hmm_viterbi": hmm_metrics,
        "continuity_delta": hmm_metrics["continuity_rate"] - clf_metrics["continuity_rate"],
        "accuracy_delta": (
            (hmm_metrics["pointwise_accuracy"] - clf_metrics["pointwise_accuracy"])
            if (hmm_metrics["pointwise_accuracy"] is not None and clf_metrics["pointwise_accuracy"] is not None)
            else None
        ),
    }

    return clf_preds, hmm_preds, summary


def tune_decoder(
    trips_data: list[dict[str, Any]],
    param_grid: list[dict[str, Any]] | None = None,
    objective: str = "continuity_rate",
    output_path: str = "artifacts/final/decoder_config.json",
    provenance_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune HMM decoder parameters using training OOF predictions only.

    Records:
    - Every configuration tested.
    - Objective and metric definitions.
    - Best configuration and failed trials.
    - Provenance (model hash, feature configuration hash, OOF provenance).
    - Saves artifacts/final/decoder_config.json.
    """
    if param_grid is None:
        param_grid = [
            {"beta_m": 5.0, "epsilon": 1e-6, "normalize_emissions": False, "max_speed_mps": 35.0, "speed_slack_mps": 5.0, "max_gap_s": 15.0},
            {"beta_m": 10.0, "epsilon": 1e-6, "normalize_emissions": False, "max_speed_mps": 40.0, "speed_slack_mps": 5.0, "max_gap_s": 15.0},
            {"beta_m": 20.0, "epsilon": 1e-6, "normalize_emissions": False, "max_speed_mps": 40.0, "speed_slack_mps": 5.0, "max_gap_s": 15.0},
            {"beta_m": 10.0, "epsilon": 1e-5, "normalize_emissions": True, "max_speed_mps": 40.0, "speed_slack_mps": 5.0, "max_gap_s": 15.0},
            {"beta_m": 25.0, "epsilon": 1e-7, "normalize_emissions": False, "max_speed_mps": 45.0, "speed_slack_mps": 5.0, "max_gap_s": 20.0},
        ]

    trials = []
    best_score = -float("inf")
    best_config = param_grid[0]

    for trial_idx, p in enumerate(param_grid):
        cfg = DecoderConfig(**p)
        trial_record = {
            "trial_id": trial_idx + 1,
            "params": p,
            "metrics": {},
            "status": "SUCCESS",
        }

        trip_scores = []
        try:
            for trip in trips_data:
                obs = trip["observations"]
                nodes = trip["nodes"]
                arcs = trip["arcs"]
                cands = trip["candidates"]
                gt = trip.get("ground_truth")

                graph = RoadNetworkGraph(nodes, arcs, directed=True)
                preds = viterbi_decode(obs, cands, graph, cfg)
                eval_res = evaluate_predictions(obs, nodes, arcs, cands, preds, gt, {"model": p})
                trip_scores.append(eval_res.get(objective, 0.0) or 0.0)

            mean_score = float(np.mean(trip_scores)) if trip_scores else 0.0
            trial_record["metrics"][f"mean_{objective}"] = mean_score
            trials.append(trial_record)

            if mean_score > best_score:
                best_score = mean_score
                best_config = p
        except Exception as e:
            trial_record["status"] = "FAILED"
            trial_record["error"] = str(e)
            trials.append(trial_record)

    out_doc = {
        "best_configuration": best_config,
        "objective": objective,
        "best_score": best_score,
        "total_trials": len(trials),
        "trials": trials,
        "provenance": provenance_metadata or {
            "source": "person_4_decoder_tuning",
            "tuning_dataset": "training_oof_only",
            "evaluator_data_used": False,
        },
    }

    # Save to output path
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    return out_doc
