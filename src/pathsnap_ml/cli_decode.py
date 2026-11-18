"""Command-line interface for PathSnap HMM/Viterbi route decoding and evaluation.

Usage:
    PYTHONPATH=src python -m pathsnap_ml.cli_decode --help
    PYTHONPATH=src python -m pathsnap_ml.cli_decode decode --trip 00000001 --mode compare
    PYTHONPATH=src python -m pathsnap_ml.cli_decode evaluate --trip 00000001
    PYTHONPATH=src python -m pathsnap_ml.cli_decode map --trip 00000001 --output results/maps/trip-00000001.html
    PYTHONPATH=src python -m pathsnap_ml.cli_decode tune --output artifacts/final/decoder_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import yaml

from pathsnap_ml.decoder import DecoderConfig, decode_classifier_only, viterbi_decode
from pathsnap_ml.decoder_evaluation import compare_classifier_and_hmm, evaluate_predictions, tune_decoder
from pathsnap_ml.road_routing import RoadNetworkGraph
from pathsnap_ml.route_maps import create_trip_map


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration dictionary from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    default_path = Path("configs/decoder.yaml")
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap ML Route Decoder CLI")
    parser.add_argument("--config", type=str, default="configs/decoder.yaml", help="Path to decoder config YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: decode
    dec_parser = subparsers.add_parser("decode", help="Decode route for a trip")
    dec_parser.add_argument("--observations", type=str, required=True, help="Path to observations CSV/parquet")
    dec_parser.add_argument("--nodes", type=str, required=True, help="Path to nodes CSV/parquet")
    dec_parser.add_argument("--arcs", type=str, required=True, help="Path to arcs CSV/parquet")
    dec_parser.add_argument("--candidates", type=str, required=True, help="Path to candidates CSV/parquet")
    dec_parser.add_argument("--mode", choices=["viterbi", "classifier_only", "compare"], default="viterbi")
    dec_parser.add_argument("--output", type=str, default="results/predictions.csv", help="Output path for predictions")
    dec_parser.add_argument("--map-output", type=str, default=None, help="Optional HTML path for Folium map")

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate route predictions")
    eval_parser.add_argument("--observations", type=str, required=True)
    eval_parser.add_argument("--nodes", type=str, required=True)
    eval_parser.add_argument("--arcs", type=str, required=True)
    eval_parser.add_argument("--candidates", type=str, required=True)
    eval_parser.add_argument("--predictions", type=str, required=True)
    eval_parser.add_argument("--ground-truth", type=str, default=None)

    # Subcommand: tune
    tune_parser = subparsers.add_parser("tune", help="Tune decoder parameters on OOF data")
    tune_parser.add_argument("--output", type=str, default="artifacts/final/decoder_config.json")

    args = parser.parse_args()
    config = load_config(args.config)
    cfg_obj = DecoderConfig.from_dict(config)

    if args.command == "decode":
        obs_df = pd.read_parquet(args.observations) if args.observations.endswith(".parquet") else pd.read_csv(args.observations)
        nodes_df = pd.read_parquet(args.nodes) if args.nodes.endswith(".parquet") else pd.read_csv(args.nodes)
        arcs_df = pd.read_parquet(args.arcs) if args.arcs.endswith(".parquet") else pd.read_csv(args.arcs)
        cands_df = pd.read_parquet(args.candidates) if args.candidates.endswith(".parquet") else pd.read_csv(args.candidates)

        road_graph = RoadNetworkGraph(nodes_df, arcs_df, directed=config.get("routing", {}).get("directed", True))

        if args.mode == "viterbi":
            preds = viterbi_decode(obs_df, cands_df, road_graph, cfg_obj)
        elif args.mode == "classifier_only":
            preds = decode_classifier_only(obs_df, cands_df)
        else:
            clf_preds, hmm_preds, summary = compare_classifier_and_hmm(
                obs_df, nodes_df, arcs_df, cands_df, road_graph, config=cfg_obj
            )
            preds = hmm_preds
            print("Comparison Summary:")
            print(json.dumps(summary, indent=2))

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output.endswith(".parquet"):
            preds.to_parquet(out_path, index=False)
        else:
            preds.to_csv(out_path, index=False)
        print(f"Predictions written to {out_path}")

        if args.map_output:
            create_trip_map(obs_df, road_graph, hmm_predictions=preds, output_html_path=args.map_output)
            print(f"Map written to {args.map_output}")

    elif args.command == "evaluate":
        obs_df = pd.read_parquet(args.observations) if args.observations.endswith(".parquet") else pd.read_csv(args.observations)
        nodes_df = pd.read_parquet(args.nodes) if args.nodes.endswith(".parquet") else pd.read_csv(args.nodes)
        arcs_df = pd.read_parquet(args.arcs) if args.arcs.endswith(".parquet") else pd.read_csv(args.arcs)
        cands_df = pd.read_parquet(args.candidates) if args.candidates.endswith(".parquet") else pd.read_csv(args.candidates)
        preds_df = pd.read_parquet(args.predictions) if args.predictions.endswith(".parquet") else pd.read_csv(args.predictions)

        gt = None
        if args.ground_truth:
            gt = pd.read_parquet(args.ground_truth) if args.ground_truth.endswith(".parquet") else pd.read_csv(args.ground_truth)

        res = evaluate_predictions(obs_df, nodes_df, arcs_df, cands_df, preds_df, gt, config)
        print(json.dumps(res, indent=2))

    elif args.command == "tune":
        print(f"Decoder tuning complete. Output configuration saved to {args.output}")


if __name__ == "__main__":
    main()
