"""Interactive Folium map generation for trajectory visualisations.

Follows PathSnap ML v1 specification:
- Visualises original GPS observations, road network segments, classifier-only predictions,
  and HMM reconstructed routes.
- Optional reference route layer for permitted offline diagnostic inspection.
- Clear visual hierarchy with distinct colours, tooltips, and legends.
- Disconnected runs / large gaps are split into separate polylines rather than false bridging lines.
- Label-free during normal inference; reference layer only added when explicitly provided.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import folium
import numpy as np
import pandas as pd

from pathsnap_ml.road_routing import RoadNetworkGraph


def create_trip_map(
    observations: pd.DataFrame,
    road_graph: RoadNetworkGraph,
    hmm_predictions: pd.DataFrame | None = None,
    classifier_predictions: pd.DataFrame | None = None,
    reference_arcs: Sequence[int] | None = None,
    output_html_path: str | Path | None = None,
    transformer: Any = None,
    title: str = "PathSnap Trajectory Reconstruction",
) -> folium.Map:
    """Generate an interactive Folium map for a trip.

    Args:
        observations: DataFrame with GPS observations (requires 'x', 'y' or 'lon', 'lat', and 'obs_id').
        road_graph: RoadNetworkGraph instance.
        hmm_predictions: Optional DataFrame of HMM Viterbi predictions.
        classifier_predictions: Optional DataFrame of classifier-only predictions.
        reference_arcs: Optional sequence of ground-truth arc IDs (offline diagnostic only).
        output_html_path: Optional path to save the generated HTML file.
        transformer: Optional pyproj Transformer to convert (x, y) metres to (lat, lon).
        title: Map title.

    Returns:
        folium.Map instance.
    """
    obs_df = observations.copy()
    x_col = "x" if "x" in obs_df.columns else ("lon" if "lon" in obs_df.columns else obs_df.columns[0])
    y_col = "y" if "y" in obs_df.columns else ("lat" if "lat" in obs_df.columns else obs_df.columns[1])
    id_col = "obs_id" if "obs_id" in obs_df.columns else obs_df.columns[0]
    time_col = "timestamp" if "timestamp" in obs_df.columns else None

    # Coordinate conversion helper: returns (lat, lon)
    def to_latlon(x_val: float, y_val: float) -> tuple[float, float]:
        if transformer is not None:
            lon, lat = transformer.transform(x_val, y_val)
            return float(lat), float(lon)
        # Check if coordinates look like UTM metres (> 1000) or degrees
        if abs(x_val) > 180 or abs(y_val) > 90:
            # Approximated if no transformer provided
            return float(y_val / 111000.0), float(x_val / 111000.0)
        return float(y_val), float(x_val)

    # Compute bounding box and center
    latlons = [to_latlon(float(r[x_col]), float(r[y_col])) for _, r in obs_df.iterrows()]
    if not latlons:
        center_lat, center_lon = 0.0, 0.0
    else:
        center_lat = float(np.mean([p[0] for p in latlons]))
        center_lon = float(np.mean([p[1] for p in latlons]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    # 1. Road Network Layer (gray lines)
    road_layer = folium.FeatureGroup(name="Road Network", show=True)
    # Collect arcs within bounds
    for aid, arc in road_graph.arcs.items():
        p1 = to_latlon(arc.from_x, arc.from_y)
        p2 = to_latlon(arc.to_x, arc.to_y)
        folium.PolyLine(
            locations=[p1, p2],
            color="#888888",
            weight=2,
            opacity=0.6,
            tooltip=f"Arc {aid} (Nodes: {arc.from_node_id} -> {arc.to_node_id}, {arc.length_m:.1f}m)",
        ).add_to(road_layer)
    road_layer.add_to(fmap)

    # 2. Reference Route Layer (Green line, diagnostic only)
    if reference_arcs is not None:
        ref_layer = folium.FeatureGroup(name="Reference Route (Diagnostic)", show=True)
        ref_coords = []
        for aid in reference_arcs:
            if aid in road_graph.arcs:
                arc = road_graph.arcs[aid]
                p1 = to_latlon(arc.from_x, arc.from_y)
                p2 = to_latlon(arc.to_x, arc.to_y)
                ref_coords.append(p1)
                ref_coords.append(p2)
        if ref_coords:
            folium.PolyLine(
                locations=ref_coords,
                color="#28a745",
                weight=5,
                opacity=0.8,
                tooltip="Ground-Truth Reference Route",
            ).add_to(ref_layer)
        ref_layer.add_to(fmap)

    # 3. Classifier-Only Route Layer (Orange dashed line)
    if classifier_predictions is not None:
        clf_layer = folium.FeatureGroup(name="Classifier-Only Route", show=True)
        pred_col = "predicted_arc_id" if "predicted_arc_id" in classifier_predictions.columns else "arc_id"
        clf_coords = []
        for _, r in classifier_predictions.iterrows():
            aid = r[pred_col]
            if pd.notna(aid) and int(aid) in road_graph.arcs:
                arc = road_graph.arcs[int(aid)]
                mid_x = (arc.from_x + arc.to_x) / 2.0
                mid_y = (arc.from_y + arc.to_y) / 2.0
                clf_coords.append(to_latlon(mid_x, mid_y))
        if clf_coords:
            folium.PolyLine(
                locations=clf_coords,
                color="#fd7e14",
                weight=3,
                opacity=0.85,
                dash_array="5, 10",
                tooltip="Classifier-Only Greedy Route",
            ).add_to(clf_layer)
        clf_layer.add_to(fmap)

    # 4. HMM Viterbi Reconstructed Route (Purple line, segmented by continuous runs)
    if hmm_predictions is not None:
        hmm_layer = folium.FeatureGroup(name="HMM Viterbi Reconstructed Route", show=True)
        pred_col = "predicted_arc_id" if "predicted_arc_id" in hmm_predictions.columns else "arc_id"
        seg_col = "segment_id" if "segment_id" in hmm_predictions.columns else None

        # Group by segment_id to avoid drawing lines across large gaps or disconnections
        if seg_col:
            groups = hmm_predictions.groupby(seg_col)
        else:
            groups = [(0, hmm_predictions)]

        for seg_id, grp in groups:
            if seg_id == -1:
                continue
            seg_coords = []
            for _, r in grp.iterrows():
                aid = r[pred_col]
                if pd.notna(aid) and int(aid) in road_graph.arcs:
                    arc = road_graph.arcs[int(aid)]
                    mid_x = (arc.from_x + arc.to_x) / 2.0
                    mid_y = (arc.from_y + arc.to_y) / 2.0
                    seg_coords.append(to_latlon(mid_x, mid_y))
            if len(seg_coords) >= 2:
                folium.PolyLine(
                    locations=seg_coords,
                    color="#6f42c1",
                    weight=5,
                    opacity=0.9,
                    tooltip=f"HMM Reconstructed Segment {seg_id}",
                ).add_to(hmm_layer)
        hmm_layer.add_to(fmap)

    # 5. GPS Observations Layer (Blue circle markers)
    gps_layer = folium.FeatureGroup(name="GPS Observations", show=True)
    for idx, (lat, lon) in enumerate(latlons):
        row = obs_df.iloc[idx]
        oid = row[id_col]
        t_str = f", Time: {row[time_col]}s" if time_col else ""
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#007bff",
            fill=True,
            fill_color="#007bff",
            fill_opacity=0.8,
            tooltip=f"Observation {oid}{t_str}",
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # Add Legend HTML
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 260px; z-index:9999; font-size:12px;
                background-color: white; padding: 10px; border: 2px solid #ccc; border-radius: 6px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
        <h5 style="margin-top:0; margin-bottom:8px; font-weight:bold;">{title}</h5>
        <div><i style="background: #007bff; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> GPS Observations</div>
        <div><i style="background: #888888; width: 16px; height: 3px; display: inline-block; margin-right: 6px;"></i> Road Network</div>
        <div><i style="background: #fd7e14; width: 16px; height: 3px; display: inline-block; margin-right: 6px; border-top: 2px dashed #fd7e14;"></i> Classifier-Only</div>
        <div><i style="background: #6f42c1; width: 16px; height: 4px; display: inline-block; margin-right: 6px;"></i> HMM / Viterbi</div>
        {"<div><i style='background: #28a745; width: 16px; height: 4px; display: inline-block; margin-right: 6px;'></i> Reference Route (Diagnostic)</div>" if reference_arcs is not None else ""}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(fmap)

    if output_html_path:
        out_path = Path(output_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_path))

    return fmap
