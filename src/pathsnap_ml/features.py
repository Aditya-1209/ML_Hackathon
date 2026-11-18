"""Build the ordered, label-free classifier features."""

import numpy as np
import pandas as pd

from .candidates import KEY_COLUMNS, segment_geometry, valid_observations

try:
    from .contracts import FEATURE_COLUMNS
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.contracts":
        raise
    # Supports independent tests before the shared contracts branch is merged.
    FEATURE_COLUMNS = [
        "distance_m", "heading_difference_deg", "arc_length_m", "projection_fraction",
        "delta_t_s", "gps_step_m", "speed_mps", "heading_available",
    ]


def make_features(observations, nodes, arcs, candidates, metadata, config):
    valid = valid_observations(observations, metadata).set_index("obs_id")
    starts, ends = segment_geometry(nodes, arcs, metadata)
    if candidates.duplicated(KEY_COLUMNS).any() or not candidates.trip_id.eq(metadata["trip_id"]).all():
        raise ValueError("Invalid candidate keys")
    if not candidates.obs_id.isin(valid.index).all() or not candidates.candidate_arc_id.isin(arcs.arc_id).all():
        raise ValueError("Candidate references an unknown arc or invalid observation")
    min_step = config["heading_min_step_m"]
    max_gap = config["motion_max_gap_s"]
    if not np.isfinite([min_step, max_gap]).all() or min_step <= 0 or max_gap <= 0:
        raise ValueError("Expected positive finite motion thresholds")
    delta = valid.timestamp_s.diff()
    displacement = valid[["x_m", "y_m"]].diff()
    step = np.hypot(displacement.x_m, displacement.y_m)
    usable_time = delta.gt(0) & delta.le(max_gap)
    speed = step / delta.where(usable_time)
    gps_heading = np.rad2deg(np.arctan2(displacement.y_m, displacement.x_m))
    gps_heading = gps_heading.where(usable_time & step.ge(min_step))
    vector = ends - starts
    road_heading = np.rad2deg(np.arctan2(vector[:, 1], vector[:, 0]))
    road_heading[np.all(vector == 0, axis=1)] = np.nan
    road_headings = pd.Series(road_heading, index=arcs.arc_id)
    gps = candidates.obs_id.map(gps_heading)
    road = candidates.candidate_arc_id.map(road_headings)
    angle = ((gps - road + 180) % 360 - 180).abs()
    if not metadata["graph_directed"]:
        angle = np.minimum(angle, 180 - angle)

    features = candidates[KEY_COLUMNS + ["distance_m", "projection_fraction"]].copy()
    features["heading_difference_deg"] = angle
    features["arc_length_m"] = candidates.candidate_arc_id.map(arcs.set_index("arc_id").length_m)
    features["delta_t_s"] = candidates.obs_id.map(delta)
    features["gps_step_m"] = candidates.obs_id.map(step)
    features["speed_mps"] = candidates.obs_id.map(speed)
    features["heading_available"] = angle.notna().astype(float)
    columns = list(FEATURE_COLUMNS)
    features = features.astype({name: "float64" for name in columns})
    if np.isinf(features[columns].to_numpy()).any():
        raise ValueError("Infinite classifier feature")
    return features[KEY_COLUMNS + columns]

