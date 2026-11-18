"""Offline proxy labels from an ordered reference route."""

import numpy as np
import pandas as pd

from .candidates import project, segment_geometry, valid_observations

LABEL_DTYPES = {
    "trip_id": "string", "obs_id": "int64", "target_arc_id": "Int64",
    "target_route_pos": "Int64", "label_status": "string", "label_source": "string",
}


def _unique_route_positions(distances, max_distance, margin):
    """Find uniquely plausible positions using one backward row at a time."""
    penalty = len(distances) * max_distance + margin + 1
    forward = np.empty_like(distances)
    prefix = np.zeros(distances.shape[1])
    for i, row in enumerate(distances):
        forward[i] = np.where(row <= max_distance, row, penalty) + prefix
        prefix = np.minimum.accumulate(forward[i])
    best = prefix[-1]
    positions = np.full(len(distances), -1, dtype=np.int64)
    if best >= penalty:
        return positions

    suffix = np.zeros(distances.shape[1])
    for i in range(len(distances) - 1, -1, -1):
        row = distances[i]
        plausible = np.flatnonzero((forward[i] + suffix <= best + margin) & (row <= max_distance))
        if len(plausible) == 1:
            positions[i] = plausible[0]
        backward = np.where(row <= max_distance, row, penalty) + suffix
        suffix = np.minimum.accumulate(backward[::-1])[::-1]
    return positions


def align_reference(observations, nodes, arcs, reference, metadata, config):
    """Align to the full route independently of the prediction candidate set."""
    if metadata["split"] not in {"train", "student_test"}:
        raise ValueError("Reference alignment is for development trips only")
    valid = valid_observations(observations, metadata)
    starts, ends = segment_geometry(nodes, arcs, metadata)
    labels = pd.DataFrame({
        "trip_id": observations.trip_id,
        "obs_id": observations.obs_id,
        "target_arc_id": pd.NA,
        "target_route_pos": pd.NA,
        "label_status": "unavailable",
        "label_source": "none",
    }).astype(LABEL_DTYPES).set_index("obs_id")
    if reference is None or reference.empty or valid.empty:
        return labels.reset_index()[list(LABEL_DTYPES)]

    if not reference.trip_id.eq(metadata["trip_id"]).all():
        raise ValueError("Reference trip ID mismatch")
    route = reference.sort_values("route_pos")
    if route.route_pos.tolist() != list(range(len(route))):
        raise ValueError("Reference route positions must be consecutive from zero")
    index = pd.Index(arcs.arc_id).get_indexer(route.arc_id)
    if (index < 0).any():
        raise ValueError("Reference route contains an unknown arc")
    max_distance = config["max_reference_distance_m"]
    margin = config["ambiguity_cost_margin_m"]
    gap = config["alignment_gap_s"]
    cell_limit = config["max_alignment_cells"]
    if not np.isfinite([max_distance, margin, gap]).all() or max_distance <= 0 or margin < 0 or gap <= 0:
        raise ValueError("Invalid alignment thresholds")
    if type(cell_limit) is not int or cell_limit <= 0:
        raise ValueError("Alignment cell budget must be a positive integer")

    route_starts, route_ends = starts[index], ends[index]
    route_arcs = route.arc_id.to_numpy()
    breaks = valid.timestamp_s.diff().gt(gap).cumsum()
    for _, run in valid.groupby(breaks):
        if len(run) * len(route) > cell_limit:
            raise ValueError("Reference alignment run exceeds max_alignment_cells")
        coordinates = run[["x_m", "y_m"]].to_numpy(dtype=float)
        distances = np.empty((len(run), len(route)))
        for i, point in enumerate(coordinates):
            distances[i] = project(point, route_starts, route_ends)[0]
        near = distances.min(axis=1) <= max_distance
        if not near.any():
            continue
        eligible = run.loc[near]
        if not near.all():
            distances = distances[near]
        positions = _unique_route_positions(distances, max_distance, margin)
        labels.loc[eligible.obs_id, "label_source"] = "reference_alignment"
        for obs_id, position in zip(eligible.obs_id, positions):
            if position < 0:
                labels.loc[obs_id, "label_status"] = "ambiguous"
                continue
            labels.at[obs_id, "target_arc_id"] = int(route_arcs[position])
            labels.at[obs_id, "target_route_pos"] = int(position)
            labels.at[obs_id, "label_status"] = "assigned"
    return labels.reset_index()[list(LABEL_DTYPES)]


def coverage_summary(observations, candidates, labels=None):
    valid_ids = observations.loc[observations.valid, "obs_id"]
    valid_count = len(valid_ids)
    candidate_count = valid_ids.isin(candidates.obs_id).sum()
    result = {
        "observations": len(observations), "valid_observations": valid_count,
        "candidate_rows": len(candidates),
        "no_candidate_observations": int(valid_count - candidate_count),
        "no_candidate_rate": float(1 - candidate_count / valid_count) if valid_count else None,
    }
    if labels is None:
        return result
    valid_labels = labels.loc[labels.obs_id.isin(valid_ids)]
    assigned = valid_labels.loc[valid_labels.label_status.eq("assigned")]
    target_keys = pd.MultiIndex.from_frame(assigned[["obs_id", "target_arc_id"]])
    candidate_keys = pd.MultiIndex.from_frame(candidates[["obs_id", "candidate_arc_id"]])
    misses = int((~target_keys.isin(candidate_keys)).sum())
    result.update({
        "assigned_observations": len(assigned),
        "ambiguous_observations": int(valid_labels.label_status.eq("ambiguous").sum()),
        "unavailable_valid_observations": int(valid_labels.label_status.eq("unavailable").sum()),
        "label_coverage": len(assigned) / valid_count if valid_count else None,
        "candidate_misses": misses,
        "candidate_miss_rate": misses / len(assigned) if len(assigned) else None,
    })
    return result


def write_alignment_svg(path, observations, nodes, arcs, metadata, reference, labels):
    """Draw development GPS label statuses against the full reference geometry."""
    valid = valid_observations(observations, metadata)
    if valid.empty:
        return
    starts, ends = segment_geometry(nodes, arcs, metadata)
    indices = pd.Index(arcs.arc_id).get_indexer(reference.sort_values("route_pos").arc_id)
    gps = valid[["x_m", "y_m"]].to_numpy()
    extent = np.vstack([starts[indices], ends[indices], gps])
    minimum, maximum = extent.min(axis=0), extent.max(axis=0)
    scale = min(740 / max(maximum[0] - minimum[0], 1),
                440 / max(maximum[1] - minimum[1], 1))

    def xy(point):
        x, y = (point - minimum) * scale
        return f"{x + 30:.2f},{495 - y:.2f}"

    status = labels.set_index("obs_id").loc[valid.obs_id, "label_status"]
    colors = {"assigned": "#155fd1", "ambiguous": "#af5300", "unavailable": "#963a80"}
    segments = " ".join(f"M{xy(starts[i])} L{xy(ends[i])}" for i in indices)
    dots = []
    for point, value in zip(gps, status):
        x, y = xy(point).split(",")
        dots.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{colors[value]}"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 520">'
        '<rect width="800" height="520" fill="white"/>'
        '<text x="30" y="28" font-family="sans-serif" font-size="15">'
        'Reference (green) · Assigned (blue) · Ambiguous (orange) · Unavailable (purple)</text>'
        f'<path d="{segments}" stroke="#348252" stroke-width="3" fill="none"/>'
        + "".join(dots) + "</svg>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
