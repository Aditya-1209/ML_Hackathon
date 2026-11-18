"""Offline targets inferred from ordered reference arcs. These are proxy labels."""
import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from .candidates import segment_geometries


def align_reference(observations, nodes, arcs, reference, config):
    roads = arcs.set_index("arc_id")
    geometry = segment_geometries(nodes, arcs)
    arc_indices = pd.Series(np.arange(len(arcs)), index=arcs.arc_id)
    route_ids = reference.sort_values("route_pos").arc_id.to_numpy()
    route = geometry[arc_indices.loc[route_ids].to_numpy()]
    lengths = shapely.length(route)
    starts = np.r_[0., np.cumsum(lengths)[:-1]]
    tree = STRtree(route)
    rows = []
    run, costs, back = [], [], []

    def finish():
        if not run:
            return
        state = int(np.argmin(costs[-1]))
        chosen = []
        for t in range(len(run)-1, -1, -1):
            chosen.append(state)
            if t:
                state = int(back[t][state])
        for item, chosen_state in zip(run, reversed(chosen)):
            obs, positions, distances, progress = item
            idx = int(positions[chosen_state])
            distance = distances[chosen_state]
            alternatives = (route_ids[positions] != route_ids[idx]) & (abs(distances-distance) <= config["ambiguous_distance_margin_m"])
            if distance > config["max_assigned_distance_m"]:
                status = "unavailable"
            elif alternatives.any():
                status = "ambiguous"
            else:
                status = "assigned"
            rows.append((obs.trip_id, obs.obs_id, int(route_ids[idx]) if status == "assigned" else None,
                         idx if status == "assigned" else None, status, "reference_alignment"))
        run.clear()
        costs.clear()
        back.clear()

    for obs in observations[observations.valid].sort_values("order_index").itertuples():
        point = shapely.Point(obs.x_m, obs.y_m)
        positions = tree.query(point, predicate="dwithin", distance=config["search_radius_m"])
        if not len(positions):
            finish()
            rows.append((obs.trip_id, obs.obs_id, None, None, "unavailable", "reference_alignment"))
            continue
        distances = shapely.distance(route[positions], point)
        order = np.lexsort((positions, distances))[:config["max_reference_candidates"]]
        positions, distances = positions[order], distances[order]
        along = shapely.line_locate_point(route[positions], point)
        progress = starts[positions] + along
        emission = distances / 20.
        if run:
            previous, old_positions, _, old_progress = run[-1]
            dt = obs.timestamp_s - previous.timestamp_s
            delta = progress[None, :] - old_progress[:, None]
            transition = np.zeros(delta.shape)
            if dt <= config["gap_s"]:
                moved = np.hypot(obs.x_m-previous.x_m, obs.y_m-previous.y_m)
                transition = abs(delta-moved) / config["transition_scale_m"]
            transition[positions[None, :] < old_positions[:, None]] = np.inf
            total = costs[-1][:, None] + transition
            parent = total.argmin(axis=0)
            score = total[parent, np.arange(len(positions))] + emission
            if np.isfinite(score).any():
                run.append((obs, positions, distances, progress))
                costs.append(score - np.nanmin(score))
                back.append(parent)
                continue
            finish()
        run.append((obs, positions, distances, progress))
        costs.append(emission)
        back.append(np.full(len(positions), -1))
    finish()
    result = pd.DataFrame(rows, columns=["trip_id","obs_id","target_arc_id","target_route_pos","label_status","label_source"])
    result = observations[["trip_id","obs_id"]].merge(result, on=["trip_id","obs_id"], how="left", validate="one_to_one")
    result["label_status"] = result.label_status.fillna("unavailable")
    result["label_source"] = result.label_source.fillna("none")
    return result.astype({"trip_id":"string","obs_id":"int64","target_arc_id":"Int64","target_route_pos":"Int64"})
