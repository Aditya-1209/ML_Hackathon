import numpy as np
import pandas as pd

from .contracts import FEATURES, KEY, validate_features


def make_features(observations, nodes, arcs, candidates, metadata, config):
    motion = observations[observations.valid].sort_values("order_index").copy()
    motion["delta_t_s"] = motion.timestamp_s.diff()
    dx, dy = motion.x_m.diff(), motion.y_m.diff()
    motion["gps_step_m"] = np.hypot(dx, dy)
    usable = (motion.delta_t_s.gt(0) & motion.delta_t_s.le(config["heading_max_gap_s"])
              & motion.gps_step_m.ge(config["heading_min_step_m"]))
    motion["gps_heading"] = np.where(usable, np.degrees(np.arctan2(dx, dy)), np.nan)
    motion["speed_mps"] = np.where(motion.delta_t_s.gt(0),
                                   motion.gps_step_m / motion.delta_t_s, np.nan)
    motion["heading_available"] = usable.astype(float)
    xy = nodes.set_index("node_id")[["x_m", "y_m"]]
    delta = xy.loc[arcs.to_node_id].to_numpy() - xy.loc[arcs.from_node_id].to_numpy()
    roads = pd.DataFrame({"trip_id": arcs.trip_id, "candidate_arc_id": arcs.arc_id,
                          "arc_length_m": arcs.length_m,
                          "road_heading": np.degrees(np.arctan2(delta[:, 0], delta[:, 1]))})
    roads.loc[roads.arc_length_m.eq(0), "road_heading"] = np.nan
    result = candidates.merge(motion[["trip_id", "obs_id", "delta_t_s", "gps_step_m",
                                      "speed_mps", "heading_available", "gps_heading"]],
                              on=["trip_id", "obs_id"], validate="many_to_one")
    result = result.merge(roads, on=["trip_id", "candidate_arc_id"], validate="many_to_one")
    diff = abs((result.gps_heading - result.road_heading + 180) % 360 - 180)
    result["heading_difference_deg"] = diff if metadata["graph_directed"] else np.minimum(diff, 180-diff)
    result.loc[result.road_heading.isna(), "heading_available"] = 0.
    result = result[KEY + FEATURES].astype({c: "float64" for c in FEATURES})
    validate_features(result)
    return result
