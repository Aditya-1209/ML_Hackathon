import numpy as np
from .road_routing import Router


def compute_continuity(observations,nodes,arcs,candidates,predictions,metadata,config):
    router = Router(nodes,arcs,metadata)
    frames = candidates.merge(predictions.dropna(subset=["predicted_arc_id"])[["trip_id","obs_id","predicted_arc_id"]],
                               left_on=["trip_id","obs_id","candidate_arc_id"],
                               right_on=["trip_id","obs_id","predicted_arc_id"])
    chosen = {row.obs_id:row for row in frames.itertuples()}
    valid = list(observations[observations.valid].sort_values("order_index").itertuples())
    invalid = observations[~observations.valid].obs_id.to_numpy()
    feasible=eligible=large_gaps=invalid_breaks=0
    for left,right in zip(valid,valid[1:]):
        if ((invalid>min(left.obs_id,right.obs_id)) & (invalid<max(left.obs_id,right.obs_id))).any():
            invalid_breaks += 1
            continue
        dt = right.timestamp_s-left.timestamp_s
        if dt>config["gap_s"]:
            large_gaps += 1
            continue
        eligible += 1
        if left.obs_id in chosen and right.obs_id in chosen:
            distance,_ = router.between(chosen[left.obs_id],chosen[right.obs_id],
                                         max(dt,0)*config["max_speed_mps"]+config["slack_m"])
            feasible += int(np.isfinite(distance))
    return dict(feasible=feasible,eligible=eligible,excluded_large_gap_pairs=large_gaps,
                invalid_observation_breaks=invalid_breaks,
                continuity=feasible/eligible if eligible else None)
