import numpy as np
import pandas as pd

from .contracts import KEY, score_keys
from .road_routing import Router


def decode_trip(observations,nodes,arcs,candidates,scores,metadata,config,mode="hmm"):
    score_keys(candidates,scores)
    joined = candidates.merge(scores[KEY+["p_match"]],on=KEY,validate="one_to_one")
    groups = {k:v.sort_values("candidate_arc_id").reset_index(drop=True) for k,v in joined.groupby("obs_id")}
    router = Router(nodes,arcs,metadata)
    chosen = {}
    run, values, back = [], [], []
    breaks = {}

    def finish():
        if not run:
            return
        state = int(np.argmax(values[-1]))
        for t in range(len(run)-1,-1,-1):
            obs,frame = run[t]
            row = frame.iloc[state]
            chosen[obs.obs_id] = row
            if t:
                state = int(back[t][state])
        run.clear()
        values.clear()
        back.clear()

    ordered = observations.sort_values(["order_index","obs_id"],na_position="last")
    # Invalid rows break chronological runs at their raw position; the input's raw order is
    # additionally checked below so a missing timestamp cannot create a silent connection.
    invalid_ids = observations[~observations.valid].obs_id.to_numpy()
    for obs in ordered.itertuples():
        if not obs.valid or obs.obs_id not in groups:
            finish()
            breaks[obs.obs_id] = "invalid_observation" if not obs.valid else "no_candidates"
            continue
        frame = groups[obs.obs_id]
        emission = np.log(np.maximum(frame.p_match.to_numpy(),config["epsilon"]))
        if run:
            previous,old = run[-1]
            dt = obs.timestamp_s-previous.timestamp_s
            invalid_between = ((invalid_ids>min(obs.obs_id,previous.obs_id)) & (invalid_ids<max(obs.obs_id,previous.obs_id))).any()
            if dt>config["gap_s"] or invalid_between:
                finish()
                breaks[obs.obs_id] = "invalid_observation" if invalid_between else "gps_gap"
            elif mode=="hmm":
                displacement = np.hypot(obs.x_m-previous.x_m,obs.y_m-previous.y_m)
                total = np.full((len(old),len(frame)),-np.inf)
                limit = config["max_speed_mps"]*max(0,dt)+config["slack_m"]
                for i,left in enumerate(old.itertuples()):
                    for j,right in enumerate(frame.itertuples()):
                        distance,_ = router.between(left,right,limit)
                        if np.isfinite(distance):
                            total[i,j] = values[-1][i]-abs(distance-displacement)/config["beta_m"]
                parents = total.argmax(axis=0)
                scores_next = total[parents,np.arange(len(frame))]+emission
                if np.isfinite(scores_next).any():
                    run.append((obs,frame)); values.append(scores_next-np.max(scores_next)); back.append(parents)
                    continue
                finish()
                breaks[obs.obs_id] = "unreachable_transition"
            else:
                finish()
        if mode=="classifier_only":
            # Finish each independent choice immediately.
            run.append((obs,frame)); values.append(emission); back.append(np.full(len(frame),-1)); finish()
        else:
            if obs.obs_id not in breaks:
                breaks[obs.obs_id] = "start"
            run.append((obs,frame)); values.append(emission); back.append(np.full(len(frame),-1))
    finish()
    rows,runs,current,previous = [],[],None,None
    for obs in ordered.itertuples():
        row = chosen.get(obs.obs_id)
        if row is None:
            reason = "invalid_observation" if not obs.valid else "no_candidates"
            rows.append((obs.trip_id,obs.obs_id,None,None,reason,True,reason))
            current,previous = None,None
            continue
        reason = breaks.get(obs.obs_id)
        connecting = [int(row.candidate_arc_id)]
        if previous is not None and reason is None:
            old_obs,old_row = previous
            dt = obs.timestamp_s-old_obs.timestamp_s
            invalid_between = ((invalid_ids>min(obs.obs_id,old_obs.obs_id)) & (invalid_ids<max(obs.obs_id,old_obs.obs_id))).any()
            if dt>config["gap_s"] or invalid_between:
                reason = "invalid_observation" if invalid_between else "gps_gap"
            else:
                distance,connecting = router.between(old_row,row,config["max_speed_mps"]*max(dt,0)+config["slack_m"])
                if not np.isfinite(distance):
                    reason = "unreachable_transition"
                    connecting = [int(row.candidate_arc_id)]
        if current is None or reason is not None:
            reason = reason or "start"
            current = {"obs_ids":[],"arc_ids":[]}
            runs.append(current)
        current["obs_ids"].append(int(obs.obs_id))
        for arc_id in connecting:
            if not current["arc_ids"] or current["arc_ids"][-1]!=arc_id:
                current["arc_ids"].append(arc_id)
        rows.append((obs.trip_id,obs.obs_id,int(row.candidate_arc_id),float(row.p_match),
                     "matched",reason is not None,reason))
        previous = (obs,row)
    result = pd.DataFrame(rows,columns=["trip_id","obs_id","predicted_arc_id","p_match","status","break_before","break_reason"])
    result = result.astype({"trip_id":"string","obs_id":"int64","predicted_arc_id":"Int64","p_match":"float64"})
    return result,{"trip_id":metadata["trip_id"],"runs":runs,
                   "breaks":result[result.break_before][["obs_id","break_reason"]].to_dict("records")}
