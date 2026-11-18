import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix


def top_predictions(scores):
    return scores.sort_values(["p_match","candidate_arc_id"],ascending=[False,True]).drop_duplicates(["trip_id","obs_id"])


def evaluate(scores, labels, observations):
    eligible = observations[observations.valid][["trip_id","obs_id"]].merge(
        labels[labels.label_status.eq("assigned")],on=["trip_id","obs_id"],validate="one_to_one")
    predicted = top_predictions(scores)[["trip_id","obs_id","candidate_arc_id"]]
    points = eligible.merge(predicted,on=["trip_id","obs_id"],how="left",validate="one_to_one")
    points["correct"] = points.candidate_arc_id.eq(points.target_arc_id).fillna(False)
    candidates = scores.merge(eligible[["trip_id","obs_id","target_arc_id"]],on=["trip_id","obs_id"])
    result = {"eligible_observations":len(eligible),"valid_observations":int(observations.valid.sum()),
              "label_coverage":len(eligible)/max(1,int(observations.valid.sum())),
              "top1_accuracy":float(points.correct.mean()) if len(points) else None,
              "candidate_miss_rate":1-float(candidates.candidate_arc_id.eq(candidates.target_arc_id).sum())/max(1,len(eligible)),
              "no_candidate_rate":float((~observations[observations.valid].set_index(["trip_id","obs_id"]).index.isin(
                  scores.set_index(["trip_id","obs_id"]).index)).mean()) if observations.valid.any() else None}
    per_trip = points.groupby("trip_id").correct.agg(["mean","count"]).reset_index().rename(columns={"mean":"top1_accuracy","count":"labelled_observations"})
    result["per_trip"] = per_trip.to_dict("records")
    if len(candidates):
        y = candidates.candidate_arc_id.eq(candidates.target_arc_id).astype(int)
        pred = candidates.p_match.ge(.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y,pred,average="macro",zero_division=0)
        result.update(candidate_accuracy=float(accuracy_score(y,pred)),candidate_macro_precision=float(precision),
                      candidate_macro_recall=float(recall),candidate_macro_f1=float(f1),
                      log_loss=float(log_loss(y,np.column_stack([1-candidates.p_match,candidates.p_match]),labels=[0,1])),
                      confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
    return result
