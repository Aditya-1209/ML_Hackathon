import json
from pathlib import Path

import pandas as pd
from .contracts import MODEL_IDS


def compare(workspace,smoke=False,continuity=True):
    profile="smoke" if smoke else "baseline"
    root=workspace/"results"/profile
    records,hashes=[],set()
    versions=set()
    for model_id in MODEL_IDS:
        path=root/"models"/model_id/"metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Run {model_id} first: {path}")
        metric=json.loads(path.read_text())
        if metric["profile"] != ("smoke" if smoke else "standard"):
            raise ValueError("Cannot mix smoke checks with complete experiments")
        hashes.add(metric["data_hash"])
        versions.add(metric["sklearn"])
        row=dict(model_id=model_id,cv_mean_top1=metric["cv_mean_top1"],
                 cv_candidate_macro_f1=metric["cv_pooled"]["candidate_macro_f1"],
                 student_top1=metric["student_test"]["top1_accuracy"],
                 student_candidate_macro_f1=metric["student_test"]["candidate_macro_f1"],
                 student_log_loss=metric["student_test"]["log_loss"],inference_seconds=metric["inference_seconds"])
        if continuity:
            from .decoder_evaluation import evaluate_decoder
            row["student_route_continuity"]=evaluate_decoder(workspace,model_id,profile,"classifier_only")["continuity"]
        records.append(row)
    if len(versions)!=1:
        raise ValueError("Use the same scikit-learn version for every model run")
    if len(hashes)!=1:
        raise ValueError("Models used different datasets/configurations/folds; rerun on the same snapshot")
    table=pd.DataFrame(records).sort_values(["cv_mean_top1","cv_candidate_macro_f1","model_id"],ascending=[False,False,True])
    root.mkdir(parents=True,exist_ok=True)
    table.to_csv(root/"comparison.csv",index=False)
    selection=dict(model_id=table.iloc[0].model_id,profile=profile,
                   objective="Mean-fold observation Top-1; ties use pooled OOF candidate Macro F1, then model ID",
                   data_hash=next(iter(hashes)),accuracy_basis="reference-derived proxy",
                   provisional=smoke)
    (root/"selection.json").write_text(json.dumps(selection,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),layout="constrained")
    table.set_index("model_id")[["cv_candidate_macro_f1","student_candidate_macro_f1","student_top1"]].plot.bar(ax=ax)
    ax.set(ylabel="Score",ylim=(0,1),title="Reference-derived development metrics"+(" — SMOKE CHECK" if smoke else ""))
    ax.tick_params(axis="x",labelrotation=20)
    fig.savefig(root/"comparison.png",dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    table.set_index("model_id").student_log_loss.plot.bar(ax=ax)
    ax.set(ylabel="Binary candidate log loss",title="Student-test probability quality")
    ax.tick_params(axis="x",labelrotation=20)
    fig.savefig(root/"log_loss.png",dpi=150)
    plt.close(fig)
    print(table.to_string(index=False),flush=True)
    return selection
