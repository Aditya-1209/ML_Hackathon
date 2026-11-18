import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .decoder_evaluation import evaluate_decoder


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["evaluate","tune"])
    p.add_argument("--workspace",type=Path,default=Path("."))
    p.add_argument("--model",default="decision_tree")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    root=args.workspace.resolve()
    profile="smoke" if args.smoke else "baseline"
    config=yaml.safe_load((root/"configs/decoder.yaml").read_text())
    if args.command=="tune":
        records=[]
        for beta in [20,50,100]:
            setting={**config,"beta_m":beta}
            metric=evaluate_decoder(root,args.model,profile,"hmm",setting,"train")
            records.append({"beta_m":beta,"training_oof_proxy_accuracy":metric["pointwise_accuracy"]})
        chosen=max(records,key=lambda row:row["training_oof_proxy_accuracy"])
        config["beta_m"]=chosen["beta_m"]
        out=root/"artifacts"/("smoke" if args.smoke else "final")
        out.mkdir(parents=True,exist_ok=True)
        (out/"decoder_config.json").write_text(json.dumps({"config":config,"trials":records,
            "source":"training OOF candidate probabilities","model_id":args.model},indent=2))
    saved=root/"artifacts/final/decoder_config.json"
    if not args.smoke and saved.exists():
        config=json.loads(saved.read_text())["config"]
    summaries={}
    for mode in ["classifier_only","hmm"]:
        summaries[mode]=evaluate_decoder(root,args.model,profile,mode,config)
    path=root/"results/decoder"/profile/args.model/"comparison.json"
    path.write_text(json.dumps(summaries,indent=2))
    predictions=pd.read_parquet(path.parent/"hmm/student_test/predictions.parquet")
    from .route_maps import save_map
    save_map(root,str(predictions.trip_id.iloc[0]),predictions,path.parent/"sample_map.html")
    print(json.dumps({m:{"pointwise_accuracy":v["pointwise_accuracy"],"continuity":v["continuity"]}
                      for m,v in summaries.items()},indent=2))


if __name__=="__main__":
    main()
