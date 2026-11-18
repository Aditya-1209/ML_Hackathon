import argparse
from pathlib import Path

from .contracts import MODEL_IDS
from .experiments import make_folds,run_experiment
from .comparison import compare


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["make-folds","train","compare","tune"])
    p.add_argument("--workspace",type=Path,default=Path("."))
    p.add_argument("--model",choices=MODEL_IDS)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--skip-continuity",action="store_true")
    p.add_argument("--trials",type=int,default=30)
    args=p.parse_args()
    workspace=args.workspace.resolve()
    if args.command=="make-folds":
        make_folds(workspace)
    elif args.command=="train":
        if not args.model:
            p.error("--model is required for train")
        run_experiment(workspace,args.model,args.smoke)
    elif args.command=="compare":
        compare(workspace,args.smoke,not args.skip_continuity)
    else:
        if args.smoke:
            p.error("Final tuning requires full baseline runs")
        from .tuning import tune
        tune(workspace,args.trials)


if __name__=="__main__":
    main()
