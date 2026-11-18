"""Run only after all four branch payloads and their completed results are present."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("OMP_NUM_THREADS","2")
from pathsnap_ml.comparison import compare

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--tune",action="store_true",help="Run Bayesian tuning and final decoder evaluation after comparison")
    p.add_argument("--trials",type=int,default=30)
    args=p.parse_args()
    selection=compare(ROOT,args.smoke)
    print("Selected classifier:",selection["model_id"],flush=True)
    if args.tune:
        if args.smoke:
            p.error("Run full experiments before final tuning")
        from pathsnap_ml.tuning import tune
        tune(ROOT,args.trials)
        subprocess.run([sys.executable,"-m","pathsnap_ml.cli_decode","tune","--workspace",str(ROOT),"--model","final"],
                       env={**os.environ,"PYTHONPATH":str(ROOT/"src")},check=True)
