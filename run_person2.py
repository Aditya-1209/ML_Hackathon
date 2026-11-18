from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
from pathsnap_ml.experiments import run_experiment

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    for model in ['random_forest']:
        run_experiment(ROOT,model,args.smoke)
