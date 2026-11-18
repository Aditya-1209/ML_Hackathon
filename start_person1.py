"""Install dependencies locally and run Person 1's assigned models."""
from pathlib import Path
import argparse
import datetime
import os
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parent

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if sys.version_info < (3,10):
        raise SystemExit("Install Python 3.12, then run this launcher again.")
    env_dir=ROOT/".venv"
    python=env_dir/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env_dir)
    logs=ROOT/"results/logs"
    logs.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commands=[
        [str(python),"-m","pip","install","--disable-pip-version-check","-r",str(ROOT/"requirements.txt")],
        [str(python),str(ROOT/"run_person1.py")]+(["--smoke"] if args.smoke else [])
    ]
    with (logs/f"person1_{stamp}.log").open("w",encoding="utf-8") as log:
        for command in commands:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding="utf-8",errors="replace")
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line); log.flush()
            status=process.wait()
            if status:
                raise SystemExit(status)
    print("Done. Open results/baseline/models for metrics, plots and predictions.")
