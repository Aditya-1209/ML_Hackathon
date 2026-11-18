"""Copy and stage one owner's files into an existing local Git clone."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
BRANCHES={1:"Aditya-1209/data-lr-knn",2:"Aditya-TJ/features-random-forest",
          3:"Aditya-1209/experiments-histgb",4:"Akash-MP19/hmm-decision-tree"}
MODELS={1:["logistic_regression","knn"],2:["random_forest"],3:["hist_gradient_boosting"],4:["decision_tree"]}


def git(repo,*args,check=True):
    return subprocess.run(["git","-C",str(repo),*args],capture_output=True,text=True,check=check)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--person",type=int,choices=[1,2,3,4],required=True)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--commit",action="store_true")
    p.add_argument("--push",action="store_true")
    args=p.parse_args()
    repo=args.repo.resolve()
    if repo==ROOT or ROOT.is_relative_to(repo):
        p.error("Extract the runnable ZIP outside the clone, then pass the separate clone as --repo")
    git(repo,"rev-parse","--show-toplevel")
    status=git(repo,"status","--porcelain").stdout.strip()
    if status:
        p.error("The target clone has existing changes. Preserve/commit them before publishing this bundle.")
    branch=BRANCHES[args.person]
    if git(repo,"show-ref","--verify","--quiet","refs/heads/"+branch,check=False).returncode==0:
        git(repo,"switch",branch)
    elif git(repo,"show-ref","--verify","--quiet","refs/remotes/origin/"+branch,check=False).returncode==0:
        git(repo,"switch","--track","origin/"+branch)
    else:
        git(repo,"switch","-c",branch,"origin/main")
    paths=json.loads((ROOT/"OWNERSHIP.json").read_text())[str(args.person)]
    extra=[]
    for model in MODELS[args.person]:
        for pattern in [f"results/baseline/models/{model}",f"artifacts/standard/{model}"]:
            folder=ROOT/pattern
            if folder.exists():
                extra.extend(str(f.relative_to(ROOT)) for f in folder.rglob("*") if f.is_file())
    if args.person == 4:
        folder = ROOT/"results/decoder/baseline/decision_tree"
        if folder.exists():
            extra.extend(str(f.relative_to(ROOT)) for f in folder.rglob("*") if f.is_file())
    paths=sorted(set(paths+extra))
    for relative in paths:
        src=ROOT/relative
        if not src.is_file():
            raise FileNotFoundError(src)
        if src.stat().st_size>95_000_000:
            raise ValueError(f"{relative} exceeds the direct-Git upload budget. Keep the trained artifact local or use Git LFS.")
        dst=repo/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.read_bytes()==src.read_bytes():
            continue
        shutil.copy2(src,dst)
    for offset in range(0,len(paths),80):
        # Explicit paths only, including owned generated data ignored by older clones.
        git(repo,"add","-f","--",*paths[offset:offset+80])
    print(git(repo,"diff","--cached","--stat").stdout)
    if args.commit and git(repo,"diff","--cached","--quiet",check=False).returncode:
        print(git(repo,"-c","commit.gpgsign=false","commit","-m",f"Complete Person {args.person} PathSnap ML component").stdout)
    if args.push:
        if not args.commit:
            p.error("--push also requires --commit")
        print(git(repo,"push","-u","origin",branch).stderr)
    print("Prepared branch",branch)


if __name__=="__main__":
    main()
