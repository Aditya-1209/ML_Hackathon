PERSON 2 | READY TO RUN
Branch: Aditya-TJ/features-random-forest
Assigned models: random_forest

1. Extract the whole ZIP into its own folder outside your Git clone.
2. Install Python 3.12 if needed. Internet access is needed once to install Python packages.
3. Windows: double-click RUN_PERSON_2.bat.
   Mac/Linux: open Terminal here and run sh RUN_PERSON_2.sh.
   Alternatively: python start_person2.py
4. This creates .venv, installs dependencies and runs your full five-fold experiments on the 80 training trips, then scores the 10 student-test trips.
5. Read results/baseline/models/<model_id>/summary.txt, metrics.json, diagnostics.png,
   cv_scores.parquet and student_test_scores.parquet.
   Trained pipelines/configuration are in artifacts/standard/<model_id>/.
   Console and installation logs are in results/logs/.
6. Person 4's command also evaluates classifier-only versus HMM and saves a map and route metrics under results/decoder/baseline/decision_tree/.
7. Everything needed to start is included. No peer handoff is needed to train.
   Data/labels are reference-derived offline proxy labels; see README.md and results/features/coverage.csv.
8. When ready, fetch your local team clone with git fetch origin.
   From THIS extracted folder, run:
   python publish_to_repo.py --person 2 --repo "ABSOLUTE_PATH_TO_YOUR_ML_HACKATHON_CLONE" --commit
   Add --push if your terminal has GitHub push access. The target clone must be clean.
   This copies only your assigned source/data files and, after training, your own results/artifacts.
9. After all four branches merge, the owner runs python finish_after_merge.py to compare the completed experiments.
   It calculates rankings using the common CV objective and writes comparison.csv, plots and selection.json.
   For final Bayesian optimisation and decoder tuning, run python finish_after_merge.py --tune --trials 30.

The archive contains prepared development data and code. Training must be invoked by your team.
There are no trained models, model scores or precomputed model rankings in this ZIP.
