# PathSnap runnable ML project

Use Python 3.12. Install with `python -m pip install -r requirements.txt`. Every standalone ZIP includes the same prepared development data and shared runtime, so each owner can train immediately. Keep each ZIP in a separate extracted folder outside the Git clone.

## Selected model for completed trips

The Bayesian-tuned GPS-context Random Forest achieved **94.98% student-test observation Top-1**. With the selected HMM/Viterbi decoder, the result was **95.18% Top-1 and 91.61% route continuity** on ten student-test trips. The fitted classifier is stored with its 27-feature configuration, and the decoder configuration records the tuned cost of starting a new route segment. [Fitted classifier and feature columns](results/experiments/models/random_forest_gps_context_tuned/) · [Selected decoder configuration](results/decoder/experiments/random_forest_gps_context_tuned_segment_breaks/decoder_config.json)

The reports cover [model comparison and selection](report_sections/06_model_comparison_and_selection.md), [Bayesian classifier tuning](report_sections/07_random_forest_bayesian_optimisation.md) and [Viterbi route reconstruction](report_sections/08_hmm_viterbi_route_reconstruction.md).

## Baseline team runners

Run the script matching your assignment.

```sh
python run_person1.py
python run_person2.py
python run_person3.py
python run_person4.py
```

Person 1 trains Logistic Regression and KNN. Person 2 trains Random Forest. Person 3 trains HistGradientBoosting. Person 4 trains Decision Tree. Each command runs the fixed five trip-grouped folds on all 80 training trips, refits on those trips and measures the 10 student-test trips. The complete results go under results/baseline/models/<model_id> and artifacts/standard/<model_id>. Pass --smoke for a smaller 10-training-trip, 2-student-trip software check using the same five-fold assignments. Smoke outputs are stored separately and cannot select the final model.

## LightGBM experiment

From the repository root, install the dependencies and run the sixth baseline with the shared experiment runner:

```sh
PYTHONPATH=src python -m pathsnap_ml.cli_models train --workspace . --model lightgbm
```

In Windows Command Prompt, enter `set PYTHONPATH=src` first, then run the command beginning with `python`. The model uses the same candidate features, folds and scoring rules as the original five. The [comparison report](report_sections/06_model_comparison_and_selection.md) includes all six measured runs.

## Put your completed part on your branch

First fetch your team clone with git fetch origin. From the extracted standalone folder, run the publishing helper with your person number and the absolute path to that clone.

```sh
python publish_to_repo.py --person 1 --repo "/path/to/ML_Hackathon" --commit
```

On Windows, quote a path such as "C:\Users\HP\ML_Hackathon". Substitute your own person number. Add --push to publish using YOUR local Git credentials, or push from your terminal afterward. The target clone must have a clean working tree. The helper switches/creates your named branch, copies and stages only your owned paths, and includes your completed full baseline model results and artifacts. It refuses files over 95 MB and leaves the work local if GitHub authentication fails. It does not merge any branches. OWNERSHIP.json lists every assigned path. Source/data paths have one owner each, so shared runtime files are published once by their owner.

The owner can merge all four branches at the end. After all owned code, prepared tables and completed model results are present, run:

```sh
python -m pip install -r requirements.txt
python finish_after_merge.py
```

This refuses missing/mismatched model runs, compares CV/student-test results, computes classifier route continuity and writes results/baseline/comparison.csv, plots and selection.json. The predeclared primary criterion is mean-fold observation Top-1 accuracy, with pooled candidate Macro F1 as the first tie-break. The student-test partition supplies development comparisons. Model selection from these development experiments does not replace the final reserved evaluator assessment.

To reproduce the earlier baseline runner’s tuning and route evaluation, run this command.

```sh
python finish_after_merge.py --tune --trials 30
```

Tuning uses fixed training CV and saves the trials. It refits the chosen classifier and generates selected-configuration OOF scores, then tunes the decoder on those training OOF scores and compares classifier-only/HMM on student-test trips. This baseline runner writes its classifier artifacts to artifacts/final. Decoder comparisons and maps are under results/decoder/baseline/final. Thirty trials are a proposed budget.

## Prepared data and label interpretation

The supplied package includes only the 80 training and 10 student-test trips. Original observation/node/arc IDs are preserved. Coordinate metadata records the per-trip projected CRS and directed-arc assumption. The road networks include disconnected components and zero-length arcs.

Because the supplied .route files give ordered traversed segments, observation targets are inferred offline from a monotonic alignment to the full reference-route sequence. Ambiguous or distant assignments remain unlabelled. This makes reported accuracy a reference-derived proxy. Coverage, candidate misses and no-candidate observations must accompany every score. Inspect results/features/coverage.csv before interpreting performance. Candidate generation itself never reads the references, and inference/decoding receive no labels.

The alignment is a reproducible baseline that the team must review on maps and defend. It is based partly on proximity, which can favour distance-based models. It does not establish independently verified ground truth. Do not describe the proxy figures as official evaluator accuracy. The scoring denominator includes every valid assigned observation, including missing true candidates and no-candidate cases.

## Reproduce the preparation

Set PYTHONPATH=src (Windows CMD: set PYTHONPATH=src) before module commands. To recreate cleaned data from the original Dataset folder:

```sh
python -m pathsnap_ml.cli_data prepare --workspace . --raw-root "/path/to/Dataset" --include-reference
python -m pathsnap_ml.cli_features --workspace .
```

Explicit --include-reference loads development reference files into data/reference for offline alignment. Prediction loaders do not open that directory. Rebuilding features invalidates previous result comparisons unless the full data hash matches. Keep the supplied baseline configuration fixed for the initial five-model comparison. Learned imputation and scaling fit inside each fold. Full model-training times depend on the computer, particularly KNN scoring and Random Forest fitting.

## Verification

Run `python -m pytest -q` with PYTHONPATH=src. The integration checks cover schema/ID matching, shuffled score keys, reference-independent features/decoding, target-miss denominators, directionality/projected offsets and model probability columns. verification/ contains the actual delivered-bundle check record. Training results in results/smoke are software checks only.
