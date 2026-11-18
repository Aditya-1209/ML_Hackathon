# Bayesian tuning and Viterbi reconstruction for the GPS-context Random Forest

On the ten student-test trips, Bayesian tuning increased classifier Top-1 by two correct road assignments. Viterbi then increased route continuity by 1,586 feasible transitions while reducing the number of correct road assignments by 1,088.

| Output | Student-test observation Top-1 | Route continuity |
|---|---:|---:|
| Previous GPS-context forest | 12,716 / 13,390 (94.97%) | — |
| Bayesian-tuned GPS-context forest | 12,718 / 13,390 (94.98%) | 17,264 / 19,149 (90.16%) |
| Same tuned probabilities with HMM/Viterbi | 11,630 / 13,390 (86.86%) | 18,850 / 19,149 (98.44%) |

Accuracy uses observations with a reference-derived road assignment. Continuity measures feasible directed transitions between eligible consecutive GPS observations. The raw classifier is the higher-accuracy output on these development trips. The decoded sequence is more connected. The student-test gain from tuning is small, and mean validation Top-1 changed from 93.80% to 93.78%.

## Training and selection

The seeded Optuna TPE search completed ten configurations, including three initial random configurations. Each trial used all five published trip-grouped folds of the 80 training trips. The objective was mean observation Top-1 across folds. The candidate set and 27 GPS and road features were fixed. The search varied tree count (32, 48 or 64) and bootstrap fraction (0.15–0.45) to test smaller forests. Tree depth (14–26) and minimum leaf size (1–12, log sampled) varied the amount of regularisation.

Trial 4 selected 64 trees, maximum depth 14, minimum leaf size 5 and a 45% bootstrap sample. The 50 search fits took 10.82 minutes including validation. Regenerating the winning out-of-fold scores, refitting on all 80 training trips, and scoring the student test brought total classifier experiment time to 13.23 minutes. All 113,648 student-test candidate probabilities were reproduced from the saved model.

The decoder used the tuned forest probabilities as emissions and the existing directed-road transition rules. We compared transition scales of 20, 50 and 100 metres using held-out training-trip probabilities. The 100-metre scale achieved the best pooled training decoded accuracy (77.54%) and was used for the student-test routes. No evaluator trips entered either stage.

Run `PYTHONPATH=src python -m pathsnap_ml.tune_context_routes` from the repository root. The [search trials](trials.csv), [winning configuration and fold scores](best.json), [classifier metrics](metrics.json), and [paired route results](../../../decoder/experiments/random_forest_gps_context_tuned/comparison.json) contain the measurements.
