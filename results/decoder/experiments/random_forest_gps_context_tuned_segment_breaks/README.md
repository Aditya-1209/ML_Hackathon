# GPS-context Random Forest with tuned HMM/Viterbi reconstruction

**Decoded student-test Top-1 accuracy was 95.18%** (12,745 correct road assignments among 13,390 labelled observations on ten trips). Nine trips exceeded 90% individually. Trip 00000077 scored 89.89%. The labels were inferred from the reference routes and cover 69.89% of the 19,160 valid observations.

| Output using the same tuned forest | Correct assignments | Observation Top-1 | Feasible transitions | Continuity |
|---|---:|---:|---:|---:|
| Independent forest predictions | 12,718 | 94.98% | 17,264 / 19,149 | 90.16% |
| Previous Viterbi decoder | 11,630 | 86.86% | 18,850 / 19,149 | 98.44% |
| Viterbi with a tuned cost for starting a new segment | 12,745 | 95.18% | 17,543 / 19,149 | 91.61% |

## How the reconstructed route changed

Each candidate road is a hidden state. Its emission weight is the saved Bayesian-tuned forest probability. The Viterbi recurrence considers every candidate predecessor and traces back the best-scoring sequence. For a feasible transition the log weight is minus the absolute difference between network travel distance and observed GPS displacement, divided by the transition scale. A pair with no feasible directed path within the speed bound may now start a new segment, at a cost subtracted from the sequence score. The route export explicitly records these breaks. The original continuity metric counts every such pair as a failure.

The new decoder changed 435 assignments across all observations. Among the labelled observations, 94 previously incorrect assignments became correct and 67 previously correct assignments became incorrect. Relative to the independent forest, the decoded route gained 27 correct assignments and 279 feasible transitions. Relative to the previous decoder, the new decoded route gained 1,115 correct assignments and lost 1,307 feasible transitions. The ten route exports contain 1,606 unreachable-transition breaks and one large GPS gap.

## Selection and reproducibility

Using the tuned forest's out-of-fold probabilities, a seeded Bayesian TPE search tried ten decoder settings across five published folds covering 80 training trips. It maximised mean fold Top-1 **after traceback**. We searched the transition scale over 10–150 metres and the segment-break cost over 0.5–8 log-score units, both on log scales. The scale controls the penalty for distance mismatch. The cost controls how much stronger the emission and path evidence must be to justify breaking the route.

Trial 4 selected a 138.12-metre scale and a 0.526 break cost. Mean decoded validation accuracy was 93.98%, with fold scores of 94.87%, 97.16%, 91.45%, 90.97% and 95.45%. The search took 4.42 minutes. After freezing these settings, we evaluated the saved student-test probabilities using the same road candidates, labels, speed bound and continuity metric as the previous decoder. The student set had already been used for earlier development comparisons. The evaluator partition remained reserved.

Run `PYTHONPATH=src python -m pathsnap_ml.tune_viterbi` from the repository root. See the [trials](trials.csv), [frozen configuration](decoder_config.json), [per-trip student-test metrics](hmm/student_test/metrics.json), and [reconstructed routes](hmm/student_test/routes.json).
