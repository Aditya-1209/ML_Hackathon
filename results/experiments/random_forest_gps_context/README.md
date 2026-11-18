# Random Forest with GPS approach and departure features

On the ten student-test trips, the forest selected the assigned road at **12,716 of 13,390 labelled observations (94.97% Top-1)**. The original forest selected 11,954 (89.28%). The new features corrected 1,028 previous errors and introduced 266 errors. Nine of ten trips improved. Trip 00000052 fell from 198/200 to 197/200.

| Measure | Original forest | Forest with GPS context |
|---|---:|---:|
| Mean Top-1 across five trip-grouped folds | 84.46% | 93.80% |
| Pooled student-test observation Top-1 | 89.28% | 94.97% |

The two forests use the same candidate sets, assigned-observation denominator, tie-break and forest settings (100 trees, maximum depth 16, minimum leaf size 3, seed 42). The experiment adds 19 features to the eight original inputs. It compares candidates’ distances at the current point, road alignment with incoming and outgoing GPS movement, and expected travel along each directed road and an adjacent connecting road. Three windows target 12, 40 and 90 metres travelled. Because the features use later observations, this model scores completed trips.

One configuration was specified in advance. The decision to refit required improving on the original forest’s mean Top-1 across the five published training folds. Each fold held out 16 entire trips. Fold scores were 94.64%, 96.92%, 91.42%, 90.69% and 95.33%. The selected forest was fitted on all 80 training trips and scored once on the ten student-test trips. These are development scores against reference-derived road labels, covering 13,390 of 19,160 valid observations. The evaluator partition was excluded. All results here use the highest-scoring candidate before HMM/Viterbi decoding.

To reproduce from the repository root, run `PYTHONPATH=src python -m pathsnap_ml.context_forest` with the project dependencies installed. The saved pipeline reproduced all 113,648 student-test probabilities on reload. The [validation record](validation.json), [student-test metrics](comparison.json) and [per-trip counts](paired_trips.csv) accompany the fitted pipeline and candidate scores.
