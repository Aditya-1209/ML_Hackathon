# Person 2 — Candidates, features, offline labels and Random Forest

## Candidate generation and features

The candidate search indexes the projected road geometry with an STRtree, then measures exact point-to-segment distances for the intersecting bounding boxes. Baseline settings are predeclared: 50 m search radius, up to eight candidates, and a 200 m search only when the primary search finds none. Ties break on the lower arc ID. Parallel arcs retain their own IDs. A zero-length arc projects to its node with fraction zero. No-candidate observations retain their original observation IDs in the processed table.

The classifier inputs in order are distance, heading difference, arc length, projection fraction, elapsed seconds, GPS step distance, speed and heading availability. Motion uses the previous valid observation in saved chronological order. The first point lacks motion features. Zero elapsed time and gaps over 30 s retain the measured step and time but lack speed and heading. Steps under 3 m lack heading. Directed roads use the stored from-to bearing. Undirected roads use the closer of the two bearings. Arc heading is undefined for zero-length segments. Travel geometry comes from Person 1's projected coordinates and saved directionality. The prediction build does not open the reference store.

## Reference-derived labels

The supplied development archive contains one ordered sequence of route arcs per trip. Its route rows have no GPS observation IDs. No official pointwise label file or alignment procedure was found in the supplied materials. The offline alignment evaluates spatial distance to every occurrence in the full ordered route. A dynamic program minimizes total distance over nondecreasing route positions, with repeats and skipped route segments allowed. Adjacent timestamps separated by over 30 s start independent runs. Invalid observations and points with no reference geometry within 50 m stay unavailable. A point is ambiguous if several route positions remain possible within 5 m of the best total run cost, or if the eligible points do not admit any monotonic sequence within the distance limit. Repeated arc occurrences remain distinct even when they share an arc ID. These are proxy labels. Parallel roads, crossings, poor GPS, and positions on the same arc that move backwards can still cause mistakes. The method does not enforce within-arc progress, graph connectivity, or travel speed during alignment. It never substitutes a nearby candidate for a missing target.

The two-trip raw geometry probe used a local tangent-plane approximation solely for inspection. These counts will need to be regenerated on the shared projected data.

| Partition and trip | GPS observations | Reference route rows | Candidates | Assigned | Ambiguous | Candidate misses among assigned |
|---|---:|---:|---:|---:|---:|---:|
| Train 00000001 | 1,315 | 525 | 8,697 | 1,038 | 277 | 1 |
| Student test 00000000 | 2,503 | 179 | 9,911 | 2,287 | 216 | 2 |

Both inspected trips had a candidate for every valid observation. The observed proxy label coverage was 78.94% and 91.37%, respectively. The candidate miss rates use all assigned valid points as the denominator. Neither result measures classifier accuracy. No threshold was changed after the probe.

## Full development-archive check

After the two-trip probe, the same exploratory projection was used on all 90 development trips. An initial run identified five long trips above the original five-million-cell alignment limit. The backward pass now retains one row at a time, and the limit is 20 million cells to cover the largest observed run (17,883,272 cells). The rerun completed every trip and reproduced the previous 85 successful trip summaries exactly. The revised dynamic programming also matched exhaustive enumeration on 300 small cases.

| Partition | Trips | Valid GPS points | Candidate rows | Assigned | Ambiguous | Candidate misses among assigned | GPS points with no candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 80 | 208,417 | 1,228,578 | 134,828 | 51,034 | 437 | 108 |
| Student test | 10 | 19,160 | 113,710 | 14,118 | 5,015 | 24 | 0 |

These figures describe the exploratory raw-trip run. Regenerate them with Person 1's projected data and the canonical Parquet workflow before reporting shared baseline metrics.

## Random Forest

The unfitted pipeline uses median imputation fitted inside each training fold, with entirely missing feature columns retained, then a 150-tree Random Forest. The baseline uses seed 42, a minimum of three samples per leaf and balanced subsampling class weights. Trees can model nonlinear interactions such as heading being informative when movement is available. Important limitations include noisy proxy labels, correlated rows within trips, and potential calibration drift. Any impurity-based feature importance should be interpreted cautiously for correlated and continuous features. If selected for Bayesian tuning, useful search parameters include max depth, minimum leaf size, max features and tree count.

The seven component tests cover deterministic keys, parallel and zero-length arcs, fallback search, invalid fixes, repeated timestamps, directed heading, all-missing imputation, unequal route/observation counts, repeated route positions, ambiguous crossings and candidate misses. Random Forest fitting was checked on a synthetic fixture. No canonical training, cross-validation, overfitting analysis, trained artifact or student-test classifier metrics have been produced yet. The provided folder lacked Person 1's processed data/reader and Person 3's experiment runner/folds. Parquet integration also requires pyarrow.

## Team handoff

From the shared repo root after Person 1 prepares the development manifest, processed trips and reference store, and Person 3's runner is present:

```powershell
python -m pip install -r requirements-features.txt
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_candidates_features.py tests/test_reference_alignment.py
python -m pathsnap_ml.cli_features build --workspace .
python -m pathsnap_ml.cli_features labels --workspace .
python -m pathsnap_ml.cli_models make-folds --workspace .
python -m pathsnap_ml.cli_models train --workspace . --model random_forest
```

Share data/features/schema.json and its config hash, the per-trip Parquet files in data/candidates, data/features and offline data/labels, and coverage and diagnostic maps under results/features. Supply Person 4 the keyed model scores and candidate projections. Use assigned valid observations for pointwise metrics, including candidate misses. The final report should fill in real CV and student-test results from the canonical runner and disclose AI implementation assistance as required by the event.

