# PathSnap development data and classifier baselines

The supplied development folders identify 80 training trips and 10 student-test trips. Trip IDs retain leading zeroes. The inventory counts repeated appearances of the same network separately when it is supplied for multiple trips.

| Measurement | Training | Student test |
|---|---:|---:|
| GPS observations | 208,417 | 19,160 |
| Invalid GPS rows | 0 | 0 |
| Repeated timestamps / backwards timestamp steps | 0 / 0 | 0 / 0 |
| One-second intervals / all intervals | 201,048 / 208,337 | 18,886 / 19,150 |
| Gaps longer than 10 seconds | 590 | 27 |
| Longest gap, seconds | 153,064 | 210 |
| Zero-length arc records | 100 | 6 |
| Reference route entries | 31,336 | 3,477 |

Although most observations are one second apart, training trip 00000090 contains a gap of more than 42 hours. Speed and transition features need an explicit gap policy. Some trip networks have hundreds of weakly connected components. An unreachable pair is a meaningful outcome for the decoder.

Each GPS row keeps its original observation ID. The loader preserves invalid observations and assigns temporal order only to valid rows. Arc and node IDs retain their original row positions. Longitude and latitude are projected to a WGS84 UTM zone selected from the median road-network coordinates for that trip. The resulting metre coordinates support endpoint distance and projection calculations. Each trip records the full CRS WKT, configuration and source hashes.

We model arcs as directed. For training trip 00000001, all 524 successive reference-arc pairs connect from the end of one arc to the start of the next. Across all training trips this holds for 31,243 of 31,256 pairs, and for 3,466 of 3,467 student-test pairs. This supports the directionality assumption but does not establish legal travel restrictions. Parallel and zero-length arcs retain their identities. Offline references are ordered sequences of arc IDs and can revisit a road. Their row positions do not supply observation-level targets.

## Reproduce and hand off

From the repository root, install requirements-data.txt in a Python 3.12 environment and use the official Dataset folder as RAW_ROOT. The full prepare command creates the manifest from the supplied development partition folders if the manifest is absent.

```sh
export PYTHONPATH=src
python -m pathsnap_ml.cli_data prepare --workspace . --raw-root "$RAW_ROOT" --include-reference --trip 00000001
python -m pathsnap_ml.cli_data prepare --workspace . --raw-root "$RAW_ROOT" --include-reference
python -m pytest -q tests/test_data_io.py tests/test_geometry.py
```

Person 2 can load data/processed/00000001 through data_io.load_processed_trip(Path("data/processed"), "00000001"). Its dictionary contains observations, nodes, arcs and metadata. The ordered development reference is in data/reference/00000001/route.parquet. Reference reading is an explicit offline step. Trip 00000001 has 1,315 observations, 88,809 nodes, 165,790 arcs and 525 reference entries. Person 3 can use data/manifest.csv for the trip groups.

The real-data CLI writes results/eda/development.csv, results/eda/development.png and results/eda/train-00000001.html. The interactive map shows the sample's valid GPS observations, their temporal order, and roads within its padded extent. The CSV includes per-trip extents, missingness, topology, gap counts and reference-transition checks.

## Logistic Regression and KNN

Both pipelines take the team's ordered candidate features. Median imputation and standardisation fit inside each training fold. The logistic baseline uses L2 regularisation with C=1 and up to 1,000 iterations to estimate class-1 probabilities. It supplies a linear baseline for the candidate features. KNN uses five neighbours, Euclidean distance in standardised feature space, and uniform votes. Its probabilities can tie and its inference time depends on the size of the training set. Model selection and tuning belong to the shared comparison.

Benchmark pending: Person 2's frozen features and reference-aligned labels, and Person 3's grouped folds and experiment runner. The synthetic pipeline tests verify probability output and preservation of all-missing feature columns. They provide no estimate of real-road accuracy. Once the handoffs arrive, run the following in the integrated development checkout and attach the actual timing, CV/student-test metrics and observed failures.

```sh
PYTHONPATH=src python -m pathsnap_ml.cli_models train --workspace . --model logistic_regression
PYTHONPATH=src python -m pathsnap_ml.cli_models train --workspace . --model knn
```
