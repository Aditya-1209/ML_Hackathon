# PathSnap evaluator app

Laptop 1 runs FastAPI. Laptop 2 opens the evaluator UI, selects a trip, and submits
its ID. Laptop 1 loads the trip, creates candidate features, scores and decodes the
route, then compares predictions with server-side references. The response contains
the geometry and aggregate metrics required by the UI.

## Run the final app

On Laptop 1, from the repository root in PowerShell:

```powershell
python -m pip install -r requirements-app.txt
$env:PYTHONPATH = 'src'
python -m uvicorn pathsnap_ml.api:app_factory --factory --host 0.0.0.0 --port 8000
```

On Laptop 2, open `http://<laptop-1-ip>:8000` on the same LAN. Pick a trip and
select **Evaluate trip**.
Keep the Laptop 1 console visible. It logs each inference request and its outcome.

The default uses the checked-in GPS-context Random Forest and tuned Viterbi
decoder. It evaluates the ten student-test trips when no private evaluation
directory is configured. Use scikit-learn 1.9.1, the version used for training.

For a separately hosted UI on Laptop 2, run `python -m http.server 8080 --directory frontend`
on that laptop and open `http://localhost:8080/?api=http%3A%2F%2F<laptop-1-ip>%3A8000`.
Before starting the API, set `PATHSNAP_UI_ORIGINS` on Laptop 1 to
`http://localhost:8080`. Keep the UI and API URLs on the same network.

## Street map and offline display

The app uses a local Leaflet build with OpenStreetMap street tiles. For the
offline demo, add `?basemap=off` to the browser URL, or turn off **Street map**
from **Layers**. The trip road network still renders from API data and the
map controls still work. Only the street tiles require internet access.

## Connect evaluator data

On Laptop 1, set the private evaluation path, then restart the same API command:

```powershell
$env:PATHSNAP_EVALUATOR_ROOT = 'C:\PathSnap\private-evaluation'
```

To run an earlier model exported by the shared training pipeline, set
`PATHSNAP_MODEL_BUNDLE`. That bundle must contain `model.joblib` and `metadata.json` in the format exported
by the training pipeline. The FastAPI adapter uses the existing
`predict_candidates` function and validates the feature order and hash against
the configured feature generation. Use the dependency versions that produced
the selected model. The workspace decoder defaults apply until a compatible
`decoder_config.json` exported by `cli_decode tune` is placed in the model bundle.
That file wraps the active decoder settings in the top-level `config` field.
If the bundle includes an earlier decoder export with a different schema, set
`PATHSNAP_DECODER_CONFIG` to a compatible YAML or JSON file. To rehearse with
the checked-in decoder defaults, point that variable at `configs/decoder.yaml`.

The private evaluation directory must contain exactly ten trip folders in
`evaluator_test`, each with matching `.track`, `.nodes`, `.arcs`, and `.route` files.
For example, `evaluator_test/00000001/00000001.track` and its three companion
files. Set this directory outside the repository and leave it on Laptop 1.

Evaluator mode requires a configured model. It runs HMM/Viterbi and aligns the
ordered reference route only after prediction, using it exclusively for accuracy
and coverage. The UI displays reference-derived pointwise accuracy, route
continuity, and label coverage as distinct metrics. Accuracy includes all assigned
valid observations, including those whose reference arc was missed by candidate
generation.

## Verify

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q tests/test_evaluator_api.py
```

API documentation is available at `/docs`. The application endpoints are
`GET /api/health`, `GET /api/trips`, and `POST /api/evaluate`.
