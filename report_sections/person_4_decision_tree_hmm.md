# Person 4: Decision Tree, Road Routing, HMM Viterbi Decoder, and Route Evaluation

**Author:** Person 4 (Akash MP - `Akash-MP19`)  
**Component:** Decision Tree Classifier, Road Network Graph & Projected Routing, Hidden Markov Model (HMM), Viterbi Sequence Decoder, Route Continuity, and Comparative Evaluation  
**Branch:** `Akash-MP19/hmm-decision-tree`

---

## 1. Decision Tree Methodology

The candidate road-segment matching task requires classifying whether a generated candidate road segment $c$ corresponds to the true road traversed by a vehicle during a given GPS observation $t$. The PathSnap feature set comprises spatial and kinematic features (including candidate-to-observation perpendicular distance, bearing differences, trajectory headings, and segment lengths).

We implement `build_estimator(params=None, seed=42)` in `src/pathsnap_ml/models/decision_tree.py` conforming to the PathSnap shared model contract:
- An unfitted `sklearn.pipeline.Pipeline` with steps:
  1. `"impute"`: `SimpleImputer(strategy="median", keep_empty_features=True)` — guarantees robust handling of missing features and prevents dimensional collapse when an entire feature column is unobserved.
  2. `"model"`: `DecisionTreeClassifier(random_state=seed)` — an orthogonal, non-linear tree-based classifier.
- Step name is strictly `"model"`, enabling hyperparameter tuning (e.g. `model__max_depth`, `model__min_samples_split`, `model__criterion`) via standard scikit-learn grid/Bayesian search runners.
- The pipeline does not fit data inside the builder.
- Supports calibrated/uncalibrated probability prediction via `predict_proba`.
- Implements `get_positive_class_index` which inspects `estimator.classes_` to safely extract positive match probabilities $P(y=1|x)$, even when classes are sorted non-trivially or binary encodings vary.

Unlike distance-weighted methods (KNN) or linear estimators (Logistic Regression), the Decision Tree models sharp non-linear boundaries in spatial metric space without requiring feature standardisation.

---

## 2. HMM Architecture

Independent classifier predictions evaluate candidate road segments at each GPS observation point isolation. Because GPS points exhibit measurement noise, multipath distortion, and sampling jitter, classifier-only top-1 predictions often produce physically impossible trajectories (e.g. flickering between parallel alleys, sudden backwards jumps on one-way streets, or teleporting across disconnected highway dividers).

To enforce spatial and temporal physical plausibility, we model the sequence of road segments as a first-order **Hidden Markov Model (HMM)**:
- **Observations:** Sequence of GPS observations $O = (o_1, o_2, \dots, o_T)$, where each observation $o_t = (x_t, y_t, \tau_t)$ provides projected spatial coordinates and timestamp $\tau_t$.
- **Hidden States:** The discrete road segment candidate $c_{t, k}$ chosen from candidate set $C_t$ at step $t$.
- **Objective:** Identify the most likely sequence of road segments $S^* = (c_1^*, c_2^*, \dots, c_T^*)$ maximizing the joint posterior probability in log space.

```
GPS Observations:       o_1               o_2               o_3        ...        o_T
                         |                 |                 |                     |
Hidden States:         c_{1,k}  ------>  c_{2,j}  ------>  c_{3,m}     ...      c_{T,n}
                       (Arc A)           (Arc B)           (Arc C)
```

---

## 3. Hidden State Representation

For each observation $o_t$, the candidate generator generates a set of candidate states $C_t = \{c_{t, 1}, c_{t, 2}, \dots, c_{t, K_t}\}$. Each candidate state $c$ stores:
- `arc_id`: The unique 0-indexed identifier of the road segment arc.
- `fraction`: The scalar projection parameter $f \in [0.0, 1.0]$ locating the nearest point on the directed line segment between `from_node_id` and `to_node_id`.
- `projected_x`, `projected_y`: Metre coordinates of the orthogonal projection on the road segment.
- `distance_to_arc_m`: Perpendicular spatial offset between the GPS observation and the road segment.
- `p_match`: Machine learning classifier score or probability $P(\text{match} \mid \text{features})$.
- `obs_id`: The original un-mutated GPS observation row identifier.

---

## 4. Emission Score Formulation

The emission probability models the likelihood that candidate $c$ generated the observed GPS signal. We transform the classifier probability $p_{\text{match}}(c)$ into log-space with a configurable numerical safety floor $\epsilon$:

$$E_t(c) = \log\left(\max(p_{\text{match}}(c), \epsilon)\right)$$

Where:
- $\epsilon = 1.0 \times 10^{-6}$ (configured via `configs/decoder.yaml`).
- When `normalize_emissions: true`, the classifier probabilities for observation $o_t$ are normalized over the candidate set:
  $$\tilde{p}_{\text{match}}(c) = \frac{p_{\text{match}}(c)}{\sum_{c' \in C_t} p_{\text{match}}(c')}$$
  and $E_t(c) = \log(\max(\tilde{p}_{\text{match}}(c), \epsilon))$.

---

## 5. Transition Score Formulation

The transition score models the physical likelihood of moving from candidate $c_{t-1, i}$ at time $\tau_{t-1}$ to candidate $c_{t, j}$ at time $\tau_t$. Following the standard map-matching formulation, the probability density decreases exponentially with the discrepancy between the shortest road-network distance and the observed great-circle/Euclidean displacement:

$$T(c_{t-1, i}, c_{t, j}) = -\frac{|d_{\text{net}}(c_{t-1, i}, c_{t, j}) - d_{\text{obs}}(o_{t-1}, o_t)|}{\beta_m}$$

Where:
- $d_{\text{net}}(c_{t-1, i}, c_{t, j})$ is the **position-aware road-network shortest path distance** in metres between the projected positions on the respective arcs.
- $d_{\text{obs}}(o_{t-1}, o_t) = \|(x_t, y_t) - (x_{t-1}, y_{t-1})\|_2$ is the Euclidean distance in metres between consecutive GPS observations.
- $\beta_m$ is the configured transition scale parameter in metres ($\beta_m = 10.0$ default).

### Physical Constraints and Impossible Transitions
$T(c_{t-1, i}, c_{t, j}) = -\infty$ under any of the following physical conditions:
1. **Unreachable Roads:** No directed network path exists from the endpoint of arc $a_i$ to the start of arc $a_j$ ($d_{\text{net}} = \infty$).
2. **Speed Limit Violations:** Effective velocity exceeds physical bounds:
   $$\frac{d_{\text{net}}}{\tau_t - \tau_{t-1}} > v_{\text{max}} + v_{\text{slack}}$$
   where $v_{\text{max}} = 40.0 \text{ m/s}$ (144 km/h) and $v_{\text{slack}} = 5.0 \text{ m/s}$ to absorb measurement noise.
3. **Large GPS Gaps:** If $\tau_t - \tau_{t-1} > \Delta \tau_{\text{max}}$ (15.0 s), the transition is excluded and triggers a clean segment boundary.
4. **Non-positive Elapsed Time:** If $\Delta \tau \le 0$, transition is only permitted if the vehicle is stationary ($d_{\text{net}} \approx 0$).

---

## 6. Viterbi Algorithm & Dynamic Programming

We solve for the globally optimal sequence $S^*$ using log-space dynamic programming:

### Recurrence
For observation $t \in \{1, \dots, T\}$ and candidate $k \in \{1, \dots, |C_t|\}$:
$$V_t(k) = \max_{j \in \{1, \dots, |C_{t-1}|\}} \left[ V_{t-1}(j) + T(c_{t-1, j}, c_{t, k}) \right] + E_t(c_{t, k})$$
$$\text{Backpointer: } B_t(k) = \arg\max_{j \in \{1, \dots, |C_{t-1}|\}} \left[ V_{t-1}(j) + T(c_{t-1, j}, c_{t, k}) \right]$$

### Deterministic Tie-Breaking
When candidate total scores tie within numerical tolerance ($10^{-9}$), ties are resolved deterministically:
1. Higher candidate emission probability $p_{\text{match}}$.
2. Smaller numerical `arc_id`.
3. Deterministic index ordering.

### Transparent Reconnection and Backtracking
1. If an observation has no valid candidates, or is flagged invalid, it terminates the active chain. The preceding segment is backtracked and saved. The decoder resumes cleanly on the next valid observation with initial emission scores $V_{t'}(k) = E_{t'}(c_{t', k})$ and $B_{t'}(k) = \text{None}$.
2. If all incoming transitions into step $t$ are $-\infty$ (e.g. an abrupt vehicle teleportation or jump across an unmapped boundary), the decoder prevents collapse by transparently starting a new segment at $t$.
3. The decoder guarantees **exactly one prediction row per original observation**, preserving original row ordering and `obs_id`.

---

## 7. Road Routing & Position-Aware Network Distances

Accurate transition scoring requires calculating distance between **projected positions on road segments**, not merely endpoint nodes:

### Projection Math
For an arc $a$ with start node $u = (x_u, y_u)$ and end node $v = (x_v, y_v)$:
$$\vec{w} = (x_v - x_u, y_v - y_u), \quad L = \|\vec{w}\|$$
For GPS observation $P = (x_p, y_p)$:
$$t = \frac{(x_p - x_u) w_x + (y_p - y_u) w_y}{L^2}, \quad f = \min(\max(t, 0.0), 1.0)$$
Projected point: $P_{\text{proj}} = (x_u + f w_x, y_u + f w_y)$.

### Network Distance Calculation
Between $c_1 = (a_1, f_1)$ and $c_2 = (a_2, f_2)$:
1. **Same Arc ($a_1 == a_2$):**
   - In directed mode, if $f_2 \ge f_1$, distance is $(f_2 - f_1) L_{a_1}$.
   - If $f_2 < f_1$, the vehicle cannot reverse along a one-way arc. It must travel to the end of $a_1$, traverse network cycles back to the start of $a_1$, and reach $f_2$:
     $$d = (1 - f_1) L_{a_1} + d_{\text{cycle}}(v_1, u_1) + f_2 L_{a_1}$$
     If no cycle exists, $d = \infty$.
2. **Different Arcs ($a_1 \neq a_2$):**
   - Vehicle completes arc $a_1$: $d_{\text{tail}} = (1 - f_1) L_{a_1}$.
   - Traverses shortest network path from $v_1$ to $u_2$: $d_{\text{node}} = \text{Dijkstra}(v_1, u_2)$.
   - Enters arc $a_2$ up to projection: $d_{\text{head}} = f_2 L_{a_2}$.
   - Total distance: $d_{\text{net}} = d_{\text{tail}} + d_{\text{node}} + d_{\text{head}}$.

### Intermediate Arc Reconstruction
The router returns both distance and the ordered sequence of traversed arc IDs:
$$\text{Route} = [a_1] + \text{IntermediateArcs}(v_1, u_2) + [a_2]$$

### Routing Cache
All shortest paths are cached in memoized lookups keyed by:
- Node pair: `(from_node_id, to_node_id)`
- Position offset: `(from_arc_id, round(f1, 4), to_arc_id, round(f2, 4))`

---

## 8. Route Continuity Predicate

`compute_continuity(observations, nodes, arcs, candidates, predictions, metadata, config)` provides an identical, impartial evaluation metric across all classifiers:

### Definition
For every pair of consecutive valid observations $(o_t, o_{t+1})$ where $\tau_{t+1} - \tau_t \le \Delta \tau_{\text{max}}$:
- A transition is **continuous** if:
  1. Both predictions are non-null road segments ($a_t \ne \text{None}, a_{t+1} \ne \text{None}$).
  2. A valid network path exists from $c_t$ to $c_{t+1}$ ($d_{\text{net}} < \infty$).
  3. The transition obeys velocity limits: $\frac{d_{\text{net}}}{\tau_{t+1} - \tau_t} \le v_{\text{max}} + v_{\text{slack}}$.
- **Failures:** Missing predictions, unreachable arcs, or excessive velocity count as transition failures.
- **Reporting:** Returns numerator (`continuous_transitions`), denominator (`total_evaluated_transitions`), exclusions (`large_gap_exclusions`), and detailed failure breakdown.

---

## 9. Handling Missing & Anomalous GPS Observations

| Anomaly | Decoder Handling | Continuity Impact |
|---|---|---|
| **Zero Candidates** | Outputs `predicted_arc_id = None`, finalizes active Viterbi segment, resumes on next valid observation. | Counts as continuity failure for adjacent transitions. |
| **Invalid GPS Row** | Retains observation ID, sets prediction to `None`, resets segment boundary. | Run broken; excluded from transition evaluation. |
| **Large GPS Gap (> 15s)** | Splits Viterbi chain; no transition penalty across gap; restarts with prior emissions. | Explicitly excluded from continuity denominator and reported in `large_gap_exclusions`. |
| **Zero Delta Time ($\Delta\tau=0$)** | Checks for stationary position ($d_{\text{net}} \approx 0$). If moving, transition score is $-\infty$. | Fails if non-zero distance is traversed instantaneously. |
| **Disconnected Components** | If all transitions into observation are $-\infty$, starts new Viterbi segment transparently. | Counts as unreachable failure in continuity. |

---

## 10. Classifier-Only vs HMM/Viterbi Comparison

| Dimension | Classifier-Only (Greedy Top-1) | Classifier + HMM Viterbi |
|---|---|---|
| **Context** | Single observation in isolation | Full temporal & network trajectory |
| **Network Feasibility** | Unconstrained; frequently chooses disconnected segments | Strictly constrained to reachable road network |
| **Flicker & Jitter** | High; noisy GPS jumps across parallel streets | Low; transition penalties smooth trajectory |
| **Route Continuity** | Poor (breaks whenever top score is on parallel arc) | Near 100% on valid road network runs |
| **Computational Cost** | $O(T \cdot K)$ | $O(T \cdot K^2)$ + cached Dijkstra shortest paths |

---

## 11. Experimental Results

The test suite (`tests/test_routing_decoder.py` and `tests/test_route_metrics.py`) was executed with the project's Python 3.12 environment (`pytest-9.2.0`):

```
============================= test session starts =============================
platform win32 -- Python 3.12.14, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\GANESH AKASH\Downloads\ML_Hackathon
collected 22 items

tests/test_routing_decoder.py::test_tiny_hand_verifiable_and_connected_routing PASSED [  4%]
tests/test_routing_decoder.py::test_parallel_edges PASSED                [  9%]
tests/test_routing_decoder.py::test_disconnected_roads PASSED            [ 13%]
tests/test_routing_decoder.py::test_zero_length_edges PASSED             [ 18%]
tests/test_routing_decoder.py::test_same_arc_projected_positions_forward_and_backward PASSED [ 22%]
tests/test_routing_decoder.py::test_undirected_graph_support PASSED      [ 27%]
tests/test_routing_decoder.py::test_repeated_road_visits PASSED          [ 31%]
tests/test_routing_decoder.py::test_viterbi_one_output_per_original_observation PASSED [ 36%]
tests/test_routing_decoder.py::test_no_candidate_observations PASSED     [ 40%]
tests/test_routing_decoder.py::test_invalid_observations PASSED          [ 45%]
tests/test_routing_decoder.py::test_large_gps_gaps PASSED                [ 50%]
tests/test_routing_decoder.py::test_zero_delta_time PASSED               [ 54%]
tests/test_routing_decoder.py::test_all_invalid_transitions_transparent_restart PASSED [ 59%]
tests/test_routing_decoder.py::test_shuffled_score_rows_order_invariance PASSED [ 63%]
tests/test_routing_decoder.py::test_duplicate_score_keys_deterministic_tie_breaking PASSED [ 68%]
tests/test_routing_decoder.py::test_label_free_inference PASSED          [ 72%]
tests/test_route_metrics.py::test_continuity_perfect_run PASSED          [ 77%]
tests/test_route_metrics.py::test_continuity_missing_prediction_failure PASSED [ 81%]
tests/test_route_metrics.py::test_continuity_large_gap_exclusion PASSED  [ 86%]
tests/test_route_metrics.py::test_continuity_speed_violation_failure PASSED [ 90%]
tests/test_route_metrics.py::test_pointwise_accuracy_and_exact_match PASSED [ 95%]
tests/test_route_metrics.py::test_comparison_classifier_vs_hmm PASSED    [100%]

============================= 22 passed in 1.86s ==============================
```

### Measured Synthetic Comparative Benchmark
In the controlled comparative experiment (`test_comparison_classifier_vs_hmm`):
- **Classifier-Only:**
  - Continuity Rate: $0.0\%$ (failed due to isolated candidate choice on parallel arc).
  - Pointwise Accuracy: $66.7\%$.
- **HMM Viterbi Decoder:**
  - Continuity Rate: $100.0\%$ ($+100.0\%$ gain).
  - Pointwise Accuracy: $100.0\%$ ($+33.3\%$ gain).
  - Recovered the ground-truth sequence $[0, 0, 1]$ by overcoming local emission noise through transition path cost.

---

## 12. Folium Visualisations

`src/pathsnap_ml/route_maps.py` renders interactive HTML maps containing:
- **Blue Circle Markers:** Original GPS observations with popup observation IDs and timestamps.
- **Gray Polylines:** Surrounding road network arcs with directional node tooltips.
- **Orange Dashed Polyline:** Classifier-only greedy route.
- **Purple Polyline:** HMM Viterbi reconstructed route, split cleanly across large gaps and disconnections.
- **Green Polyline:** Ground-truth reference route (diagnostic inspection only).
- **Interactive Legend:** Floated UI control detailing active layers and color keys.

---

## 13. Limitations

1. **Planar Projection Assumption:** Metric distances rely on local UTM projections. For trajectories spanning UTM zone boundaries, coordinate projection distortion increases slightly.
2. **First-Order Markov Property:** State transitions depend solely on the immediate previous road segment, which can occasionally struggle in dense multi-level flyovers or complex grade-separated junctions without heading directionality features.
3. **Graph Directionality:** Arcs are modeled strictly as directed from `from_node_id` to `to_node_id` per the dataset convention. If a vehicle performs a two-way maneuver on an undocumented bidirectional street, the directed graph incurs cycle penalty.

---

## 14. Future Improvements

1. **Second-Order Transition Modelling:** Incorporating heading/turn-angle penalties across arc triplets $(a_{t-2}, a_{t-1}, a_t)$ to penalize unrealistic sharp U-turns.
2. **Spatial R-Tree Indexing:** Accelerating candidate projection and bounding box queries for city-wide road networks exceeding 500,000 arcs.
3. **End-to-End Decoder Tuning:** Tuning $\beta_m$ jointly with classifier probability calibration (temperature scaling / Platt scaling) using Bayesian optimization over training out-of-fold predictions.
