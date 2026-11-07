# Models

Tensor Clock predicts simulation runtime for a `(circuit, threshold)` pair from
the circuit's QASM. The pipeline is one shared path; models differ only in the
estimator. The 32 features are in `features.md`, the selection rationale in
`feature_selection.md`, and the cost model behind the features in
`simulator_cost_model.md`.

**Model class is the main lever.** On identical features, 10-fold out-of-fold
metric: TabPFN 0.902, MLP 5-seed ensemble 0.879, gradient boosting 0.830, linear
regression 0.421 (see `figures/model_comparison`). Tree methods plateau because
axis-aligned splits cannot represent
the smooth, multiplicative `operations x threshold` surface; a neural net or a
pretrained transformer on log-scaled inputs can. We also found the problem is
**data-limited, not feature-limited** (see `learning_curve.md`), which is why a
small-data prior model like TabPFN wins.

## Pipeline (`tensor_clock/`)

1. **Parse / featurize** (`qasm_text.py`): `text_features` reads OpenQASM 2.0/3.0
   line by line (regex, no circuit object, no Qiskit dependency) and assembles the
   32-feature dict; `features.py` holds the shared helpers. The cache builder and
   the harness both call it the same way (`text_features(text.splitlines())`), so
   training and inference cannot drift. (Validated against a Qiskit parse: accuracy
   matched within cross-validation noise, so Qiskit was dropped entirely — see
   `feature_selection.md`.)
2. **Assemble** (`data.py`): `build_dataset` joins per-circuit features (cached to
   `feature_cache.csv`) with the per-`(circuit, threshold)` labels. Target is
   `log10(duration)`; timeouts use the 6-hour cap. Features plus `threshold` and
   `log2_threshold` are the model inputs.
3. **Evaluate** (`metrics.py`): `score` is the project metric
   `max(0, 1 - |log10(pred/actual)|)`; `grouped_cv` runs 10-fold `GroupKFold`
   grouped by circuit, so no circuit's rows straddle train and test.

## Models

- **TabPFN** (`tabpfn_model.py`) — the expected model. A pretrained tabular
  transformer, log1p-scaled features, `n_estimators=4`. Best accuracy with no
  tuning (~0.90). Needs the `tabpfn` package and a one-time TABPFN_TOKEN to
  download its weights, which we do not redistribute.
- **MLP** (`mlp.py`) — the self-contained fallback. A 5-seed `VotingRegressor` of
  MLPs on log1p-scaled, standardized features (~0.88). Pure scikit-learn, no token
  or weights, millisecond inference.

Both are fit at harness construction from the committed feature cache; the harness
(`submission-harness/model.py`) uses TabPFN and falls back to the MLP when TabPFN
is unavailable.

## What we tried and rejected

Grouped 10-fold CV, project metric. The recurring lesson: cheap structural
features plateau ~0.82 for trees and ~0.88-0.90 for TabPFN/MLP, and the remaining
signal (dynamical entanglement) cannot be extracted cheaply.

| Experiment | Result | Verdict |
|---|---|---|
| Model class: linear vs boosting vs MLP vs TabPFN | 10-fold CV 0.421 / 0.830 / 0.879 / 0.902 (figures/model_comparison) | the real lever |
| Algorithm-family fingerprints (angle-grid fraction, angle entropy, degree CV, layer periodicity) | helps both models, consistent sign | kept |
| Redundancy prune to 32 features | ties the full set, drops a family-proxy (`n_barrier`) | kept (see `feature_selection.md`) |
| CAMPS GF(2) bond-dimension proxy (arXiv:2412.17209) | nullity rho~0.71 vs runtime, but equals T-count on rotation-heavy circuits | rejected (redundant) |
| Small-chi (chi=16) MPS probe via quimb | max-bond correlates (r=0.82) but 10-90s/circuit | rejected (not cheaply extractable) |
| k>=3 distance-weighted entanglement; drop raw threshold; liveness/depth_1q | all at the +-0.003 noise floor | rejected |
| Dependency-light text parser vs a Qiskit parse | text-only TabPFN 0.9016 vs 0.9025 (within noise), despite differing on ~5% of circuits | adopted text-only; dropped the Qiskit dependency |

Takeaway: model class and data volume dominate; more structural features give only
noise-level nudges. Measuring entanglement cheaply is about as hard as running the
simulation.
