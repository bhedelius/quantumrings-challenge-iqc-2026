# Feature selection

We explored a wide candidate set — our own structural features plus the teammate
exploration in `tensor_clock/exploratory.py` (SupermarQ metrics, interaction-
distance and degree statistics, custom-gate and utilization counts) — and pruned
it to the **32 features** in `features.md`. This note walks through what was
dropped and why.

We judged candidates with grouped 10-fold cross-validation (by circuit) on the two
models that actually move, MLP and TabPFN; trees were a sanity check. Per-feature
effects below about +-0.003 are fold noise, so a feature had to clear that to earn
a place. A 15%-circuit hold-out and agreement between the two models kept us from
reading noise as signal.

## 1. Dropped: exact or linear duplicates of features we already had

Many exploratory features restate one of ours; keeping both adds only
multicollinearity. (Left: exploratory name. Right: the kept equivalent.)

| Exploratory feature | Equivalent kept feature |
|---|---|
| `interaction_distance_mean` | `span_mean` |
| `interaction_distance_max` | `span_max` |
| `interaction_distance_sum` | `span_sum` (= this + `n_2q`) |
| `supermarq_program_communication` | `graph_density` (identical formula) |
| `supermarq_entanglement_ratio` | `frac_2q` |
| `supermarq_critical_depth` | `depth_2q / depth` |
| `supermarq_measurement_ratio` | `n_meas / n_qubits` |
| `num_qubits`, `num_1q/2q/multi_q_gates`, `num_measurements`, `circuit_depth` | `n_qubits`, `n_1q/n_2q/n_multi`, `n_meas`, `depth` |
| `unique_interacting_pairs`, `qubit_degree_max/mean` | `n_edges`, `max_degree/mean_degree` |
| `total_ops`, `non_clifford_ratio`, `num_parameterized_gates` | `n_gates`-equivalent, `frac_nonclifford`, `n_param_gates` |
| `idle_qubits`, `file_size_bytes`, `qubit_degree_std` | linear combos of features already present |

## 2. Dropped: degenerate or near-constant

Features that carry almost no information on this dataset.

| Feature | Why dropped |
|---|---|
| `n_gates` | exactly `n_1q + n_2q + n_multi` — a pure linear combo; removing it even nudged TabPFN up |
| `active_qubits` | equals `n_qubits` for 526 of 532 circuits |
| `frac_active` | ~1.0 for 526 of 532 circuits (near-constant) |
| `n_reset` | non-zero in only 7 circuits |
| `n_cz` | non-zero in only 27 circuits; overlaps `n_2q` |

## 3. Dropped: leakage risk

| Feature | Why dropped |
|---|---|
| `n_barrier` | Barriers are simulation no-ops, so they cannot causally affect runtime. Its cross-validation signal (~0.006) was a **circuit-family proxy** — certain generators insert barriers — which would not generalize to the separate hold-out families the competition scores on. Dropped on principle; the lean set ties the full set without it. |

## 4. Tested additions that did not clear the noise floor

| Candidate | Result | Verdict |
|---|---|---|
| CAMPS GF(2) bond-dimension proxy (arXiv:2412.17209) | nullity rho~0.71 vs runtime, but equals T-count on rotation-heavy circuits (its rank saturates) | rejected (redundant) |
| Bravyi-Gosset T-count complexity proxy | hurt TabPFN (0.881 vs 0.898) | rejected |
| k>=3 distance-weighted entanglement (multi-span / multi-dist) | +-0.0001 | rejected (noise) |
| `supermarq_liveness`, `depth_1q` | slightly negative to neutral | rejected (noise) |
| Drop raw `threshold`, keep only `log2_threshold` | -0.0027 on TabPFN | rejected (TabPFN uses both scales) |
| treewidth estimate; small-chi quimb MPS probe | too slow for the 15 s budget (probe 10-90 s/circuit) | rejected (not cheaply extractable) |

## Result

That leaves the **32 features** in `features.md`. The broader finding
(`learning_curve.md`) is that accuracy is data-limited rather than feature-limited
— more structural features don't help — so we optimized for a small, clean set
rather than chasing feature count.
