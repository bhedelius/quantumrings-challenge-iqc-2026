# Features

Tensor Clock extracts **32 structural features** from each circuit, plus two
threshold terms added at dataset assembly (34 model inputs total). They are
grouped by the two factors of the cost model (`runtime ~ operations x chi^3`; see
`simulator_cost_model.md`). The selection rationale — which candidates were tried
and dropped — is in `feature_selection.md`.

All features are computed from circuit *structure* only (no filename or source
lookup) by `tensor_clock/qasm_text.py`, a dependency-light streaming parser for
OpenQASM 2.0/3.0 (no Qiskit); `tensor_clock/features.py` holds the shared helpers.

## Operations factor — how much work (roughly chi-independent)

| Feature | Meaning |
|---|---|
| `n_1q`, `n_2q`, `n_multi` | gate counts by arity (1-, 2-, 3+-qubit) |
| `n_t` | T / Tdg count — the explicit non-Clifford "magic" count (a simulation-hardness driver) |
| `n_cx` | CNOT count (the dominant entangling gate) |
| `n_meas` | measurement operations |
| `n_param_gates` | gates carrying at least one numeric rotation angle |
| `depth` | circuit depth (longest gate chain through any qubit) |
| `depth_2q` | depth counting only two-qubit gates (entanglement-generating depth) |
| `span_sum` | sum over 2q gates of `|q_i - q_j| + 1` — qubit-index distance (locality) |
| `span_max`, `span_mean` | largest / mean 2q qubit-index distance |
| `file_length_lines` | source length in lines — a parser-independent size proxy |

## Entanglement / chi proxies — the hard-to-measure factor

| Feature | Meaning |
|---|---|
| `entangling_sum` | summed entangling power of 2q gates, weighted by gate type and angle (swap 0, near-zero rotation ~0, CNOT 1) |
| `max_cut` | most 2q gates crossing any single cut of the 1-D ordering (peak bond pressure) |
| `bisection_cut` | 2q gates crossing the register midpoint (fixed-cut companion to `max_cut`) |
| `reordered_edge_span` | total interaction-graph edge span after a bandwidth-reducing (reverse Cuthill-McKee) reorder |
| `n_edges` | distinct qubit pairs joined by a 2q gate (interaction-graph edges) |
| `graph_density` | `n_edges / (n_qubits choose 2)` — interaction-graph connectivity |
| `max_degree`, `mean_degree` | most / mean 2q gates incident on a qubit |
| `degree_cv` | coefficient of variation of interaction-graph degrees (uniform vs hub-heavy) |
| `n_components` | connected components of the interaction graph (idle qubits count as singletons) |
| `frac_2q` | fraction of gates that are two-qubit (entangling density) |
| `frac_nonclifford` | fraction of gates outside the Clifford set (magic density) |
| `frac_near_identity` | fraction of rotation angles near zero (create ~no entanglement) |
| `angle_mean` | mean rotation magnitude over numeric parameters (wrapped to `[0, pi]`) |
| `frac_angle_clifford_grid` | fraction of angles on the `k*pi/4` (Clifford+T) grid |
| `angle_entropy` | Shannon entropy of the rotation-angle spectrum |
| `layer_periodicity` | peak autocorrelation of the per-layer gate-count sequence (ansatz/Trotter regularity) |

## Size / other

| Feature | Meaning |
|---|---|
| `n_qubits` | number of qubits |
| `is_dynamic` | 1 if mid-circuit measurement or classical control flow, else 0 |

## Threshold terms (added in `build_dataset`, not per-circuit)

| Input | Meaning |
|---|---|
| `threshold` | the simulator's bond-dimension (chi) cap |
| `log2_threshold` | its base-2 log (TabPFN benefits from both scales) |
