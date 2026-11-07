# Quantum Rings simulator: cost model

Notes on how the Quantum Rings simulator works and the theory of what actually
drives runtime. This is the basis for our feature engineering.

## 1. What the simulator is

The engine is an **approximate tensor-network simulator** (MPS-style), not a
state-vector, stabilizer, or density-matrix simulator. A paper trained on it
(Xing et al., arXiv:2606.11620) calls it the "Quantum Rings tensor-network
simulator." A third-party benchmark (arXiv:2607.09882) called it a "state-vector
interface" — that describes the public API surface (`QrStatevector`, etc.), a
view it reconstructs on demand, not the underlying engine, which is tensors.

The `threshold` column in our data is the **bond-dimension cap** (χ): the
simulator truncates the tensor-network representation to this many retained
Schmidt values, and that truncation is the source of the approximation. Runtime
rises steeply with it, consistent with the χ³ cost of MPS contraction.

## 2. What actually drives runtime

Three ways to simulate, with very different scaling:

- **State vector**: stores 2^n amplitudes; each gate costs ~2^n FLOPs. Exact but
  capped near ~40 qubits. (Not what Quantum Rings uses.)
- **Tensor network / MPS**: cost ~ `n · χ³`, where **χ (bond dimension) is where
  the exponential hides**. χ measures entanglement across cuts; it stays small
  for low-entanglement circuits and grows toward 2^(n/2) for highly entangled
  ones.
- **`threshold` caps χ** (truncation). Cost scales steeply with it, consistent
  with χ³: in our data going threshold 16→512 can blow up ~10^4×.

So, to first order:

```
runtime ≈ (number of operations) × (χ-dependent blow-up, capped by threshold)
```

This two-factor structure is visible in the data: at threshold 16 the median run
is 0.30 s, but the max is ~17,600 s — some circuits are slow purely from having
millions of cheap gates (the operations factor), independent of entanglement.

## 3. The upper bound, and why it is only an upper bound

The question we explored: can χ be estimated cheaply from graph structure
instead of doing the linear algebra?

Partly yes. The exact-contraction cost of a tensor network is `~2^(treewidth)`
of its interaction graph (the "contraction width"), computable from structure by
contraction-order optimizers (e.g. `cotengra`, `quimb`). This gives a principled
**upper bound** on difficulty.

But the structural bound is not the runtime, for reasons the graph cannot see:

1. **Entanglement depends on the amplitudes, not just the wiring.**
   - Clifford/stabilizer circuits (H, S, CNOT) can be densely connected — awful
     treewidth — yet are efficiently simulable (Gottesman–Knill). Structure says
     "impossible," physics says "trivial."
   - Gate angles: `CRZ(θ→0)` is ~identity and creates no entanglement; `CRZ(π/2)`
     is maximally entangling. Same graph edge, opposite cost. The graph has no
     angle information, and our circuits are full of parameterized rotations.
   - Interference and symmetry can keep real entanglement far below the
     worst case.

2. **Truncation makes the actual χ a numerical property.** The simulator discards
   the smallest Schmidt values up to the threshold. Whether that loses little or
   much depends on how fast the Schmidt spectrum **decays**, which is only known
   by (approximately) propagating the state. This is why Xing et al. train an ML
   model to predict the threshold needed for a target fidelity rather than
   computing it.

3. **The relevant graph is space–time** (qubits × layers), so depth is part of
   the difficulty; a static qubit-connectivity clique misses it.

4. **Exact treewidth is NP-hard**; heuristics are decent but can themselves
   exceed a tight per-circuit time budget on large, deep circuits.

**Conclusion:** structure gives a ceiling (the exact-simulation cost); the actual
runtime sits below it by an amount set by gate types, gate angles, and spectrum
decay — none of which are structural. That residual is irreducibly numerical,
which is why runtime prediction is a machine-learning problem, not a formula.

## 4. Implications for features

Model the two factors:

- **Operations (cheap, structural):** `span_sum` (summed qubit-index distance
  over 2-qubit gates, a proxy for nearest-neighbor swaps an MPS engine performs),
  2-qubit gate count, depth, total gate count, interaction-graph edges and
  degrees.
- **χ / entanglement (what the graph misses, approximate cheaply):** gate-type
  mix (Clifford vs non-Clifford), rotation-angle statistics (near-identity
  fraction), and optionally a treewidth/contraction-width estimate where it fits
  in budget.
- **`threshold`** is the χ cap and must be an input feature; runtime is roughly
  threshold-independent for easy circuits and scales ~threshold³ for hard ones
  until it saturates.

### Empirical baseline (log space)

Using `span_sum` alone against measured `duration_s` (514/532 circuits):

- `log10(span_sum)` vs `log10(duration)`, per threshold: r ≈ 0.42–0.61.
- OLS `log10(dur) = 0.56·log10(span_sum) + 0.83·log10(threshold) − 4.27`,
  R² = 0.475 (n = 2328). Threshold-only R² = 0.158; span_sum-only R² = 0.308.

So the structural span signal is genuine but partial; the remaining variance is
the entanglement/truncation residual that richer features and a nonlinear model
must capture.
