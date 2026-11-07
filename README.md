# Quantum Rings Circuit Runtime Prediction Challenge

Predict how long a quantum-circuit simulation will take, straight from the circuit file, without running it.

---

## Tensor Clock — our solution

Predicting tensor-network simulation runtime from circuit structure. The simulator
is an approximate matrix-product-state (MPS) engine, so `runtime ~ operations x chi^3`
where `chi` (bond dimension) is set by the circuit's entanglement and capped by the
`threshold`. We extract **32 structural features** split across those two factors
and predict `log10(duration)`.

- **Model:** TabPFN (a pretrained tabular transformer) is the expected model
  (~0.90 cross-validated); a self-contained scikit-learn MLP (~0.88) is the
  fallback the harness uses when TabPFN's weights, which we do not redistribute,
  are unavailable. Model class is the main lever (trees plateau ~0.82).
- **Key finding:** the problem is data-limited, not feature-limited — more circuits
  would help, more structural features do not (`docs/learning_curve.md`).

### Repository layout

`tensor_clock/` — the package (dependency-light: numpy, scipy, scikit-learn; no Qiskit):

| File | Purpose |
|---|---|
| `qasm_text.py` | streaming OpenQASM 2.0/3.0 parser → the 32-feature dict; defines `FEATURE_NAMES` |
| `features.py` | shared constants and helpers (Clifford set, entangling power, graph metrics) |
| `data.py` | feature cache (`build_feature_cache` / `load_or_build_features`) and `build_dataset` |
| `metrics.py` | the project `score` and `grouped_cv` (circuit-grouped cross-validation) |
| `tabpfn_model.py` | the TabPFN model (`make_model`) — the expected submission model |
| `mlp.py` | the scikit-learn MLP ensemble — the self-contained fallback |
| `exploratory.py` | the wider candidate feature set we pruned from (not used in production) |

`submission-harness/` — the competition harness:

| File | Purpose |
|---|---|
| `model.py` | `RuntimeModel`: `featurize` + `predict` (the only file the contest lets you edit) |
| `run.py`, `score.py` | organizer-provided runner (→ `submission.csv`) and scorer |
| `requirements.txt` | model dependencies |

`docs/` — write-up and analysis:

| File | Purpose |
|---|---|
| `features.md` | the 32 features, grouped and defined |
| `models.md` | pipeline, models, and the tried/rejected table |
| `feature_selection.md` | how the candidate set was pruned to 32, and why |
| `simulator_cost_model.md` | the MPS cost model behind the features |
| `learning_curve.md` | the data-limited finding |
| `figures/` | figures (`.pdf`/`.png`) plus the scripts that build them: `model_sweep.py` (CV + learning curves → `sweep_results.json`), `learning_curve.py`, `model_comparison.py` |

**Data** (repo root): `train-circuits/` (compressed QASM), `train-runtime-data.csv` (labels), `feature_cache.csv` (derived features, used to fit the model at startup).

### Run

```bash
pip install -r submission-harness/requirements.txt   # includes tabpfn (needs a one-time TABPFN_TOKEN)
cd submission-harness && python run.py --team "Tensor Clock" --circuits ../train-circuits
```
The model fits at startup from `feature_cache.csv`; rebuild that cache from the raw
circuits with `python -m tensor_clock.data` (or delete it and it is rebuilt). To
reproduce the analysis figures: `python docs/figures/model_sweep.py` then
`python docs/figures/learning_curve.py` and `python docs/figures/model_comparison.py`.

---

## Contents

- [Overview](#overview)
- [Timeline](#timeline)
- [The task](#the-task)
- [Runtime & timeouts](#runtime--timeouts)
- [Dataset](#dataset)
- [What you build](#what-you-build)
- [Judging & scoring](#judging--scoring)
- [Submission](#submission)
- [Rules](#rules)
- [Simulator environment](#simulator-environment)
- [Resources](#resources)

## Overview

You get a library of OpenQASM circuits and the measured wall-clock time to simulate
each one on a fixed CPU simulator, at several settings. Build a program that reads a
raw circuit file and predicts how long the simulation will take, in seconds.

Some circuits are too complex to finish in the 6-hour cap and time out. Predicting
those matters too, since a timeout is itself a signal that a circuit is overly
complex for that setting. Your output is always a number of seconds: predict
≥ 21,600 s (the cap) for a run you expect to time out. How your model gets there,
e.g. a timeout classifier first or one continuous runtime target, is up to you.

You will build two things:

1. A **feature parser** — turns a `.qasm` file into a feature vector.
2. A **model** — maps `(features, simulator setting)` → predicted runtime.

Designing the feature set is the core of the challenge. No parser or feature list is provided.

## Timeline

| Milestone | When |
|---|---|
| Kickoff — training dataset released | 3:15PM 11/6 |
| Hold-out circuits released | final hours of the event (exact time TBA) |
| Submissions close | 3:00PM 11/7 |
| Results | 5:00PM 11/7 |

Competition is **team-based**. Team size and division of labor are up to you.

## The task

- **Input:** one OpenQASM circuit, plus the simulator `threshold` it will run at.
- **Output:** the predicted wall-clock runtime, in seconds, for that
  `(circuit, threshold)` pair.
- Threshold controls an internal simulator tradeoff between efficiency and accuracy. The same circuit at a higher
  threshold generally runs longer, so `threshold` is a model input, not a constant.
- Runs are capped at **6 hours**. Some circuits time out, predicting that is part
  of the task, since it signals a circuit is too complex to simulate at that
  setting. There is no separate timeout flag: predict ≥ 21,600 s for a run you
  expect to time out.

Filenames are **scrambled IDs**. The source algorithm is not a usable feature.
Circuit *structure* that happens to reveal the algorithm (e.g. a recognizable QFT block) is fair game.

## Runtime & timeouts

Each `(circuit, threshold)` pair in the training data has:

| Field | Description |
|---|---|
| `status` | `success` or `timeout` |
| `duration_s` | wall-clock seconds; blank when `status = timeout` |

A `timeout` means the run was killed at the 6-hour cap so the true duration is
unknown. These circuits are included deliberately:
recognizing that a circuit is heading for a timeout (i.e. is overly complex for the
given setting) is a meaningful part of the prediction problem. For scoring, a
timeout counts as exactly the cap (21,600 s), so any prediction ≥ 21,600 s gets
full credit on it.

## Dataset

**~532 circuits** from four sources, each run at up to **5 simulator thresholds**
(`16, 64, 256, 512, 1024`) → **~2,600 labeled runs**.

| Source | Circuits | Focus |
|---|---|---|
| MQT Bench | 180 | Algorithmic benchmarks, each algorithm swept across sizes (2–260 qubits) |
| QASMBench | 80 | Application benchmarks (small / medium / large tiers) |
| Quantinuum | 82 | Challenge circuits across 5 families |
| Google Quantum Supremacy 2019 | 190 | Sycamore random circuits, full + elided variants, 12–53 qubits |

- **Circuits:** [`train-circuits/`](train-circuits/), one
  `<id>.qasm.zst` per circuit.
- **Format:** OpenQASM, one circuit per file, `zstd`-compressed as `.qasm.zst`
  (see [below](#getting-the-raw-qasm-files) to decompress). Some circuits are
  OpenQASM 2.0 while others are 3.0. Make sure your parser can handle both.
- **Labels:** [`train-runtime-data.csv`](train-runtime-data.csv), one row per run: `filename`, `threshold`, `duration_s` (blank on timeout),
  `status` (`success` / `timeout`), plus the fixed run settings (`shots`,
  `backend`, `precision`). There is one prediction target, the runtime in
  seconds; a `timeout` row counts as ≥21,600 s.
- **Missing runs:** 12 of the 2,660 `(circuit, threshold)` runs have no row,
  because the simulator crashes when trying to run them. As a result, a few circuits have
  labels for fewer than five thresholds.
- **Qubit cap:** 260

### Getting the raw `.qasm` files

The circuits are compressed to keep the repo small: ~27 MB compressed vs. ~950 MB
of raw QASM (the largest single circuit is ~200 MB). The harness reads `.qasm.zst`
directly, so you only need raw files for your own exploration and training.

Decompress into a folder **outside** `train-circuits/`. The harness reads every
`.qasm` and `.qasm.zst` in the folder it is pointed at, so raw copies there would
be predicted twice.

Install `zstd` (`brew install zstd`, `apt install zstd`, or
[releases](https://github.com/facebook/zstd/releases) on Windows), then:

```bash
mkdir -p raw-circuits
zstd -d --output-dir-flat raw-circuits train-circuits/*.qasm.zst
```

To read them from Python instead, reuse `read_qasm()` in
[`submission-harness/run.py`](submission-harness/run.py). Plain
`zstandard.decompress()` fails on these files.

## What you build

| Component | Requirement |
|---|---|
| **Feature parser** | `.qasm` → features. You design the features. Must run in **≤ 15 s per circuit**, large circuits included. |
| **Model** | Predicts runtime, in seconds, from `(features, threshold)`; predict ≥ 21,600 s for a run you expect to time out. Final inference should take **≤ 15 s per circuit**. |

Both plug into the submission harness: implement `featurize` and `predict` in
[`submission-harness/model.py`](submission-harness/model.py). See the
[harness README](submission-harness/README.md).

You must be able to **explain every part of your submission**.

## Judging & scoring

Your final score is **1/3 automated + 2/3 subjective**.

### Automated — did you get the right results? (1/3)

**Duration accuracy** on a **hidden hold-out set** of circuits, separate from the
training library. For each `(circuit, threshold)`, predicted vs. measured runtime is
scored on a **log scale**:

```
score = max(0, 1 − |log10(pred / actual)|)
```

An exact prediction scores 100%; off by 10× scores 0%. Log scale because runtimes
span seconds to hours — being 2× off costs the same whether the run is 1 second or
1 hour. Your automated score is the average over **every** hold-out run; a run
missing from your submission scores 0.

**Timeouts** are folded into the same metric: a timed-out circuit is scored against
the 6-hour cap, so you "predict a timeout" simply by predicting a duration ≥ the cap
(that earns full credit), and predicting seconds under the cap for timed out circuits is penalized like any other miss. **There is no separate timeout flag**.

**Speed** is measured too: your parser and inference must stay under the per-circuit
cutoff (see *What you build*). Runs over the cutoff may be penalized.

### Reference scores

Automated scores for two reference models, to give you a sense of scale:

| Model | Train (cross-validated) | Hold-out |
|---|---|---|
| Trivial baseline: predict median train runtime for each threshold | 28.4% | 27.4% |
| Our internal model | 82.0% | 77.9% |

A model scoring below the trivial baseline has learned less than ignoring the
circuit entirely.

For reference, our internal model extracts 131 total features from each circuit. Not
all of them help, so feature pruning narrows these down to 25 used in the
final model.

### Subjective — how you got there (2/3)

Judged live from your presentation and write-up, each category scored 1–5. The five
are **weighted equally** — each is worth ≈ 13.3% of your final score (together the
2/3 subjective portion).

| Category | What we look for |
|---|---|
| **Novelty of approach** | Original, well-justified feature engineering / modeling — not blind tool use. |
| **Above & beyond** | Meaningful work past the core: extra analysis, validation, depth, polish. |
| **Thought process & learning** | Clear problem understanding, hypotheses, iteration, and learning over the project. |
| **Quality of presentation** | Clear explanation of problem/approach/results; honest error analysis; direct answers. |
| **Handling challenges** | Smart tradeoffs and resilience against messy data, time, and resource limits. |

You must be able to **explain every part of your submission** — this is scored.

Hold-out circuits are pushed to the repo in the **final hours** so you run your model on them
and submit your predictions before final judging.

## Submission

In the final hours we push the hold-out circuits to this repo, in a
`holdout-circuits/` folder at the top level next to `train-circuits/`. Run
`git pull` to get them, then run the harness on that folder:

```bash
cd submission-harness
python run.py --team "Your Team Name" --circuits ../holdout-circuits
```

It writes **`submission.csv`**. **DM `submission.csv` back to us** — that's your
entry for the automated score. Your presentation and write-up are judged live for
the subjective score.

## Rules

- **Teams.** Team-based; size and work split up to you.
- **Build your own parser.** No starter parser or feature list is provided.
- **No filename shortcuts.** IDs are scrambled — do not try to recover the source
  or algorithm from file metadata.
- **Own your solution.** You must understand and be able to explain everything you
  submit.

## Simulator environment

The runtimes you are given come from:

- A fixed **single-node CPU backend**.
- **Single precision.**
- `threshold` ∈ `{16, 64, 256, 512, 1024}`.
- **6-hour** timeout per run.

Fidelity is not modeled, this is purely runtime prediction for fixed simulator settings.

### Reference hardware

All runtimes were measured on one machine:

| | |
|---|---|
| CPU | Intel Core Ultra 9 285K |
| Cores | 24 physical / 24 logical |
| RAM | 128 GB @ 6400 MHz |

## Resources

Potentially useful starting points for thinking about circuit features:

- **OpenQASM spec** — <https://openqasm.com>
- **MQT Bench** — benchmark suite and circuit metrics — <https://www.cda.cit.tum.de/mqtbench/>
- **MQT Predictor** — ML on circuit features to choose the best device/compiler; a
  closely related feature-engineering problem — <https://github.com/cda-tum/mqt-predictor>
- **SupermarQ** — feature-based benchmarking (program communication, critical depth,
  entanglement ratio, liveness, measurement) — <https://arxiv.org/abs/2202.11045>
- **QASMBench** — application benchmark suite with circuit characterization metrics —
  <https://arxiv.org/abs/2005.13018> · <https://github.com/pnnl/QASMBench>
- **Quantum supremacy using a programmable superconducting processor** — Google, 2019;
  background on the Sycamore full vs. elided circuits — <https://www.nature.com/articles/s41586-019-1666-5>
