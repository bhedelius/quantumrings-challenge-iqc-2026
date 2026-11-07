"""Exploratory candidate features (the superset we pruned from).

A wider feature set evaluated before settling on the 32 in features.py: SupermarQ
benchmark metrics, qubit utilization, custom-gate counts, and degree/distance
statistics. Kept for the record and reproducibility; the production pipeline does
not import it. See feature_selection.md for what was dropped and why. Reformatted
from the team's exploration to our conventions, reusing features._CLIFFORD (so the
controlled-H-is-not-Clifford bug is fixed) and scipy for connected components.
"""

import math
import re
from collections.abc import Iterable

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .features import _CLIFFORD
from .qasm_text import _DECL_Q2, _DECL_Q3, _DECL_Q3_1, _LEAD, _PARAMS, _QREF

_DECL_C = re.compile(r"^\s*(?:creg\s+\w+|bit)\s*\[\s*(\d+)\s*\]")
_SKIP_LEAD = {"openqasm", "include", "qubit", "qreg", "bit", "creg", "int",
              "uint", "float", "angle", "const", "if", "else", "for", "while",
              "def", "return", "input", "output", "stretch", "gphase", "pragma"}


def _std(values: Iterable[float]) -> float:
    """Population standard deviation; 0 for an empty input."""
    vals = list(values)
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def candidate_features(text: str) -> dict[str, float]:
    """Compute the broad candidate feature set from QASM text in one streaming pass.

    `feature_selection.md` maps each entry to its verdict (kept, duplicate,
    degenerate, leakage, or noise).
    """
    offset, n_q, n_clbits = {}, 0, 0
    total_ops = n_1q = n_2q = n_multi = n_meas = n_reset = n_barrier = n_param = 0
    n_clifford = n_nonclifford = 0
    custom_defs = custom_internal = custom_calls = 0
    custom_bodies = {}            # def name -> internal op count
    clock, clock_2q = {}, {}      # ASAP layer per qubit (all gates / 2q only)
    n_1q_on, first_op, last_op, degree = {}, {}, {}, {}
    pairs = {}                    # (lo, hi) -> occurrence count
    dists = []                    # one 1-D distance per two-qubit gate
    in_def, brace, cur_def = False, 0, None

    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if in_def:
            brace += line.count("{") - line.count("}")
            body = line.split("{")[-1]
            lead = _LEAD.match(body.strip())
            if lead and lead.group(1).lower() not in ("barrier",):
                custom_bodies[cur_def] = custom_bodies.get(cur_def, 0) + 1
            if brace <= 0:
                in_def = False
            continue
        if line.startswith("gate ") or line.startswith("opaque "):
            cur_def = line.split()[1].split("(")[0]
            custom_defs += 1
            custom_bodies.setdefault(cur_def, 0)
            brace = line.count("{") - line.count("}")
            in_def = not (brace <= 0 and "{" in line)
            continue

        m3, m2 = _DECL_Q3.match(line), _DECL_Q2.match(line)
        if m3:
            offset[m3.group(2)] = n_q; n_q += int(m3.group(1)); continue
        if m2:
            offset[m2.group(1)] = n_q; n_q += int(m2.group(2)); continue
        if _DECL_Q3_1.match(line) and "[" not in line:
            offset[_DECL_Q3_1.match(line).group(1)] = n_q; n_q += 1; continue
        mc = _DECL_C.match(line)
        if mc:
            n_clbits += int(mc.group(1)); continue

        lead = _LEAD.match(line)
        token = lead.group(1).lower() if lead else ""
        if "measure" in line:
            n_meas += 1; total_ops += 1; continue
        if token == "reset":
            n_reset += 1; total_ops += 1; continue
        if token == "barrier":
            n_barrier += 1; continue
        if token in _SKIP_LEAD or "=" in line or token == "":
            continue

        qubits = [offset[nm] + int(ix)
                  for nm, ix in _QREF.findall(line) if nm in offset]
        k = len(qubits)
        if k == 0:
            continue
        total_ops += 1
        if _PARAMS.match(line):
            n_param += 1
        if token in _CLIFFORD:
            n_clifford += 1
        else:
            n_nonclifford += 1
        if token in custom_bodies:
            custom_calls += 1

        level = 1 + max((clock.get(q, 0) for q in qubits), default=0)
        for q in qubits:
            clock[q] = level
            first_op.setdefault(q, level)
            last_op[q] = level
        if k == 1:
            n_1q += 1
            n_1q_on[qubits[0]] = n_1q_on.get(qubits[0], 0) + 1
        elif k == 2:
            n_2q += 1
            lo, hi = min(qubits), max(qubits)
            pairs[(lo, hi)] = pairs.get((lo, hi), 0) + 1
            dists.append(hi - lo)
            degree[lo] = degree.get(lo, 0) + 1
            degree[hi] = degree.get(hi, 0) + 1
            lvl2 = 1 + max(clock_2q.get(lo, 0), clock_2q.get(hi, 0))
            clock_2q[lo] = clock_2q[hi] = lvl2
        else:
            n_multi += 1
            for q in qubits:
                degree[q] = degree.get(q, 0) + 1

    depth = max(clock.values(), default=0)
    depth_2q = max(clock_2q.values(), default=0)
    depth_1q = max(n_1q_on.values(), default=0)
    n_active = len(first_op)
    possible = n_q * (n_q - 1) / 2 if n_q > 1 else 1.0
    comps = _components(pairs, n_q)
    half = n_q / 2.0
    bisection = sum(c for (lo, hi), c in pairs.items() if lo < half <= hi)
    lifetimes = sum(last_op[q] - first_op[q] + 1 for q in first_op)
    custom_internal = sum(custom_bodies.values())

    return {
        "file_size_bytes": len(text),
        "file_length_lines": text.count("\n") + 1,
        "num_qubits": n_q,
        "num_clbits": n_clbits,
        "active_qubits": n_active,
        "idle_qubits": n_q - n_active,
        "qubit_utilization_ratio": n_active / n_q if n_q else 0.0,
        "total_ops": total_ops,
        "circuit_depth": depth,
        "depth_1q": depth_1q,
        "depth_2q": depth_2q,
        "parallelism": total_ops / depth if depth else 0.0,
        "active_qubit_parallelism": total_ops / (depth * n_active)
        if depth and n_active else 0.0,
        "num_1q_gates": n_1q,
        "num_2q_gates": n_2q,
        "num_multi_q_gates": n_multi,
        "num_clifford_gates": n_clifford,
        "num_non_clifford_gates": n_nonclifford,
        "non_clifford_ratio": n_nonclifford / total_ops if total_ops else 0.0,
        "num_parameterized_gates": n_param,
        "num_measurements": n_meas,
        "num_resets": n_reset,
        "num_barriers": n_barrier,
        "custom_gate_defs_count": custom_defs,
        "custom_gate_internal_ops": custom_internal,
        "custom_gate_invocations": custom_calls,
        "unique_interacting_pairs": len(pairs),
        "graph_density": len(pairs) / possible if n_q > 1 else 0.0,
        "num_connected_components": comps,
        "qubit_degree_max": max(degree.values(), default=0),
        "qubit_degree_mean": sum(degree.values()) / n_q if n_q else 0.0,
        "qubit_degree_std": _std(degree.values()),
        "interaction_distance_mean": sum(dists) / len(dists) if dists else 0.0,
        "interaction_distance_max": max(dists, default=0),
        "interaction_distance_sum": sum(dists),
        "bisection_cut_size": bisection,
        "supermarq_program_communication": len(pairs) / possible if n_q > 1 else 0.0,
        "supermarq_critical_depth": depth_2q / depth if depth else 0.0,
        "supermarq_entanglement_ratio": n_2q / total_ops if total_ops else 0.0,
        "supermarq_liveness": lifetimes / (n_active * depth)
        if n_active and depth else 0.0,
        "supermarq_measurement_ratio": n_meas / n_q if n_q else 0.0,
    }


def _components(pairs: Iterable[tuple[int, int]], n: int) -> int:
    """Connected components of the interaction graph over n qubits (scipy)."""
    if n < 1:
        return 0
    if not pairs:
        return n
    rows, cols = zip(*pairs)
    adj = csr_matrix(([1] * len(pairs), (rows, cols)), shape=(n, n))
    return int(connected_components(adj, directed=False)[0])
