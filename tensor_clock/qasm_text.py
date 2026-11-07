"""Streaming text parser for circuit features.

The parser for Tensor Clock: it reads OpenQASM 2.0/3.0 line by line, never builds
a circuit object, and assembles the 32-feature dict (`FEATURE_NAMES`, defined at
the bottom). It has no Qiskit dependency. Gates are counted at the top level;
custom gate-definition bodies are skipped, and whole-register broadcasts (e.g.
``x q;`` applied to a whole register) are not expanded. It was validated to match
a Qiskit parse on model accuracy (within cross-validation noise); see
docs/feature_selection.md.
"""

import math
import re
from collections.abc import Iterable

from tensor_clock.features import (_ANGLE_BINS, _CLIFFORD, _entropy, _on_clifford_grid,
                            degree_cv, entangling_weight, layer_periodicity,
                            n_connected_components, reordered_edge_span)

_DECL_Q3 = re.compile(r"^\s*qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)")
_DECL_Q3_1 = re.compile(r"^\s*qubit\s+([A-Za-z_]\w*)")
_DECL_Q2 = re.compile(r"^\s*qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
_LEAD = re.compile(r"^\s*([A-Za-z_]\w*)")
_QREF = re.compile(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
_PARAMS = re.compile(r"^\s*[A-Za-z_]\w*\s*\(([^)]*)\)")
_SAFE_EXPR = re.compile(r"^[\d\s.+\-*/()eEpi]+$")

_SKIP_LEAD = {
    "OPENQASM", "include", "qubit", "qreg", "bit", "creg", "int", "uint",
    "float", "angle", "const", "gate", "def", "if", "else", "for", "while",
    "box", "barrier", "reset", "return", "input", "output", "stretch",
    "defcal", "defcalgrammar", "cal", "pragma", "duration",
}


def _as_float(token: str) -> float | None:
    """Parse one gate-angle token to a float, allowing simple ``pi`` expressions.

    Returns None for non-numeric or symbolic tokens so the caller can skip them.
    """
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        if _SAFE_EXPR.match(token):
            try:
                return float(eval(token, {"__builtins__": {}}, {"pi": math.pi}))
            except Exception:
                return None
        return None


def text_features(lines: Iterable[str]) -> dict[str, float]:
    """Parse OpenQASM 2.0/3.0 source lines into the Tensor Clock feature dict.

    One streaming pass: tracks qubit-register offsets, an ASAP layer per qubit, the
    interaction graph, and rotation-angle statistics, never building a circuit
    object.

    Args:
        lines: The circuit's QASM text as an iterable of lines.

    Returns:
        The feature dict keyed by ``FEATURE_NAMES`` (all values numeric).
    """
    offset, next_q = {}, 0
    n_1q = n_2q = n_multi = n_meas = n_clifford = n_nonclifford = 0
    n_t = n_cx = 0
    span_sum = span_max = near_zero = n_angles = angle_total = n_param_gates = 0
    entangling_sum = 0.0
    grid_hits = n_lines = 0
    angle_bins = [0] * _ANGLE_BINS
    edges, degree = set(), {}
    layer, q2layer, cut_delta, layer_count = {}, {}, {}, {}
    seen_measure = is_dynamic = 0
    in_def = False
    brace = 0

    for raw in lines:
        n_lines += 1
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue

        if in_def:  # inside a gate definition body (brace may open on a later line)
            brace += line.count("{") - line.count("}")
            if brace <= 0:
                in_def = False
            continue

        m3 = _DECL_Q3.match(line)
        m2 = _DECL_Q2.match(line)
        if m3:
            offset[m3.group(2)] = next_q
            next_q += int(m3.group(1))
            continue
        if m2:
            offset[m2.group(1)] = next_q
            next_q += int(m2.group(2))
            continue
        if _DECL_Q3_1.match(line) and "[" not in line:
            offset[_DECL_Q3_1.match(line).group(1)] = next_q
            next_q += 1
            continue

        if line.startswith("gate ") or line.startswith("opaque "):
            brace = line.count("{") - line.count("}")
            in_def = not (brace <= 0 and "{" in line)  # single-line def ends now
            continue

        lead = _LEAD.match(line)
        token = lead.group(1) if lead else ""
        if token in ("if", "else", "for", "while"):
            is_dynamic = 1
        if "measure" in line:
            n_meas += 1
            seen_measure = 1
            for name, idx in _QREF.findall(line):
                if name in offset:  # advance depth on measured qubits only
                    q = offset[name] + int(idx)
                    layer[q] = 1 + layer.get(q, 0)
            continue
        if token in _SKIP_LEAD or "=" in line or token == "":
            continue
        if seen_measure:
            is_dynamic = 1

        qubits = [offset[name] + int(idx)
                  for name, idx in _QREF.findall(line) if name in offset]
        k = len(qubits)
        if k == 0:
            continue

        pm = _PARAMS.match(line)
        angles = []
        if pm:
            for part in pm.group(1).split(","):
                value = _as_float(part)
                if value is not None:
                    angles.append(value)

        if k == 1:
            n_1q += 1
        elif k == 2:
            lo, hi = min(qubits), max(qubits)
            span_sum += hi - lo + 1
            span_max = max(span_max, hi - lo)
            edges.add((lo, hi))
            degree[lo] = degree.get(lo, 0) + 1
            degree[hi] = degree.get(hi, 0) + 1
            cut_delta[lo + 1] = cut_delta.get(lo + 1, 0) + 1
            cut_delta[hi + 1] = cut_delta.get(hi + 1, 0) - 1
            entangling_sum += entangling_weight(token, angles)
            lvl2 = 1 + max(q2layer.get(lo, 0), q2layer.get(hi, 0))
            q2layer[lo] = q2layer[hi] = lvl2
            n_2q += 1
        else:
            n_multi += 1

        n_clifford += token in _CLIFFORD
        n_nonclifford += token not in _CLIFFORD
        n_t += token in ("t", "tdg")
        n_cx += token in ("cx", "cnot")
        had_param = False
        for value in angles:
            wrapped = abs((value + math.pi) % (2 * math.pi) - math.pi)
            angle_total += wrapped
            n_angles += 1
            near_zero += wrapped < 1e-3
            grid_hits += _on_clifford_grid(wrapped)
            angle_bins[min(int(wrapped / math.pi * _ANGLE_BINS), _ANGLE_BINS - 1)] += 1
            had_param = True
        n_param_gates += had_param

        level = 1 + max((layer.get(q, 0) for q in qubits), default=0)
        for q in qubits:
            layer[q] = level
        layer_count[level] = layer_count.get(level, 0) + 1

    n = next_q
    n_gates = n_1q + n_2q + n_multi
    running = max_cut = bisection_cut = 0
    mid = (n + 1) // 2
    for p in range(n + 2):
        running += cut_delta.get(p, 0)
        max_cut = max(max_cut, running)
        if p == mid:
            bisection_cut = running

    return {
        "n_qubits": n,
        "depth": max(layer.values(), default=0),
        "n_1q": n_1q,
        "n_2q": n_2q,
        "n_multi": n_multi,
        "n_meas": n_meas,
        "n_t": n_t,
        "n_cx": n_cx,
        "span_sum": span_sum,
        "span_max": span_max,
        "span_mean": (span_sum - n_2q) / n_2q if n_2q else 0.0,
        "entangling_sum": entangling_sum,
        "max_cut": max_cut,
        "reordered_edge_span": reordered_edge_span(edges, n),
        "n_edges": len(edges),
        "graph_density": len(edges) / (n * (n - 1) / 2) if n > 1 else 0.0,
        "max_degree": max(degree.values(), default=0),
        "mean_degree": sum(degree.values()) / len(degree) if degree else 0.0,
        "frac_2q": n_2q / n_gates if n_gates else 0.0,
        "frac_nonclifford": n_nonclifford / n_gates if n_gates else 0.0,
        "n_param_gates": n_param_gates,
        "frac_near_identity": near_zero / n_angles if n_angles else 0.0,
        "angle_mean": angle_total / n_angles if n_angles else 0.0,
        "frac_angle_clifford_grid": grid_hits / n_angles if n_angles else 0.0,
        "angle_entropy": _entropy(angle_bins),
        "degree_cv": degree_cv(degree.values()),
        "layer_periodicity": layer_periodicity(layer_count),
        "depth_2q": max(q2layer.values(), default=0),
        "n_components": n_connected_components(edges, n),
        "bisection_cut": bisection_cut,
        "file_length_lines": n_lines,
        "is_dynamic": is_dynamic,
    }


FEATURE_NAMES = list(text_features([]).keys())
