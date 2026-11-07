"""Shared constants and helpers for the circuit feature parser.

Tensor Clock models runtime as operations x bond-dimension cost; these helpers
compute pieces of that model -- Clifford/entangling gate membership, entangling
power, rotation-angle statistics, and interaction-graph structure (reordered span,
connected components) -- that `qasm_text.text_features` assembles into the feature
dict. See docs/simulator_cost_model.md.
"""

import math
from collections.abc import Iterable

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee

_CLIFFORD = {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "cy", "swap",
             "id", "sx", "sxdg"}
_SKIP = {"measure", "barrier", "reset", "delay", "snapshot"}
_PARAM_2Q = {"cp", "cphase", "crx", "cry", "crz", "cu1", "rxx", "ryy", "rzz",
             "rzx", "xx_plus_yy", "xx_minus_yy"}


_ANGLE_BINS = 18
_QUARTER = math.pi / 4


def _is_num(param: object) -> bool:
    """Whether a gate parameter is a plain number (not a symbolic expression)."""
    try:
        float(param)
        return True
    except (TypeError, ValueError):
        return False


def _on_clifford_grid(wrapped: float, tol: float = 1e-3) -> bool:
    """Whether a [0, pi]-wrapped angle sits on the k*pi/4 (Clifford+T) grid."""
    k = round(wrapped / _QUARTER)
    return abs(wrapped - k * _QUARTER) < tol


def _entropy(counts: Iterable[float]) -> float:
    """Shannon entropy (bits) of a histogram; 0 when empty or single-bin."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def degree_cv(degree_values: Iterable[float]) -> float:
    """Coefficient of variation of interaction-graph degrees (0 if uniform)."""
    vals = list(degree_values)
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / mean


def layer_periodicity(layer_count: dict[int, int], max_lag: int = 64,
                      max_layers: int = 4096) -> float:
    """Peak normalized autocorrelation of the per-layer gate-count sequence.

    High for circuits built from a repeated block (ansatz, Trotter step), low
    for irregular/random circuits. Truncated to the first `max_layers` layers.
    """
    depth = max(layer_count, default=0)
    seq = [layer_count.get(l, 0) for l in range(1, min(depth, max_layers) + 1)]
    d = len(seq)
    if d < 4:
        return 0.0
    mean = sum(seq) / d
    centered = [s - mean for s in seq]
    denom = sum(c * c for c in centered)
    if denom == 0:
        return 0.0
    best = 0.0
    for lag in range(1, min(max_lag, d // 2) + 1):
        num = sum(centered[i] * centered[i + lag] for i in range(d - lag))
        best = max(best, num / denom)
    return best


def entangling_weight(name: str, angles: list[float]) -> float:
    """Approximate entangling power of a two-qubit gate in [0, 1]."""
    if name == "swap":
        return 0.0  # pure permutation, creates no entanglement
    if name in _PARAM_2Q and angles:
        return math.sin(angles[0] / 2.0) ** 2  # 0 at theta=0, 1 at theta=pi
    return 1.0  # cx/cz/ecr/iswap/... treated as strong entanglers


def reordered_edge_span(edges: Iterable[tuple[int, int]], n: int) -> int:
    """Total interaction-graph edge span after a bandwidth-reducing reorder."""
    if not edges or n < 2:
        return 0
    rows, cols = zip(*edges)
    adj = csr_matrix(([1] * len(edges), (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    order = reverse_cuthill_mckee(adj, symmetric_mode=True)
    pos = [0] * n
    for rank, node in enumerate(order):
        pos[node] = rank
    return sum(abs(pos[a] - pos[b]) for a, b in edges)


def n_connected_components(edges: Iterable[tuple[int, int]], n: int) -> int:
    """Connected components of the interaction graph over n qubits (idle count)."""
    if n < 1:
        return 0
    if not edges:
        return n
    rows, cols = zip(*edges)
    adj = csr_matrix(([1] * len(edges), (rows, cols)), shape=(n, n))
    return int(connected_components(adj, directed=False)[0])
