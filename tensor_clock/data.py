"""Feature caching and dataset assembly for model training.

`load_or_build_features` returns per-circuit features (cached to disk), and
`build_dataset` joins them with the per-(circuit, threshold) labels into arrays.
The cache is built serially; featurization is cheap next to the parse,
and a serial pass avoids the memory pressure of many large circuits at once.
"""

import csv
import math
import subprocess
from pathlib import Path

import numpy as np

from .metrics import CAP, Meta
from .qasm_text import FEATURE_NAMES, text_features

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "train-runtime-data.csv"
CIRCUITS = ROOT / "train-circuits"
CACHE = ROOT / "feature_cache.csv"

Cache = dict[str, dict[str, float]]  # circuit filename -> {feature name: value}


def _read_zst(path: Path) -> str:
    """Decompress a ``.qasm.zst`` file to text via the ``zstd`` CLI."""
    out = subprocess.run(["zstd", "-dc", str(path)], capture_output=True)
    return out.stdout.decode("utf-8", "replace")


def build_feature_cache() -> Cache:
    """Featurize every training circuit and write the cache to ``CACHE``.

    Serial on purpose: featurization is cheap next to decompression, and one pass
    avoids holding several large circuits in memory at once. A circuit that fails
    to parse is stored as all-NaN rather than dropped.

    Returns:
        The feature cache (also written to disk as ``feature_cache.csv``).
    """
    paths = sorted((p.name[:-9] + ".qasm", p) for p in CIRCUITS.glob("*.qasm.zst"))
    feats = {}
    for i, (name, path) in enumerate(paths, 1):
        try:
            feats[name] = text_features(_read_zst(path).splitlines())
        except Exception:
            feats[name] = {k: float("nan") for k in FEATURE_NAMES}
        if i % 100 == 0:
            print(f"  featurized {i}/{len(paths)}", flush=True)
    with open(CACHE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + FEATURE_NAMES)
        writer.writeheader()
        for name, row in feats.items():
            writer.writerow({"filename": name, **row})
    return feats


def load_feature_cache() -> Cache:
    """Read the feature cache from ``CACHE``."""
    with open(CACHE) as f:
        return {r["filename"]: {k: float(r[k]) for k in FEATURE_NAMES}
                for r in csv.DictReader(f)}


def load_or_build_features() -> Cache:
    """Return the cached features, building and writing the cache if absent."""
    return load_feature_cache() if CACHE.exists() else build_feature_cache()


def load_labels() -> list[dict]:
    """Return one label row per (circuit, threshold); a success supersedes a timeout."""
    best = {}
    for row in csv.DictReader(open(LABELS)):
        key = (row["filename"].strip(), int(float(row["threshold"])))
        if key not in best or row.get("status", "").strip().lower() == "success":
            best[key] = row
    return list(best.values())


def build_dataset(
    feats: Cache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Meta], list[str]]:
    """Join per-circuit features with the labels into model-ready arrays.

    Args:
        feats: The feature cache (e.g. from ``load_or_build_features``).

    Returns:
        ``(X, y, groups, meta, columns)``: feature matrix (circuit features plus
        ``threshold`` and ``log2_threshold``), target ``log10(duration)``, circuit
        id per row, ``(actual_seconds, is_timeout)`` per row, and the column names.
    """
    columns = FEATURE_NAMES + ["threshold", "log2_threshold"]
    X, y, groups, meta = [], [], [], []
    for lab in load_labels():
        name = lab["filename"].strip()
        if name not in feats:
            continue
        is_timeout = lab.get("status", "").strip().lower() == "timeout"
        actual = CAP if is_timeout else float(lab["duration_s"])
        if actual <= 0:
            continue
        thr = int(float(lab["threshold"]))
        row = feats[name]
        X.append([row[k] for k in FEATURE_NAMES] + [thr, math.log2(thr)])
        y.append(math.log10(actual))
        groups.append(name)
        meta.append((actual, is_timeout))
    return np.array(X), np.array(y), np.array(groups), meta, columns


if __name__ == "__main__":
    print(f"building feature cache from {CIRCUITS} ...")
    build_feature_cache()
    print(f"wrote {CACHE}")
