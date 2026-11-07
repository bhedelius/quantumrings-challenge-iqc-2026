"""Scoring metric and cross-validation for runtime models.

Reusable across models: `score` is the README metric, `grouped_cv` evaluates a
model factory with GroupKFold so every circuit's rows stay in one fold.
"""

import math
from collections.abc import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

CAP = 21600.0  # 6-hour timeout cap (seconds); a timeout scores against this

Meta = tuple[float, bool]  # (actual_seconds, is_timeout) for one run


def score(pred_s: float, actual_s: float, is_timeout: bool) -> float:
    """Project metric for one run: ``max(0, 1 - |log10(pred/actual)|)``.

    Args:
        pred_s: Predicted runtime in seconds.
        actual_s: Measured runtime in seconds (ignored when ``is_timeout``).
        is_timeout: Whether the run hit the 6-hour cap; then both prediction and
            target are clamped to ``CAP``, so any prediction >= CAP scores 1.

    Returns:
        A score in [0, 1]; 1 is exact, 0 is off by 10x or more.
    """
    actual = CAP if is_timeout else actual_s
    pred = min(pred_s, CAP) if is_timeout else pred_s
    return max(0.0, 1.0 - abs(math.log10(max(pred, 1e-9) / actual)))


def grouped_cv(make_model: Callable[[], object], X: np.ndarray, y: np.ndarray,
               groups: np.ndarray, meta: list[Meta],
               n_splits: int = 10) -> tuple[list[float], float]:
    """Cross-validate a model, grouping rows by circuit so none leak across folds.

    Args:
        make_model: Zero-argument factory returning a fresh unfitted estimator
            (one is trained per fold).
        X: Feature matrix, one row per (circuit, threshold).
        y: Target, ``log10(duration)``.
        groups: Circuit id per row; all of a circuit's rows share a fold.
        meta: ``(actual_seconds, is_timeout)`` per row, for scoring.
        n_splits: Number of ``GroupKFold`` folds.

    Returns:
        ``(fold_scores, overall)``: the per-fold mean project score and the pooled
        out-of-fold project score.
    """
    oof = np.zeros(len(y))
    fold_scores = []
    for train, val in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model()
        model.fit(X[train], y[train])
        oof[val] = model.predict(X[val])
        fold_scores.append(np.mean([score(10.0 ** oof[i], *meta[i])
                                    for i in val]))
    overall = np.mean([score(10.0 ** oof[i], *meta[i]) for i in range(len(y))])
    return fold_scores, overall
