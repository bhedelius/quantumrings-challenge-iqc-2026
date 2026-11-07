"""Neural-net runtime model: MLP on log-scaled features.

A neural net captures the smooth, multiplicative operations x threshold
interaction better than the tree models (OOF ~0.88 vs ~0.82 for gradient
boosting). Features are log1p-compressed and standardized before the MLP. It is
the self-contained fallback the harness ships alongside TabPFN.
"""

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from tensor_clock.data import build_dataset, load_or_build_features
from tensor_clock.metrics import grouped_cv

N_SEEDS = 5


def _mlp(seed: int) -> Pipeline:
    """One MLP on log1p-compressed, standardized features."""
    return make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1e-4,
                     max_iter=800, early_stopping=True, random_state=seed))


def make_model() -> VotingRegressor:
    """Seed-averaged ensemble of ``N_SEEDS`` MLPs (averaging cuts seed variance)."""
    return VotingRegressor([(f"mlp{s}", _mlp(s)) for s in range(N_SEEDS)],
                           n_jobs=-1)


def main() -> None:
    """Print the grouped-CV score of the MLP ensemble on the cached features."""
    X, y, groups, meta, columns = build_dataset(load_or_build_features())
    print(f"dataset: {len(y)} runs over {len(set(groups))} circuits, "
          f"{X.shape[1]} features")
    fold_scores, overall = grouped_cv(make_model, X, y, groups, meta)
    print(f"GroupKFold(10) per-fold metric: "
          f"{np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"out-of-fold metric (automated 1/3 estimate): {overall:.4f}")


if __name__ == "__main__":
    main()
