"""Foundation-model runtime model: TabPFN on log-scaled features.

TabPFN (a pretrained tabular transformer) matches or beats the tuned models with
no tuning (OOF ~0.90) and is the expected model to run. n_estimators=4 keeps
predict under the 15s cap at essentially full accuracy; drop it to 1-2 for more
latency margin. Needs the `tabpfn` package and a one-time TABPFN_TOKEN to download
its weights (we do not redistribute them); the harness falls back to the MLP when
they are unavailable. Module is named tabpfn_model (not tabpfn) so it does not
shadow the library.
"""

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from tensor_clock.data import build_dataset, load_or_build_features
from tensor_clock.metrics import grouped_cv


def make_model() -> Pipeline:
    """TabPFN on log1p-compressed, standardized features.

    ``tabpfn`` is imported lazily so the package stays importable without it.
    """
    from tabpfn import TabPFNRegressor
    # fit_mode="fit_with_cache" encodes the training context once at fit (done in
    # the harness __init__, outside the per-circuit cap) and reuses it, so each
    # predict is ~0.1s instead of ~9s -- fast and within the 15s budget.
    return make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        TabPFNRegressor(device="cpu", n_estimators=4, random_state=0,
                        fit_mode="fit_with_cache"))


def main() -> None:
    """Print the grouped-CV score of the TabPFN model on the cached features."""
    X, y, groups, meta, columns = build_dataset(load_or_build_features())
    print(f"dataset: {len(y)} runs over {len(set(groups))} circuits, "
          f"{X.shape[1]} features")
    fold_scores, overall = grouped_cv(make_model, X, y, groups, meta)
    print(f"GroupKFold(10) per-fold metric: "
          f"{np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"out-of-fold metric (automated 1/3 estimate): {overall:.4f}")


if __name__ == "__main__":
    main()
