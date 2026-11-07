"""Compute model-class comparison data: CV scores and learning curves.

For each model class, computes (1) the 10-fold grouped-CV out-of-fold project
score and (2) a learning curve over a fixed 20%-circuit hold-out (train on
20/40/60/80/100% of the remaining circuits, 3 random draws per fraction below
full). Writes docs/figures/sweep_results.json for the plotting scripts, plus the
Quantum Rings internal-model reference scores (README: CV 0.820, hold-out 0.779).
Metric and splits mirror tensor_clock.metrics.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from tensor_clock import mlp, tabpfn_model
from tensor_clock.data import build_dataset, load_or_build_features
from tensor_clock.metrics import grouped_cv, score

DRAWS = 3
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
HOLDOUT = 0.2
SEED = 0
REFERENCE = {"Quantum Rings validation": 0.820, "Quantum Rings test": 0.779}


def gb_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=500,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=0)


def linreg():
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                         LinearRegression())


def test_score(model, X, y, idx, meta):
    pred = model.predict(X[idx])
    return float(np.mean([score(10.0 ** pred[j], *meta[i])
                          for j, i in enumerate(idx)]))


def learning_curve(make_model, X, y, groups, meta, train_circ, test_idx):
    rng = np.random.default_rng(SEED)
    curve = []
    for frac in FRACTIONS:
        n = max(1, round(frac * len(train_circ)))
        draws = 1 if frac == 1.0 else DRAWS
        scores, n_runs = [], []
        for _ in range(draws):
            pick = set(rng.choice(train_circ, size=n, replace=False))
            tr = np.array([i for i, g in enumerate(groups) if g in pick])
            model = make_model()
            model.fit(X[tr], y[tr])
            scores.append(test_score(model, X, y, test_idx, meta))
            n_runs.append(len(tr))
        curve.append({"frac": frac, "n_circuits": n,
                      "n_runs": int(np.mean(n_runs)),
                      "score": float(np.mean(scores)),
                      "std": float(np.std(scores))})
    return curve


def main():
    X, y, groups, meta, cols = build_dataset(load_or_build_features())
    print(f"dataset {X.shape} over {len(set(groups))} circuits", flush=True)

    models = {
        "Linear regression": linreg,
        "Gradient boosting": gb_model,
        "MLP ensemble": mlp.make_model,
        "TabPFN": tabpfn_model.make_model,
    }

    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT, random_state=SEED)
    tr_all, test_idx = next(gss.split(X, y, groups))
    train_circ = np.array(sorted(set(groups[tr_all])))
    print(f"hold-out: {len(set(groups[test_idx]))} circuits / {len(test_idx)} runs; "
          f"pool {len(train_circ)} circuits", flush=True)

    results = {"meta": {"draws": DRAWS, "fractions": list(FRACTIONS),
                        "holdout": HOLDOUT, "n_test_runs": len(test_idx),
                        "n_test_circuits": len(set(groups[test_idx]))},
               "reference": REFERENCE, "cv": {}, "curves": {}}

    for name, mk in models.items():
        t = time.time()
        fold, oof = grouped_cv(mk, X, y, groups, meta)
        results["cv"][name] = {"oof": float(oof),
                               "fold_mean": float(np.mean(fold)),
                               "fold_std": float(np.std(fold))}
        print(f"[{name}] 10-fold OOF = {oof:.4f}  ({time.time()-t:.0f}s)", flush=True)

        t = time.time()
        results["curves"][name] = learning_curve(mk, X, y, groups, meta,
                                                 train_circ, test_idx)
        print(f"[{name}] learning curve done  ({time.time()-t:.0f}s)", flush=True)

    out = Path(__file__).resolve().parent / "sweep_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
