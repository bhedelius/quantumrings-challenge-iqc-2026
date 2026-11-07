"""
model.py  --  Tensor Clock submission model.

The harness (run.py) calls:
  RuntimeModel()                      fit the model from the committed feature cache
  .featurize(qasm_text)   -> dict     parse one circuit into 32 features
  .predict(features, thr) -> seconds  predicted wall-clock runtime

featurize and predict share tensor_clock's parser and training pipeline, so
inference matches training exactly. Models are fit at construction from the
committed feature cache -- no weight files are shipped. TabPFN is the expected
model; it downloads its own weights via the tabpfn package + a TABPFN_TOKEN (we do
not redistribute them), and the harness falls back to the self-contained MLP when
TabPFN is unavailable. The model predicts log10(duration); we invert and cap at
the 6-hour timeout.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import tensor_clock

from tensor_clock.qasm_text import FEATURE_NAMES, text_features  # noqa: E402

CAP_SECONDS = 21600.0  # 6-hour timeout cap


class RuntimeModel:
    """Runtime predictor fit at construction from the committed feature cache.

    Prefers TabPFN and falls back to the self-contained MLP if TabPFN (package or
    weights) is unavailable. All fitting happens in ``__init__``, which the harness
    calls once outside the per-circuit time budget.
    """

    def __init__(self, artifacts_dir: str | None = None):
        """Fit the model; ``artifacts_dir`` is accepted for harness compatibility
        but unused (nothing is loaded from disk)."""
        from tensor_clock import mlp, tabpfn_model
        from tensor_clock.data import build_dataset, load_or_build_features

        X, y, *_ = build_dataset(load_or_build_features())
        probe = np.zeros((1, X.shape[1]))
        for kind, module in (("tabpfn", tabpfn_model), ("mlp", mlp)):
            try:
                model = module.make_model()
                model.fit(X, y)
                model.predict(probe)  # validate it runs (package, weights, deps)
                self.model, self.kind = model, kind
                break
            except Exception as e:
                print(f"  {kind} unavailable ({type(e).__name__}); trying next")
        else:
            raise RuntimeError("no usable model (TabPFN or MLP) could be fit")
        print(f"using {self.kind} model")

    def featurize(self, qasm_text: str) -> dict[str, float]:
        """Parse one circuit's QASM text into the 32-feature dict."""
        return text_features(qasm_text.splitlines())

    def predict(self, features: dict[str, float], threshold: int) -> float:
        """Predict runtime in seconds for one (circuit, threshold).

        Args:
            features: The dict returned by ``featurize``.
            threshold: The simulator bond-dimension cap for this run.

        Returns:
            Predicted wall-clock seconds, capped at the 6-hour timeout.
        """
        row = [features[k] for k in FEATURE_NAMES] + [threshold, math.log2(threshold)]
        pred_log10 = float(self.model.predict(np.array([row]))[0])
        return float(min(10.0 ** pred_log10, CAP_SECONDS))
