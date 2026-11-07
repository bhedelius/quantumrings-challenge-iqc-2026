# Learning curve: are we data-limited or feature-limited?

A learning curve answers whether the ~0.90 accuracy ceiling comes from too little
training data or from the features/model. We plot held-out accuracy against the
number of training circuits: a curve that is still rising at full data means more
data would help (data-limited); a flat curve means more data would not
(feature/model-limited).

## Method

- **Model:** the production TabPFN pipeline — `log1p` + `StandardScaler` +
  `TabPFNRegressor(n_estimators=4, random_state=0)` on the 32-feature set plus
  `threshold` and `log2_threshold`.
- **Split:** a fixed 20% of circuits held out as the test set (grouped by circuit,
  `GroupShuffleSplit(random_state=0)`), so no circuit's rows appear in both train
  and test. The test set is the same at every training size.
- **Curve:** train on 20/40/60/80/100% of the remaining 80% of circuits,
  subsampling whole circuits (not rows). Each fraction below 100% is averaged over
  3 random circuit draws.
- **Metric:** the project score `max(0, 1 - |log10(pred/actual)|)` on the fixed
  test set, timeouts capped at 21,600 s.
- **Reproduce:** `docs/figures/model_sweep.py` writes `sweep_results.json` (this
  curve plus the per-model comparison); `docs/figures/learning_curve.py` renders it.

## Result

| Train circuits | Train runs | Test score | Gain vs previous |
|---|---|---|---|
| 85 | 421 | 0.7450 | — |
| 170 | 846 | 0.8425 | +0.098 |
| 255 | 1265 | 0.8500 | +0.008 |
| 340 | 1690 | 0.8879 | +0.038 |
| 425 (all) | 2113 | 0.9061 | +0.018 |

The curve is **still rising at 100%**: the final 85 circuits add +0.018, well above
the ~±0.003 feature-noise floor. We are **data-limited**, not at a pure feature
ceiling.

## Interpretation

- **Data-limited and feature-saturated are both true, and consistent.** More
  training examples keep improving the fit (this curve), while adding more features
  does not (every feature experiment landed at the ±0.003 noise floor). The current
  32 features carry signal that more data lets the model exploit better; additional
  features carry no new information. The lever is data/model, not more features —
  which is why we stopped the feature hunt.
- **This is why TabPFN wins.** TabPFN is built for the small-data regime (a
  Bayesian in-context prior). A data-limited problem is exactly where a prior-based
  model beats data-hungry ones, which explains the gap on identical features: 10-fold
  CV TabPFN 0.902, MLP 0.879, gradient boosting 0.830, linear regression 0.421 (see
  `figures/model_comparison`). Linear plateaus by ~170 circuits while TabPFN/MLP keep
  rising — the simple class is model-limited, the strong classes data-limited.
- **Hold-out calibration.** On the fixed 107-circuit hold-out at full training data,
  TabPFN scores **0.906**, close to the 10-fold CV estimate of 0.902. The
  competition's separate hold-out is the real test; for reference, the organizers'
  internal model dropped 0.82 (CV) to 0.78 (hold-out), so expect some gap there.

## Takeaway

The accuracy ceiling is a **data-limited regime where model choice (a small-data
prior) is the dominant lever**, not a flat feature ceiling. More labeled circuits
would help; more structural features would not.
