"""Score vs bond dimension: mean out-of-fold project score at each threshold.

The ``threshold`` column is the simulator's bond-dimension cap (chi). A larger cap
allows more entanglement to be represented, so runtime becomes more chi-dependent
and harder to predict -- the score drifts down as the cap grows. Computes the
10-fold grouped-CV OOF with the TabPFN model and renders a bar per threshold.
House style mirrors the other figures; writes .pdf, .png, and a values JSON.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from sklearn.model_selection import GroupKFold

from tensor_clock.data import build_dataset, load_or_build_features
from tensor_clock.metrics import score
from tensor_clock import tabpfn_model

SINGLE_COLUMN = 3.5
LABEL_PT, TICK_PT, FINE_PT = 7, 6, 5
BLUE = "#2a78d6"
STYLE = {
    "figure.dpi": 150, "savefig.dpi": 400,
    "font.size": LABEL_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": TICK_PT,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
HERE = Path(__file__).resolve().parent


def save(figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    content = figure.get_tightbbox(figure.canvas.get_renderer())
    width, _ = figure.get_size_inches()
    crop = Bbox.from_extents(0.0, content.y0 - 0.02, width, content.y1 + 0.02)
    for suffix in (".pdf", ".png"):
        figure.savefig(out.with_suffix(suffix), bbox_inches=crop)


def main() -> None:
    X, y, groups, meta, cols = build_dataset(load_or_build_features())
    thr = X[:, cols.index("threshold")]

    oof = np.zeros(len(y))
    for train, val in GroupKFold(10).split(X, y, groups):
        model = tabpfn_model.make_model()
        model.fit(X[train], y[train])
        oof[val] = model.predict(X[val])
    rs = np.array([score(10.0 ** oof[i], *meta[i]) for i in range(len(y))])

    thrs = sorted(set(thr))
    scores = [float(rs[thr == t].mean()) for t in thrs]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.82))
    xpos = np.arange(len(thrs))
    ax.bar(xpos, scores, width=0.68, color=BLUE)
    for x, s in zip(xpos, scores):
        ax.annotate(f"{s:.3f}", (x, s), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=FINE_PT, color="0.25")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(int(t)) for t in thrs])
    ax.set_xlabel(r"Bond-dimension cap  (threshold $\chi$)")
    ax.set_ylabel("Mean out-of-fold score")
    ax.set_ylim(0.80, 0.94)  # zoomed: all thresholds score ~0.9, with a mild decline
    fig.tight_layout(pad=0.4)

    save(fig, HERE / "score_by_threshold")
    (HERE / "score_by_threshold.json").write_text(
        json.dumps({str(int(t)): s for t, s in zip(thrs, scores)}, indent=2))
    print("wrote score_by_threshold.{pdf,png,json}:",
          {int(t): round(s, 4) for t, s in zip(thrs, scores)})


if __name__ == "__main__":
    main()
