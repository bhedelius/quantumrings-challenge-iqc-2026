"""Learning-curve figure: TabPFN hold-out score vs number of training circuits.

House style mirrors the manuscript figures (single-column width, 7/6/5 pt type
scale, no top/right spines, blue series, dashed gray reference, saved as both
.pdf and .png cropped to content). Data is read from sweep_results.json, written
by model_sweep.py (32-feature set, fixed 20%-circuit hold-out, 3 random draws per
training fraction). A curve still rising at full data means the model is
data-limited, not at a feature ceiling.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

# --- house style ---
SINGLE_COLUMN = 3.5
LABEL_PT, TICK_PT, FINE_PT = 7, 6, 5
BLUE = "#2a78d6"
RULE = "0.72"
STYLE = {
    "figure.dpi": 150, "savefig.dpi": 400,
    "font.size": LABEL_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": TICK_PT,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.5, "errorbar.capsize": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "sweep_results.json"


def save(figure, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    content = figure.get_tightbbox(figure.canvas.get_renderer())
    width, _ = figure.get_size_inches()
    crop = Bbox.from_extents(0.0, content.y0 - 0.02, width, content.y1 + 0.02)
    for suffix in (".pdf", ".png"):
        figure.savefig(out.with_suffix(suffix), bbox_inches=crop)


def main():
    data = json.loads(RESULTS.read_text())
    curve = data["curves"]["TabPFN"]
    circuits = np.array([p["n_circuits"] for p in curve])
    score = np.array([p["score"] for p in curve])
    std = np.array([p["std"] for p in curve])
    cv_ref = data["cv"]["TabPFN"]["oof"]
    gain = score[-1] - score[-2]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.82))

    ax.axhline(cv_ref, color=RULE, ls=(0, (4, 2)), lw=0.8, zorder=1)
    ax.annotate(f"10-fold CV ({cv_ref:.3f})", xy=(circuits[0], cv_ref),
                xytext=(0, 2), textcoords="offset points", fontsize=FINE_PT,
                color="0.4", va="bottom")

    ax.errorbar(circuits, score, yerr=std, color=BLUE, lw=1.1, marker="o",
                ms=3.6, mew=0.0, capsize=1.5, zorder=3,
                label="TabPFN hold-out (mean $\\pm$ s.d., 3 draws)")

    ax.annotate(f"+{gain:.3f}", xy=(circuits[-1], score[-1]), xytext=(-2, -9),
                textcoords="offset points", fontsize=FINE_PT, color=BLUE, ha="right")

    ax.set_xlabel("Training circuits")
    ax.set_ylabel("Hold-out score")
    lo = min(score.min() - 0.02, cv_ref - 0.02)
    ax.set_ylim(np.floor(lo * 50) / 50, np.ceil((cv_ref + 0.01) * 50) / 50)
    ax.set_xlim(0, circuits[-1] + 20)
    ax.legend(loc="lower right", handlelength=1.4, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.4)

    save(fig, HERE / "learning_curve")
    print("wrote learning_curve.{pdf,png}")


if __name__ == "__main__":
    main()
