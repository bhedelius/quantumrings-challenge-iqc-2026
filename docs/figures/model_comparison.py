"""Model-class comparison figures from sweep_results.json (model_sweep.py).

Two figures in the manuscript house style:
  model_comparison       overlaid hold-out learning curves, one line per model
                         class (every class is still rising -> data-limited, and
                         TabPFN sits above the rest at every training size).
  model_comparison_bars  10-fold grouped-CV out-of-fold score per model class.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

SINGLE_COLUMN = 3.5
LABEL_PT, TICK_PT, FINE_PT = 7, 6, 5
STYLE = {
    "figure.dpi": 150, "savefig.dpi": 400,
    "font.size": LABEL_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": FINE_PT,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.0, "errorbar.capsize": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# model classes shown in the comparison (one color + marker each)
COLORS = {
    "TabPFN": "#2a78d6", "MLP ensemble": "#1baf7a",
    "Gradient boosting": "#8a63d2", "Linear regression": "#eb6834",
}
MARKERS = {"TabPFN": "o", "MLP ensemble": "s",
           "Gradient boosting": "D", "Linear regression": "^"}
MODELS = list(COLORS)
REF_COLOR = "0.45"

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


def curves_figure(data):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.82))
    xmax = max(p["n_circuits"] for p in data["curves"]["TabPFN"])

    # Quantum Rings internal-model reference scores (README).
    ref = data.get("reference", {})
    ymin = 1.0
    for label, ls in (("Quantum Rings validation", (0, (4, 2))),
                      ("Quantum Rings test", (0, (1, 1.5)))):
        if label in ref:
            v = ref[label]
            ax.axhline(v, color=REF_COLOR, ls=ls, lw=0.8, zorder=1)
            short = label.replace("Quantum Rings ", "QR ")
            ax.annotate(f"{short} ({v:.2f})", xy=(xmax, v), xytext=(0, 1.5),
                        textcoords="offset points", fontsize=FINE_PT,
                        color="0.4", ha="right", va="bottom")

    order = sorted((m for m in MODELS if m in data["curves"]),
                   key=lambda m: -data["cv"][m]["oof"])
    for name in order:
        c = data["curves"][name]
        x = np.array([p["n_circuits"] for p in c])
        yv = np.array([p["score"] for p in c])
        sd = np.array([p["std"] for p in c])
        ymin = min(ymin, (yv - sd).min())
        ax.errorbar(x, yv, yerr=sd, color=COLORS[name], marker=MARKERS[name],
                    ms=3.0, mew=0.0, lw=1.0, capsize=1.2, label=name, zorder=3)

    ax.set_xlabel("Training circuits")
    ax.set_ylabel("Hold-out score")
    ax.set_xlim(0, xmax + 20)
    ax.set_ylim(np.floor((ymin - 0.02) * 20) / 20, 0.93)
    ax.legend(loc="center right", handlelength=1.4, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, HERE / "model_comparison")
    print("wrote model_comparison.{pdf,png}")


def bars_figure(data):
    plt.rcParams.update(STYLE)
    names = sorted((m for m in MODELS if m in data["cv"]),
                   key=lambda m: data["cv"][m]["oof"])
    vals = [data["cv"][m]["oof"] for m in names]
    colors = ["#2a78d6" if m == "TabPFN" else "#9fb6cf" for m in names]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.72))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.66, zorder=3)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3f}", xy=(v, yi), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=FINE_PT,
                    color="0.2")
    qr = data.get("reference", {}).get("Quantum Rings validation")
    if qr:
        ax.axvline(qr, color=REF_COLOR, ls=(0, (4, 2)), lw=0.8, zorder=2)
        ax.annotate(f"Quantum Rings\nvalidation ({qr:.2f})", xy=(qr, 0),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=FINE_PT, color="0.4")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("10-fold CV score (out-of-fold)")
    ax.set_xlim(0, 1.0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4)
    save(fig, HERE / "model_comparison_bars")
    print("wrote model_comparison_bars.{pdf,png}")


def main():
    data = json.loads(RESULTS.read_text())
    curves_figure(data)
    bars_figure(data)


if __name__ == "__main__":
    main()
