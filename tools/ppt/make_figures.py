"""Re-plot the two figures the deck uses, in the deck's own palette.

The originals in docs/figures are matplotlib defaults: blue-and-red bars inside a boxed
grid. They read fine in the evidence pack but clash with the deck and, worse, the default
styling is itself a giveaway. These carry exactly the same measured numbers -- lifted from
docs/evidence-pack.md, not recomputed -- with the chart furniture stripped back so the
data is the only thing on the page.

    python tools/ppt/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("D:/sih2026/depthwizard/tools/ppt/figures")
INK, SLATE, ACCENT, STEEL, RULE = "#14181D", "#5B6670", "#C1440E", "#2C4A63", "#D3D8DD"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def bare(ax, keep_left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def error_by_height() -> None:
    """Where the error actually lives. 79% of squared error, 1.9% of the buildings."""
    bands = ["0–3 m", "3–6 m", "6–10 m", "10–20 m", "> 20 m"]
    rmse = [1.282, 1.429, 1.947, 4.034, 23.451]
    n = [166, 2305, 443, 116, 60]

    fig, ax = plt.subplots(figsize=(6.4, 2.55), dpi=220)
    cols = [STEEL] * 4 + [ACCENT]
    bars = ax.bar(bands, rmse, color=cols, width=0.62)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.2f} m",
                ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_ylim(0, 27)
    ax.set_ylabel("Per-building RMSE (m)", fontsize=8.5)
    bare(ax)
    # Sample size belongs on the axis, not floating under the bars, where it collided
    # with the tick labels.
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{b}\nn={c:,}" for b, c in zip(bands, n)],
                       fontsize=8.5, color=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "error_by_height.png", transparent=False)
    plt.close(fig)


def terrain() -> None:
    """Stability across the four landscape classes ISRO names."""
    names = ["Sparse", "Mixed", "Forested", "Urban"]
    rmse = [1.837, 2.723, 3.319, 13.860]

    fig, ax = plt.subplots(figsize=(4.5, 2.15), dpi=220)
    cols = [STEEL] * 3 + [ACCENT]
    bars = ax.barh(names[::-1], rmse[::-1], color=cols[::-1], height=0.58)
    for b, v in zip(bars, rmse[::-1]):
        ax.text(v + 0.35, b.get_y() + b.get_height() / 2, f"{v:.2f} m",
                va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xlim(0, 17)
    ax.set_xlabel("Whole-tile RMSE (m)", fontsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "terrain.png", transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    error_by_height()
    terrain()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size / 1024:.0f} KB")
