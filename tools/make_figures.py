"""Figures for the evidence pack.

Same design rules as the viewer, for the same reason -- a SAC jury has read remote-sensing
figures for twenty years and reads sloppiness as a signal:

  * Light background, near-neutral chrome, one accent colour.
  * Never a rainbow/jet colormap. Sequential ramps are monotonic in lightness so they
    survive greyscale printing; the error map is diverging and centred on zero.
  * Every axis carries a unit. Counts are annotated so nobody has to guess the n.
  * The failure is plotted, not cropped out.

    python tools/make_figures.py --metrics out/eval_run02_ship/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

ROOT = Path("D:/sih2026")
ACCENT = "#0b5cab"
INK = "#1a2027"
DIM = "#5b6672"
GRID = "#e2e6ea"
WARN = "#a8442a"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": "#c7ced6", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.dpi": 160,
})


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def fig_by_height(m, out: Path):
    """Where the error actually lives. This is the single most important figure."""
    rows = m["building_wise_by_height"]
    labels = [r.get("band", f"{r.get('lo','?')}-{r.get('hi','?')}") for r in rows]
    rmse = [r["rmse"] for r in rows]
    n = [r["n"] if "n" in r else r.get("n_buildings", 0) for r in rows]
    bias = [r.get("bias", 0.0) for r in rows]

    fig, ax = ax2 = None, None
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    cols = [ACCENT if r < 10 else WARN for r in rmse]
    ax.bar(labels, rmse, color=cols, width=0.62)
    for i, (r, c) in enumerate(zip(rmse, n)):
        ax.text(i, r + max(rmse) * 0.02, f"{r:.2f} m", ha="center", fontsize=8, color=INK)
        ax.text(i, -max(rmse) * 0.085, f"n={c:,}", ha="center", fontsize=7.5, color=DIM)
    ax.set_ylabel("Per-building RMSE (m)")
    ax.set_title("Error by true building height — 94% of buildings are under 10 m",
                 fontsize=10, loc="left", color=INK)
    # Headroom below zero so the counts sit clear of the axis rather than straddling it.
    ax.set_ylim(-max(rmse) * 0.15, max(rmse) * 1.18)

    ax2.bar(labels, bias, color=[ACCENT if b > -5 else WARN for b in bias], width=0.62)
    ax2.axhline(0, color=DIM, lw=0.8)
    ax2.set_ylabel("Bias (m)")
    ax2.set_xlabel("True building height")
    for i, b in enumerate(bias):
        ax2.text(i, b + (0.6 if b >= 0 else -1.6), f"{b:+.2f}", ha="center",
                 fontsize=7.5, color=INK)
    save(fig, out / "fig_error_by_height.png")


def fig_calibration(m, out: Path):
    """Does the model know when it is wrong? Differentiator 6.1 lives or dies here."""
    cal = m.get("calibration") or []
    if not cal:
        return
    exp = [c["expected"] for c in cal]
    obs = [c["observed"] for c in cal]
    # The bins are multiples of sigma (k = 0.5, 1.0, ...), and carry no pixel count, so
    # size the markers by k rather than inventing a weight. An earlier version divided by
    # max(count) and would have raised on a schema that has no count at all.
    ks = [c.get("k", 1.0) for c in cal]
    lim = max(max(exp), max(obs)) * 1.1

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, lim], [0, lim], color=DIM, ls="--", lw=1, label="perfectly calibrated")
    sizes = 24 + 26 * np.asarray(ks, float)
    ax.scatter(exp, obs, s=sizes, color=ACCENT, alpha=0.85, zorder=3,
               label="observed (marker size = σ multiple)")
    for x, y, k in zip(exp, obs, ks):
        ax.annotate(f"{k:g}σ", (x, y), textcoords="offset points", xytext=(7, -3),
                    fontsize=7, color=DIM)
    ax.set_xlabel("Predicted uncertainty σ (m)")
    ax.set_ylabel("Actual RMSE in that bin (m)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ece = m.get("ece")
    rank = m.get("sigma_rank_corr")
    ax.set_title(f"Uncertainty calibration\nECE {ece:.3f}   ·   σ-vs-error rank corr "
                 f"{rank:+.3f}", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out / "fig_calibration.png")


def fig_terrain(m, out: Path):
    """The stability criterion ISRO names, answered directly."""
    per = m.get("per_terrain") or {}
    if not per:
        return
    order = ["urban", "sparse", "forested", "mixed"]
    keys = [k for k in order if k in per] + [k for k in per if k not in order]
    rmse = [per[k]["rmse"] for k in keys]
    # per_terrain records no tile count, so count them from per_tile where the evaluator
    # does record each tile's class. Falling back to "0 tiles" would have printed a label
    # that is simply false.
    counts = {}
    for t in m.get("per_tile", []):
        c = t.get("terrain") or t.get("class")
        if c:
            counts[c] = counts.get(c, 0) + 1
    n = [counts.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(keys, rmse, color=ACCENT, width=0.55)
    for i, (r, c) in enumerate(zip(rmse, n)):
        ax.text(i, r + max(rmse) * 0.02, f"{r:.2f} m", ha="center", fontsize=8.5, color=INK)
        if c:
            ax.text(i, -max(rmse) * 0.07, f"{c} tiles", ha="center", fontsize=7.5, color=DIM)
    ax.set_ylabel("Whole-tile RMSE (m)")
    ax.set_ylim(0, max(rmse) * 1.2)
    ax.set_title("Stability across landscapes — ISRO's stated criterion",
                 fontsize=10, loc="left", color=INK)
    save(fig, out / "fig_terrain.png")


def fig_scatter(npz: Path, out: Path):
    """Every building, ours against LiDAR. The tail collapse is visible, not described."""
    if not npz.exists():
        return
    d = np.load(npz)
    ours, truth = d["ours"], d["truth"]
    lim = max(truth.max(), ours.max()) * 1.05

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, lim], [0, lim], color=DIM, ls="--", lw=1, zorder=1)
    ax.scatter(truth, ours, s=7, color=ACCENT, alpha=0.28, linewidths=0, zorder=2)
    tall = truth > 20
    ax.scatter(truth[tall], ours[tall], s=12, color=WARN, alpha=0.75, linewidths=0,
               zorder=3, label=f"above 20 m (n={tall.sum()})")
    ax.set_xlabel("LiDAR building height (m)")
    ax.set_ylabel("Our building height (m)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{len(ours):,} buildings, region-disjoint validation\n"
                 f"points below the line are under-called", fontsize=10, loc="left",
                 color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out / "fig_buildings_scatter.png")


def _ramp(a, lo, hi, stops, gamma=1.0):
    """Map values through a ramp. gamma < 1 gives the low end more of the colour range.

    Needed because one 93.6 m building sets the top of the scale while 94% of the stock is
    under 10 m: on a linear ramp the entire urban fabric collapses into one pale tone. The
    figure says which scale it is using, because a nonlinear ramp that does not announce
    itself is a way of overstating agreement.
    """
    t = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    if gamma != 1.0:
        t = t ** gamma
    stops = np.asarray(stops, float)
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def fig_error_map(tile: str, out: Path):
    """Imagery, LiDAR, ours, and the signed difference. The figure a specialist reads first."""
    import rasterio
    import warnings
    warnings.simplefilter("ignore")
    rgb_p = ROOT / "data/extracted/Track1-RGB" / f"{tile}_RGB.tif"
    agl_p = ROOT / "data/extracted/Track1-Truth" / f"{tile}_AGL.tif"
    pred_p = ROOT / "out" / f"scene_{tile}.height.tif"
    if not (rgb_p.exists() and agl_p.exists() and pred_p.exists()):
        print(f"  (skipping error map: missing inputs for {tile})")
        return
    with rasterio.open(rgb_p) as s:
        rgb = s.read()[:3].transpose(1, 2, 0)
    with rasterio.open(agl_p) as s:
        agl = s.read(1).astype("float32")
    with rasterio.open(pred_p) as s:
        pred = s.read(1).astype("float32")

    HEIGHT = [[0.80, 0.87, 0.90], [0.55, 0.75, 0.82], [0.30, 0.58, 0.71],
              [0.16, 0.40, 0.56], [0.06, 0.22, 0.36]]
    ERR = [[0.13, 0.31, 0.55], [0.42, 0.60, 0.78], [0.92, 0.92, 0.90],
           [0.85, 0.50, 0.40], [0.63, 0.12, 0.14]]
    hi = float(np.nanpercentile(agl, 99.5))
    err = pred - agl
    e = float(np.nanpercentile(np.abs(err), 98))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, img, title in [
        (axes[0], rgb, "Satellite image"),
        (axes[1], _ramp(agl, 0, hi, HEIGHT, 0.5), f"LiDAR truth (0–{hi:.0f} m, √ scale)"),
        (axes[2], _ramp(pred, 0, hi, HEIGHT, 0.5), f"Our estimate (0–{hi:.0f} m, √ scale)"),
        (axes[3], _ramp(err, -e, e, ERR), f"Signed error (±{e:.0f} m)"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[3].set_xlabel("blue = we said too low   ·   red = too high", fontsize=8, color=DIM)
    save(fig, out / f"fig_error_map_{tile}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "out/eval_run02_ship/metrics.json"))
    # Inside the repo, not out/, so docs/evidence-pack.md is self-contained when
    # the source tree is shipped on its own.
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "figures"))
    ap.add_argument("--tile", default="OMA_288_042", help="tile for the error map")
    a = ap.parse_args()

    m = json.loads(Path(a.metrics).read_text(encoding="utf-8"))
    out = Path(a.out)
    print(f"figures from {a.metrics}")
    fig_by_height(m, out)
    fig_calibration(m, out)
    fig_terrain(m, out)
    fig_scatter(Path(a.metrics).with_name("per_building.npz"), out)
    fig_error_map(a.tile, out)
    print(f"\nwrote to {out}")


if __name__ == "__main__":
    main()
