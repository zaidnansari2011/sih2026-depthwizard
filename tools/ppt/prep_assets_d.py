"""Build every image deck D places, from sources that cannot go stale.

Deck C's two charts were typed-in numbers from run02 (tools/ppt/make_figures.py), so after
run07 shipped they disagreed with the evidence pack: urban 13.86 m on the slide against
13.084 m measured. Here both charts are drawn straight from the shipped metrics file, and
every number a slide quotes about them is printed so it can be checked against the text.

Crops:
  omaha_input      the satellite panel of the evidence-pack error map -- the *same* tile
                   (OMA_288_042) the viewer's "Urban - Omaha" scene is baked from, so the
                   before/after pair on slide 2 really is one scene in and out.
  omaha_height_d,  the viewer captures, re-cropped to deck D's strip aspect.
  omaha_sigma_d
  sikkim_hilly_d   the supplied Sikkim render, top-anchored like deck C's.
  sikkim_flood_d   the inundation tool running on the Indian valley, HUD readout included.

    python tools/ppt/capture_viewer.py sikkim_flood     # only if the flood still is missing
    python tools/ppt/prep_assets_d.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from crop_shots import fit_supplied

ROOT = Path("D:/sih2026/depthwizard")
METRICS = Path("D:/sih2026/out/eval_run07_ship/metrics.json")
SHOTS = ROOT / "tools/ppt/shots"
OUT = ROOT / "tools/ppt/figures/d"

STRIP = 3.30 / 1.74        # slide 2's after-panels, width over height in inches
FLOOD = 6.10 / 2.26        # slide 5's use-case frame
E_TERRAIN = (3.36, 2.02)   # deck E slide 7's two chart slots, in inches, placed 1:1
E_HEIGHT = (3.36, 1.88)

INK, SLATE, ACCENT, STEEL, RULE = "#14181D", "#5B6670", "#C1440E", "#2E6E92", "#D3D8DD"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": SLATE, "xtick.color": SLATE,
    "ytick.color": SLATE, "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# --------------------------------------------------------------------------- charts
def terrain(m: dict, size=(4.5, 2.15), fs=8.5, name="terrain.png") -> None:
    """`size` is the figure in inches. Drawn at the size it is placed on the slide, `fs`
    prints at its nominal point size; decks B-D shrink the default by 25-50 %."""
    order = ["sparse", "mixed", "forested", "urban"]
    rmse = [m["per_terrain"][k]["rmse"] for k in order]
    corr = [m["per_terrain"][k]["corr"] for k in order]
    fig, ax = plt.subplots(figsize=size, dpi=220)
    cols = [STEEL] * 3 + [ACCENT]
    names = [k.capitalize() for k in order]
    bars = ax.barh(names[::-1], rmse[::-1], color=cols[::-1], height=0.62)
    for b, v in zip(bars, rmse[::-1]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.2f} m", va="center",
                fontsize=fs + 0.5, color=INK, fontweight="bold")
    ax.set_xlim(0, math.ceil(max(rmse) / 2) * 2 + 2)
    ax.set_xlabel("Whole-tile RMSE (m), 80 held-out tiles", fontsize=fs - 0.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=fs - 0.5)
    ax.tick_params(axis="y", length=0, labelsize=fs, labelcolor=INK)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / name)
    plt.close(fig)
    if name == "terrain.png":
        for k, r, c in zip(order, rmse, corr):
            print(f"    terrain {k:<9} rmse {r:6.3f}  corr {c:+.3f}")


def error_by_height(m: dict, size=(6.4, 2.55), fs=8.5, name="error_by_height.png",
                    bare_y=False) -> None:
    """`bare_y` drops the y axis: every bar carries its value, and at slide size the axis
    only competes with those labels for room."""
    bands = m["building_wise_by_height"]
    labels = ["0–3 m", "3–6 m", "6–10 m", "10–20 m", "> 20 m"]
    assert len(bands) == len(labels), f"expected 5 height bands, got {len(bands)}"
    rmse = [b["rmse"] for b in bands]
    n = [b["n_buildings"] for b in bands]

    fig, ax = plt.subplots(figsize=size, dpi=220)
    bars = ax.bar(labels, rmse, color=[STEEL] * 4 + [ACCENT], width=0.66)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f} m", ha="center",
                fontsize=fs + (0.5 if bare_y else 0), color=INK, fontweight="bold")
    ax.set_ylim(0, math.ceil(max(rmse) / 5) * 5 + (1 if bare_y else 3))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if bare_y:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    else:
        ax.set_ylabel("Per-building RMSE (m)", fontsize=fs)
        ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7)
    ax.tick_params(length=0, labelsize=fs - 0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{b}\nn={c:,}" for b, c in zip(labels, n)], fontsize=fs,
                       color=INK)
    if bare_y:
        ax.set_xlabel("Per-building RMSE by true height", fontsize=fs - 0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / name)
    plt.close(fig)
    if name != "error_by_height.png":
        return

    # The sentences beside this chart, recomputed rather than carried over.
    sq = [k * r * r for k, r in zip(n, rmse)]
    total = sum(n)
    under10 = math.sqrt(sum(sq[:3]) / sum(n[:3]))
    print(f"    > 20 m: {n[-1]} buildings = {100 * n[-1] / total:.1f}% of {total:,}, "
          f"carrying {100 * sq[-1] / sum(sq):.1f}% of squared error")
    print(f"    under 10 m: {sum(n[:3]):,} buildings = {100 * sum(n[:3]) / total:.1f}% "
          f"of stock, RMSE {under10:.3f} m")
    print(f"    > 20 m bias {bands[-1]['bias']:+.2f} m")


# ---------------------------------------------------------------------------- crops
def crop_to(src: Path, box, aspect: float, dest: Path) -> None:
    """Take `box` and grow or shrink it vertically about its centre to `aspect`."""
    im = Image.open(src).convert("RGB")
    l, t, r, b = box
    h = (r - l) / aspect
    cy = (t + b) / 2
    t, b = max(0, int(cy - h / 2)), min(im.height, int(cy + h / 2))
    out = im.crop((l, t, r, b))
    out.save(dest, optimize=True)
    print(f"    {dest.name:<22} {out.width}x{out.height}  ({out.width / out.height:.2f}:1)")


def omaha_input() -> None:
    """The first panel of the error map, found by its frame rather than by a guessed box."""
    import numpy as np
    src = ROOT / "docs/figures/fig_error_map_OMA_288_042.png"
    a = np.asarray(Image.open(src).convert("RGB")).astype(int)
    # Columns and rows where the panel's image content sits: not near-white background.
    dark = (a.min(axis=2) < 225)
    left_third = dark[:, : a.shape[1] // 4]
    rows = np.where(left_third.mean(axis=1) > 0.5)[0]
    cols = np.where(left_third[rows.min():rows.max()].mean(axis=0) > 0.5)[0]
    pad = 4                                             # stay inside the 1 px frame
    box = (cols.min() + pad, rows.min() + pad, cols.max() - pad, rows.max() - pad)
    out = Image.open(src).convert("RGB").crop(box)
    side = min(out.size)
    out = out.crop((0, 0, side, side))
    dest = OUT / "omaha_input.png"
    out.save(dest, optimize=True)
    print(f"    {dest.name:<22} {out.width}x{out.height}  from box {box}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text())
    assert m["checkpoint"].endswith("run07/best.pt") and m["tta"], (
        f"{METRICS} is not the shipped configuration: {m['checkpoint']}, tta={m['tta']}")
    print(f"  metrics  {METRICS}  ({m['n_tiles']} tiles, "
          f"{m['building_wise']['n_buildings']:,} buildings)")
    print(f"    per-building RMSE {m['building_wise']['rmse']:.3f} m, "
          f"ECE {m['ece']:.3f}, sigma rank r {m['sigma_rank_corr']:+.3f}")
    terrain(m)
    error_by_height(m)
    # Deck E's slide 7 places these at exactly E_TERRAIN / E_HEIGHT, so 9.5 pt is 9.5 pt.
    terrain(m, size=E_TERRAIN, fs=9.5, name="terrain_e.png")
    error_by_height(m, size=E_HEIGHT, fs=9.5, name="error_by_height_e.png", bare_y=True)

    print("  crops")
    omaha_input()
    # Both HUD panels excluded: control panel ends x 585, stats panel starts x 2688.
    viewport = (600, 330, 2680, 1780)
    crop_to(SHOTS / "urban_height.png", viewport, STRIP, OUT / "omaha_height_d.png")
    crop_to(SHOTS / "urban_sigma.png", viewport, STRIP, OUT / "omaha_sigma_d.png")
    top, _ = fit_supplied(ROOT / "docs/sikkimhilly.png", OUT / "sikkim_hilly_d.png",
                          "top", aspect=STRIP)
    im = Image.open(OUT / "sikkim_hilly_d.png")
    print(f"    {'sikkim_hilly_d.png':<22} {im.width}x{im.height}  (supplied, top={top})")
    # The viewport only. The HUD's readout is the point of this still, but at slide size
    # it rendered as grey specks, so make_deck_d redraws those three captured numbers
    # (FLOOD_READOUT there) as a legible strip over the image instead.
    crop_to(SHOTS / "sikkim_flood.png", (600, 640, 2660, 1520), FLOOD,
            OUT / "sikkim_flood_d.png")


if __name__ == "__main__":
    main()
