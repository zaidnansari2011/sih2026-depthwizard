# Probe 05b — does auto-zoom still hold at 0.6 m on the shipped model?

**Why now.** ISRO's FAQ (22 Sep, `docs/isro-faq.md`) settles the evaluation imagery:
*"Cartosat 2S imageries with resolution of 0.6m will be used for final evaluation."* The only
0.6 m measurement we have is probe 05's, and it is weak evidence for the model we ship:

- it is **run02**, not run07;
- it is **three hand-picked non-urban tiles**, averaged per tile rather than pooled;
- it was a manual `infer.py --zoom` sweep, **single pass**, not the shipped TTA + fusion.

So "0.6 m costs 3.6% after auto-zoom" is not yet a claim we can make about run07.

## Pre-registration — written 25 Sep 2026, before any run

**Method.** `tools/gsd_recheck.py` scores run07 on the **same 80 validation tiles**
`tools/evaluate.py` uses (seed 1337, region-disjoint val split), in-process, with the
**shipped configuration** (8-way TTA + zoom-2 Laplacian fusion, σ = 8). There are three
conditions per tile:

| | input | zoom (what `--auto-zoom` would pick) |
|---|---|---|
| **A** native | the 0.3 m tile as delivered | 1 |
| **B** untreated | Lanczos-downsampled to 0.6 m (512 px) | 1 |
| **C** auto-zoom | the same 0.6 m image | 2, with the fusion detail pass at 4 |

B and C predictions are bilinearly resampled back onto the 1024 px truth grid. Truth never
moves. Buildings are pooled across all 80 tiles, as in the headline metric.

**Sanity gate (must pass or the run is void).** Condition A reproduces the shipped
per-building RMSE of **3.464 m within ±0.01 m**. It is the same code path, so any larger gap
means the script is wrong, not the model.

**Primary prediction.** With auto-zoom, 0.6 m input costs **no more than 10 %** of native
pooled per-building RMSE: C ≤ 1.10 × A (≤ 3.81 m if A = 3.464). **Falsified if C > 1.10 × A.**

**Secondary prediction.** Auto-zoom helps: C < B on pooled per-building RMSE.
**Falsified if C ≥ B.**

**Also reported, not pre-judged:**
- per-building RMSE on buildings ≤ 20 m, where the tall-building tail cannot dominate;
- per-building bias;
- whole-tile RMSE;
- per-terrain whole-tile RMSE.

**What the deck will say, decided now.**
- If the primary holds: *"At ISRO's 0.6 m, measured on 80 tiles: +X % per-building error
  after auto-zoom"*, with X as measured.
- If it fails: the deck states the measured cost and the "within 3.6 %" line is removed
  everywhere it appears.

**Limitation stated in advance.** 0.6 m is *simulated* by downsampling 0.3 m WorldView-3.
Cartosat-2S differs in sensor, spectral response and look angle, so this measures the
resolution gap only, not the sensor gap.

## Result

*(filled in after the run; the section above is not edited)*
