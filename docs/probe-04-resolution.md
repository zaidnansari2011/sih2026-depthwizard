# Probe 04 — where the missing detail actually went

**The complaint.** Buildings render as rounded mounds rather than flat-topped blocks. Below
LOD1, in CityGML terms.

**Probe 03 established what it is not.** Two independent families of post-process —
guided filtering and morphological toggle contrast — both made every metric *worse*. The
reason is the useful part: a guided filter preserves edges that exist and ours do not
exist; toggle contrast does manufacture a step from a ramp, but places it at the ramp
midpoint, and the true roof edge is not there. **A sharp edge in the wrong place costs more
than a soft edge in roughly the right one.** Post-processing has no information about where
the true edge is, so the detail has to be recovered during inference.

## The cause is token resolution

The backbone has patch size 14. A 518 px window is 37x37 tokens, so **one token covers
14 px = 4.2 m** at DFC2019's 0.3 m GSD. A 12 m building is under three tokens across. The
structure is not being blurred away — it was never resolved.

## Measured: zoom 2 recovers the detail and loses accuracy

`infer.py --zoom 2` takes a 259 px ground window, upsamples it to 518, predicts, and
resamples back. A token then spans 2.1 m. Sharpness is the mean local height range
(3x3 dilate minus erode) over building pixels, with LiDAR setting the bar.

Four val tiles — urban, sparse, forested, mixed:

| | per-pixel | building-px | **per-building** | bias | **sharpness** | vs LiDAR |
|---|---|---|---|---|---|---|
| zoom 1 (shipping) | 4.692 | 6.853 | **7.243** | -3.23 | 0.565 | 48% |
| zoom 2 | 5.220 | 7.693 | 7.896 | -3.51 | 0.778 | 66% |
| fused sigma=2 | **4.683** | **6.848** | 7.246 | -3.22 | 0.590 | 50% |
| fused sigma=4 | 4.685 | 6.867 | 7.258 | -3.23 | 0.650 | 55% |
| fused sigma=8 | 4.722 | 6.944 | 7.315 | -3.31 | **0.729** | **62%** |
| LiDAR truth | | | | | 1.183 | 100% |

Fusion is a Laplacian blend: low frequencies from zoom 1, which carries the absolute
calibration, high frequencies from zoom 2, which carries the edges.

**Zoom 2 alone is not shippable** — it is sharper but worse on every accuracy metric,
because the head was fine-tuned at 0.3 m and now sees objects at twice their trained size.
Bias drifts from -3.23 to -3.51.

**Fusion at sigma=8 buys +29% sharpness for +0.072 m per-building RMSE — 1.0% relative.**

### A caution about single tiles

On JAX_203_010 alone, fusion at sigma=4 *beat* the baseline outright: per-building 1.458
against 1.500, with sharpness up 22%. That looked like a free win and it is not one. Across
four tiles the same setting costs +0.015 m. **n=1 is not a result**, and the four-tile mean
is itself skewed by the urban tile, which contains a 93.6 m building against our 31.5 m
ceiling.

## What this points at

The real fix is to **fine-tune at the higher resolution** rather than blend at inference:
train on 259 px crops upsampled to 518, so the head is calibrated for the scale at which
the detail exists. That gets sharpness *and* accuracy instead of trading one for the other.
It is a training run, so it waits behind run05.

Until then fusion is available, tunable, and costs no training. The setting is a judgement
about how much headline RMSE a visibly better surface is worth — and the viewer carries
half the marks.

**Whatever is shipped must be what is scored.** Rendering a fused surface in the viewer
while quoting zoom-1 numbers in the report would be quoting a metric from a product we are
not showing.
