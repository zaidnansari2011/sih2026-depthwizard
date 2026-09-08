# Probe 03 — can a guided filter sharpen our buildings?

**Question.** Our surfaces are smooth: buildings render as rounded mounds rather than the
flat-topped blocks an LOD1 product should show. Guided filtering (He, Sun & Tang, ECCV
2010) is the standard post-process in depth refinement — it smooths within regions but
stops at edges present in a guide image. Feeding it the satellite image should, in
principle, transfer roof boundaries into the height map. No retraining, seconds per tile.

**Prediction, written before the run.** Sharper to look at, but per-building RMSE
unchanged, because per-building scoring takes the *median* height inside each footprint
and medians are robust to edge blur. Falsified if per-building RMSE improves by more than
0.1 m.

**Falsification condition for the method itself.** If per-building RMSE gets *worse* at
every setting, guided filtering is the wrong tool and no amount of tuning saves it.

## Result — it is the wrong tool

Four val tiles (urban, sparse, forested, mixed), run02+TTA predictions, swept over radius
and epsilon. Mean across tiles:

| setting | per-pixel | building-px | **per-building** |
|---|---|---|---|
| unfiltered | 4.477 | 6.504 | **6.863** |
| guided r=4 | 4.496 | 6.532 | 6.872 (+0.009) |
| guided r=8 | 4.556 | 6.625 | 6.971 (+0.108) |
| guided r=16 | 4.760 | 6.928 | 7.222 (+0.359) |

Epsilon made almost no difference; radius made all of it. **Every setting is worse than
doing nothing, monotonically with radius.** The bias also drifts negative as the radius
grows — on the sparse tile, from −0.01 m to −1.26 m at r=16 — because the filter pulls
rooftops down toward the surrounding ground.

## Why, and what it rules out

A guided filter **preserves** edges that exist in the source; it cannot **create** edges
that were never predicted. Guided upsampling works when the input is low-resolution but
locally sharp and the guide supplies the missing detail. Ours is the opposite: full
resolution and intrinsically smooth. There is no sharp content to preserve, so the filter
does the only other thing it can — smooth further.

This rules out the whole family of post-hoc edge-transfer fixes. **The blur is structural,
not cosmetic**: with a direct-regression head, the loss-minimising output at an ambiguous
roof-edge pixel is the average of ground and roof, so a step in the world becomes a ramp in
the prediction. Nothing downstream can undo an average that was never a step.

## What that leaves

The fix has to be in the head, and the mechanism is known: a network that **picks a bin**
rather than averaging cannot hedge. That is HTC-DC Net's binned soft-argmax, already
implemented (`--bins`, head-tail cut).

We have run this once. run03 lost on the headline — whole-tile RMSE 6.950 against run02's
6.456 — but the result was not clean, and one number in it supports the sharpness story:
**run03's ground RMSE was 1.75 m against run02's 1.92 m**, and crop-wise scoring put run03
ahead (7.894 vs 7.980). Bins sharpened the surface and lost elsewhere.

Cost of this probe: about 20 minutes, no GPU. Cost of learning the same thing from a
training run: 3.5 hours. That is the process rule working.
