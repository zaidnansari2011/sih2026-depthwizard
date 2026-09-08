# Probe 06 — we compress every height, and it cannot be fixed at inference

**Found by looking, not by measuring.** Zaid put our surface next to the LiDAR reference in
the viewer's drag-to-compare and said the heights looked lacking. He was right, and the
metric table I had been quoting was hiding it.

## The compression

Over all 3,090 validation buildings:

    ours = 0.473 * truth + 2.66 m

| true height | median ours / median truth |
|---|---|
| 0–3 m | **1.20** — over-called |
| 3–6 m | **1.14** — over-called |
| 6–10 m | 0.88 |
| 10–20 m | 0.99 |
| 20–40 m | **0.65** |
| 40 m+ | **0.52** |

Short buildings come out too tall, tall ones at half height: everything is pulled toward
the ~4 m median of the training distribution. That is regression to the mean. Under
uncertainty the loss-minimising prediction is the conditional average, and with a
long-tailed target the safe answer is always "about four metres".

**The per-height RMSE table concealed this.** Quoting "94% of buildings are under 10 m and
we are at 1.4 m there" is true in absolute metres and hides a systematic 14–20% over-call.
An error of 0.6 m on a 4 m building is small in metres and 15% in ratio; only one of those
appears in an RMSE column.

## Can it be inverted at inference?

Fit the correction on TRAIN tiles, measure on VAL — fitting on validation would tune the
number we report. 24 training tiles, 1,007 buildings.

| setting | RMSE | MAE | bias | corr |
|---|---|---|---|---|
| uncorrected | 3.667 | **1.498** | −0.24 | +0.778 |
| de-compressed, full | **3.543** | 1.578 | −0.25 | +0.778 |

RMSE improves 3.4%, MAE worsens 5.3%, **correlation does not move at all** — as it cannot,
since correlation is invariant under a linear map. That is the tell: this adds no
information, it only redistributes error from the few large mistakes RMSE punishes onto the
many small ones MAE counts. ISRO names all three metrics.

## Why it under-corrects, and why nothing better is available

The train fit gives slope 0.8055; validation needs 0.473. **That gap is population, not
behaviour** — a confound worth checking before claiming a generalisation failure, which is
what it looks like at first glance. On matched height ranges the two agree:

| range | TRAIN slope | VAL slope |
|---|---|---|
| 0–20 m | 0.821 (n=1000) | 0.786 (n=3030) |
| 0–10 m | 0.669 (n=959) | 0.646 (n=2914) |
| all | 0.806 (n=1,007) | 0.473 (n=3,090) |

Training tiles hold **7** buildings above 20 m, max 38.4 m. Validation holds **60**, max
85.9 m. The compression behaves identically on both; it simply worsens with height, and the
training sample barely reaches the heights where it becomes severe.

So a correction fitted on training data applies a 1.24× stretch where validation's tall
buildings need about 2.1×. **The data needed to calibrate the correction is the same data
we do not have** — the identical wall probe 01 hit. Fitting it on validation instead would
mean tuning on the reported number.

## Decision: not shipped

- The gain is a bias/variance retrade, not information: correlation is unchanged.
- It improves RMSE and worsens MAE, and the brief names both.
- It amplifies noise by the same factor it amplifies signal.
- Applied to the whole height map it would scale **ground** too, which is currently our best
  result — 1.906 m RMSE at near-zero bias — and is not compressed, because it sits at zero.
  Any real deployment would need a monotone quantile map that leaves the low end alone
  rather than a straight line.

## What this actually points at

Not capacity. A larger backbone trained with the same loss on the same long-tailed target
hedges toward the median just as hard, which is worth remembering when run05 lands.

The mechanism is the loss, and the fix is a head that **cannot** average two plausible
answers into a wrong middle one. That is precisely what binned-ordinal prediction with
soft-argmax does, and it is what HTC-DC Net (2023), Chen et al. (2025) and TSONet (2026)
independently converge on.

We built it once, as run03, and dropped it because whole-tile RMSE lost 6.950 to 6.456. That
judgement now looks too quick: run03's **ground RMSE improved, 1.75 against 1.92**, and
crop-wise scoring put it ahead. Sharper, less-averaged predictions are exactly the signature
this probe predicts an ordinal head would produce.

**Revisiting the binned head with the compression measured is the best-founded next
experiment we have** — better founded than a backbone change, because it targets a mechanism
we have now measured rather than a capacity we have only assumed is short.
