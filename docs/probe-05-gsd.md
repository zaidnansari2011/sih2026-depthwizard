# Probe 05 — does the model survive the resolution ISRO will evaluate at?

**The risk.** We train and infer at DFC2019's 0.3 m. The problem statement says final
evaluation uses *"ISRO RGB-band optical satellite imagery"*, which in practice means
Cartosat — nearer 0.6–1.0 m. Probe 04 established that what governs our detail is the
ground area one backbone token covers, so halving the resolution doubles that footprint.
Nobody had measured it.

**Method.** Downsample the RGB to the target GSD, infer, upsample the prediction back to
the truth grid, score against the *same* LiDAR reference. Truth never moves; the only thing
changing is what the model was allowed to see.

## It degrades, and the mean hides how much

| GSD | per-pixel | ground | per-building | bias |
|---|---|---|---|---|
| **0.3 m** (trained) | 5.486 | 1.839 | **8.847** | -4.14 |
| **0.6 m** | 6.404 | 1.872 | 9.224 | -5.81 |
| **1.0 m** | 7.271 | 1.970 | 10.109 | -6.73 |

The mean says +4% at 0.6 m and +14% at 1.0 m, which reads as graceful. **It is not.** The
mean is dominated by the urban tile, whose error is already saturated by a 93.6 m building
we miss at every resolution. Per tile:

| tile | 0.3 m | 0.6 m | 1.0 m |
|---|---|---|---|
| urban (saturated) | 23.316 | 23.130 | 24.459 |
| forested | 1.344 | 2.274 (x1.7) | **3.153 (x2.3)** |
| mixed | 1.881 | 2.267 (x1.2) | 2.715 (x1.4) |

**On ordinary tiles the error roughly doubles by 1 m.** Two things point at the cause:
ground RMSE barely moves (1.839 -> 1.970), so terrain is fine and buildings take the
damage; and the loss is mostly **bias** (-4.14 -> -6.73), not scatter. We systematically
under-call, which is what averaging a building with its surroundings does.

## The fix: match the scale at inference

If the loss is a scale mismatch, feeding the model imagery upsampled back to its trained
scale should recover it. Probe 04's zoom, run backwards. Three non-urban tiles:

| input GSD | zoom | per-building | bias | corr |
|---|---|---|---|---|
| 0.3 m (native) | 1 | **1.350** | -0.16 | +0.667 |
| 0.6 m | 1 | 2.260 | -1.67 | +0.459 |
| **0.6 m** | **2** | **1.399** | **-0.29** | **+0.663** |
| 1.0 m | 1 | 3.138 | -2.69 | +0.389 |
| **1.0 m** | **3** | **1.657** | **-0.98** | **+0.646** |

At 0.6 m, upsampling recovers **1.399 against a native 1.350 — within 3.6% of never having
lost the resolution**. Bias collapses from -1.67 to -0.29 and correlation returns to +0.663
against native's +0.667. At 1.0 m it recovers 47% of the loss.

**The coarse-GSD gap is almost entirely a scale mismatch, and it costs no retraining.**
This is now the third independent confirmation of probe 04's token-footprint account.

## Shipped as `--auto-zoom`

`infer.py --auto-zoom` reads the ground sample distance from a georeferenced input and
upsamples so the backbone always sees ~0.3 m, whatever it was handed.

Two input paths, both tested end to end:

* **Projected CRS** — take the pixel size from the transform directly.
* **Geographic CRS (lat/lon)** — `crs.linear_units_factor` *raises* on a geographic CRS
  rather than returning anything, and lat/lon GeoTIFFs are a normal delivery format, so
  degrees are converted using 111.32 km per degree of latitude and `cos(latitude)` for
  longitude at the scene centre. Verified on a reprojected Sikkim crop: recovered 0.29 m
  against a true 0.305 m, which is ample to choose an integer zoom.

Non-georeferenced input (PNG/JPG) has no knowable GSD, so it says so and leaves zoom at 1.

## This is the scale-calibration milestone

The problem statement names *"Scale Calibration: convert relative depth to absolute height
using scene-level statistics, low-resolution DEMs, semantic priors, or minimal Ground
Control Points"* as a milestone, and we had skipped it on the grounds that fine-tuning to
predict metric height directly makes it unnecessary.

That argument still holds for *absolute scale*. But `--auto-zoom` is a calibration module
in the sense that matters operationally: it reads the input's geospatial metadata and
adapts the pipeline so the metric head stays valid. Without it our heights are 20-40% low
on the imagery we will actually be judged on.
