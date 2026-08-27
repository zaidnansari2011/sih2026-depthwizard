# DepthWizard — Plan of Record

**SIH26175** · ISRO / Space Applications Centre · Software
**Idea submission closes 20 September 2026** · cap 500 submissions per problem statement
Repo: `zaidnansari2011/sih2026-depthwizard` (private) · Workspace: `D:\sih2026`

> This is the living reference. If a decision changes, change it *here* first.

---

## 1. What we are building

An end-to-end pipeline that takes **one** optical image and produces a **Digital Surface Model**, then projects that image onto a navigable 3D terrain mesh.

Two input paths are mandatory:

| Input | Output | Calibration |
|---|---|---|
| Plain PNG / JPG, no spatial metadata | **Relative** DSM | none — scale-free |
| Georeferenced GeoTIFF | **Absolute** DSM, real metric heights | coarse DEM (SRTM 30 m) or ground control points |

Deliverable is a deployable suite: an elevation module emitting a DSM in a standard geospatial format, plus an interactive viewer where a user uploads imagery, sees the reconstructed terrain, and **validates estimated heights against reference data**.

## 2. Scoring — the thing that drives every decision

| Weight | Criterion | What it actually means |
|---|---|---|
| **50%** | DSM accuracy & validation | RMSE, MAE, correlation vs LiDAR/reference. **Stability required across urban, sparse, hilly, forested.** |
| **50%** | Rendering quality & UX | Projection accuracy, visual fidelity, navigability, interface intuitiveness, stability, successful standalone deployment |

**Half the marks are the front end.** A pure-CV team pours 36 hours into the model and demos a clunky viewer, forfeiting most of the second half. A pure-web team cannot produce a defensible DSM at all. We can contest both halves, and they parallelise from hour one.

Corollary: *never* let the viewer become the last night's work.

## 3. Why this problem statement

226 statements, 172 software, roughly one winner each. Teams lose at selection, before any code. The crowd flows to whatever looks easiest — attendance apps, chatbots, dashboards — where hundreds of near-identical builds are judged on polish alone.

"Estimate height from a single image" *sounds* like original research, so most teams self-select out. It isn't research; see §5. That gap between perceived and actual difficulty is the entire opportunity.

**Honest caveat:** thin competition is not weak competition. Teams that do pick this will be strong CV teams. We must genuinely execute — we just aren't fighting 300 lookalikes.

**Open check:** every PS currently reads 0/500 because the window just opened. Crowding is inferred from sponsor patterns, not measured. **Re-check live counts around 10 September** and be willing to switch to SIH26143 (NTRO, oil-spill detection with AIS attribution) if this one turns out swamped.

## 4. Compute & environment

| Resource | Role |
|---|---|
| RTX 3060 12 GB (local) | dev loop, debugging, **all preprocessing**, viewer, demo machine |
| Kaggle — 30 GPU-h/week, 12 h sessions | experiment farm: benchmark sweeps, long unattended runs |
| Azure $100 credit | **hosting the standalone demo**, not training |

- 3060 beats a Kaggle P100 (Pascal, no tensor cores) and roughly matches a single T4. Only 2×T4 with DDP is faster, and only ~1.6×. Free cloud buys **parallelism**, not speed.
- **T4/P100 have no bf16.** Training code autodetects and falls back to fp16 + GradScaler.
- Preprocess locally (CPU/disk-bound, burns zero quota) → upload shards **once** as a Kaggle Dataset → every session mounts `/kaggle/input` instantly.
- Code lives in git and is cloned into notebooks. **Never write code inside a notebook.**
- Checkpoint every epoch — 12 h session cuts are routine, not exceptional.
- Verify Azure GPU quota in week 1: student subscriptions default N-series quota to 0 and cap vCPUs. If blocked, nothing is lost — training was always local.

### Measured throughput and thermals, 26 Aug 2026

| Batch | ms/step | crops/s | Peak VRAM |
|---|---|---|---|
| 4 | 174 | 23.0 | 2.21 GB |
| **8** | **327** | **24.5** | **6.01 GB** |
| 12 | 469 | 25.6 | 7.57 GB |

Throughput plateaus around **25 crops/s**, so batch 8 — the Depth Any Canopy setting — is
also the efficient one, and 12 GB is not the binding constraint. **Heat is.**

**The card runs at its thermal limit long before it runs out of memory.** Unregulated, the
3060 reached **91 °C in under ten minutes** and down-clocked itself from 1875 to
**1492 MHz** — so the "full speed" run was not faster, just hotter. `nvidia-smi -pl` is the
clean fix but needs an administrator shell (`Insufficient Permissions` otherwise), so
`train.py` carries a software `ThermalGovernor`: sample every N steps, pause proportional
to overshoot, hard-stop above a ceiling until it cools.

#### Solved 27 Aug 2026 — and the software governor was always a workaround

Duty-cycling in software was the wrong tree. MSI Afterburner has a **"Prioritize" switch
between power ⚡ and temperature 🌡**, and it defaults to *power* — which makes the Temp
Limit **advisory**. The card draws to its power limit and lets temperature land wherever
it lands. That is why a Temp Limit of 77 °C still produced 91 °C readings, and why raising
the fan from 62% to 85% bought only ~3 °C.

Setting **Prioritize = temperature** makes the limit binding in hardware:

| | Before | After |
|---|---|---|
| Temperature | 91 °C | **mean 80.1 °C, peak 85 °C** |
| Clocks | throttled to 1492 MHz | **1777–1867 MHz** |
| Throughput | 53 steps/min (governed) | **143 steps/min** |
| 12-epoch run | 3.6 h | **~1.6 h** |

Cooler *and* 2.7× faster, because the card down-clocks smoothly instead of being stopped
and started. Working config: **fan 85% (auto off), Temp Limit 80, Power Limit 85%,
Prioritize = temperature, ⇄ link off** (that toggle ties the two limits together; turn it
off to set them independently). The software `ThermalGovernor` now fires **zero** times and
remains only as a backstop at `83 / 87`.

> ⚠️ **Never change GPU settings while a CUDA job is running.** Applying Afterburner
> settings resets the display driver and invalidates live CUDA contexts. Doing it mid-run
> killed a training process with `torch.AcceleratorError: CUDA error: unknown error` on a
> `.to(device)` call, and silently reverted the power limit to 170 W. Pause first.

`tools/vitals.py` samples GPU, RAM and disk on a timer, logs to a CSV that outlives the
session, and prints only on breach. It exists because both hardware failures in this
project — ten minutes at 91 °C, and an OOM kill with no traceback — were sitting in
numbers nobody was reading.

> **User-side action worth taking:** MSI Afterburner is already installed. A more
> aggressive fan curve would let training run *both* faster and cooler than software
> pausing can, and running one shell as administrator would allow a proper 120 W power
> limit. Either beats duty-cycling.

## 5. Method

Three stages. Only the middle is genuinely novel work.

### Stage 01 — Relative depth · *CV pair*

Depth Anything V2 (DINOv2 encoder + DPT decoder) producing a scale-free depth map.

> **This is NOT zero-shot.** The team brief's claim that "the weights already exist, we train nothing" is wrong, and it is the plan's one load-bearing wrong assumption. DA-V2 was trained on natural photographs with a ground plane and a vanishing point; a nadir satellite tile has neither.

#### Measured, 26 Aug 2026 — and the earlier wording here was too harsh

This section used to claim that run zero-shot, DA-V2 "correlates with rooftop albedo, not height." That was an assertion, so it got tested: `tools/zero_shot_baseline.py`, DA-V2-Small over **80 held-out test tiles drawn from 16 regions that appear in no training split**.

| | RMSE | MAE | r | bias | Deployable? |
|---|---|---|---|---|---|
| Raw output vs metres | 5.50 m | 3.10 m | 0.43 | −1.56 m | units are arbitrary — read *r*, not RMSE |
| **Global affine** — one scale+shift fitted on *train* tiles | **4.68 m** | 3.32 m | 0.43 | +0.70 m | **yes — deployable, and the bar on the TEST split** |
| Oracle affine — refitted per test tile from its own truth | 3.94 m | 2.58 m | 0.63 | 0.00 m | no — needs the answer to compute the answer |

> **These are TEST-split numbers, and the test split is materially easier than val.**
> The same deployable global-affine baseline reads **9.308 m on val**. Development
> decisions use the val figure; test is reserved for one final reported number.
> Both tools now default to `--split val`. See
> [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

**Verdict: the pretrained features do see height at nadir.** r ≈ 0.43 is not albedo, it is signal. What is broken is *calibration*, not perception — and the failure has a specific, diagnosable shape:

| Class | RMSE | Bias | |
|---|---|---|---|
| ground | 3.62 m | **+2.53 m** | flat ground pushed *up* |
| building | 7.60 m | **−3.30 m** | rooftops pulled *down* |
| vegetation | 5.93 m | −3.57 m | |
| water | 1.80 m | +0.60 m | |

The model **compresses dynamic range**: it lifts the ground and flattens the structures. That is what a relative-depth prior does when it has no absolute reference, and it is what fine-tuning on metric labels exists to fix. Per terrain: mixed 4.31 m, sparse 5.74 m (bias **+5.16 m**), urban 6.17 m. Per-tile RMSE spans 2.11–8.07 m, so cross-terrain stability is a measured problem, not a hypothetical one (§7.5).

**The fix is smaller than feared, but the need for it is confirmed.** Fine-tuning must beat the baseline **on the same split** -- 9.31 m on val, not the 4.68 m test figure above. Landing above it would mean we did worse than a two-parameter linear correction, and §6.1 would have nothing to stand on.

*The earlier "overestimates tree height" concern did not reproduce: vegetation bias is −3.57 m, an under-estimate. Range compression dominates it.*

**Fix — the [Depth Any Canopy](https://github.com/DarthReca/depth-any-canopy) recipe**, which did exactly this adaptation for canopy height and beat prior SOTA:

| Setting | Value |
|---|---|
| Loss | MSE on min-max normalised height (we replace this — see §6.1) |
| Crops | 256×256 |
| Optimiser | AdamW, lr 5e-6 |
| Schedule | 5% warmup, linear decay, 3 epochs |
| Batch | 8 |
| Cost | 1.5–2.6 h on an A6000 → **~5–9 h on our 3060** |

Backbones to benchmark: DA-V2 **Small** (Apache-2.0), **Base** / **Large** (CC-BY-NC-4.0), **Depth Anything 3**, and Marigold / Depth Pro as alternates.

*Keep a Small result current* — it is the only clean-licence fallback if ISRO's spec demands redistributable code.

### Stage 02 — Metric calibration · *CV pair + data*

Fit the relative map to absolute metres by regressing against coarse SRTM 30 m elevation. This is the real problem, and it is regression, not research.

> **Metric anchor is SRTM 30 m or Copernicus GLO-30 — not CartoDEM.** CartoDEM 30 m is free only to Indian Government Entities; as students we are an NGE and it is a priced product. SRTM is genuinely free and is named in the problem statement anyway.

### Stage 03 — 3D flythrough · *Full-stack pair*

Displace a mesh by the heightmap, texture with the original image, fly a camera through it in the browser. Three.js (Unity and Babylon.js also accepted). **Worth 50% of the marks.**

### Fine-tuned results, 27 Aug 2026

Three runs. All val figures are whole-tile `--split val`, the same 80 tiles for every
row; see [docs/evaluation-protocol.md](docs/evaluation-protocol.md) for why crop-wise and
whole-tile numbers differ by ~1.5 m and must not be mixed.

| Run | Head | Objective | val RMSE | bldg RMSE | bldg bias | ECE | sigma rank rho |
|---|---|---|---|---|---|---|---|
| baseline | zero-shot + global affine | - | 9.308 | 22.47 | -9.64 | - | - |
| baseline | zero-shot + oracle affine *(not deployable)* | - | 7.285 | 17.33 | -2.64 | - | - |
| run01 | direct regression | NLL + grad | 6.811 | 16.72 | -3.23 | 0.163 | +0.562 |
| run02 | direct regression | NLL + grad, beta=0.5 | **6.454** | 16.37 | -4.70 | 0.083 | +0.836 |
| run03 | binned soft-argmax, N=128 | + Chamfer + distribution CE | *running* | | | | |

**The differentiator is real:** run02 beats the deployable baseline by 30.7%, and beats
the *oracle* affine, which is allowed to fit scale and shift from each tile's own truth.
The uncertainty head works - ECE 0.083 and sigma/error rank correlation +0.836.

**The beta-NLL ablation is closed, negative.** run02 ran 12 full epochs against run01's 5
and moved building RMSE 21.63 -> 21.17 crop-wise, with the bias still at -6.42 m.
Reweighting a regression loss does not fix a long-tailed output space, which is what the
literature said would happen (docs/literature.md section 4).

**The binned head still deploys.** Checked before trusting it: ONNX export of a binned
checkpoint agrees with PyTorch to 0.0014 cm on height and 0.0006 cm on sigma against a
5 cm tolerance, at 101.0 MB versus 100.7 for the regression head, 1125 ms per tile on
CPU. Soft-argmax, cumsum bin edges and the pooled width predictor all survive the
export. Still fixed at 518x518, which is what sliding-window inference uses anyway.
Section 6.4 is not at risk from the architecture change.

**Buildings remain the entire problem:** 12.8% of pixels carrying 91.8% of squared error.
run03 tests the actual fix - a distribution over adaptive height bins with soft-argmax,
supervised in shape as well as mean, because soft-argmax alone averages across bimodal
roof edges (docs/literature.md section 5).

**View angle is not what limits us.** DFC2019 images every AOI from many satellite
positions and the Track 3 metadata carries the geometry, so the domain-gap question in
section 7.1 is measurable today without Bhoonidhi. Joining per-tile error to
`meanOffNadirViewAngle` over 80 val tiles, 4.2 to 29.2 degrees:

| Measure | Pearson r | slope | 95% CI | p |
|---|---|---|---|---|
| Naive across tiles *(terrain-confounded)* | +0.000 | +0.000 m/deg | - | 0.998 |
| **Within-region** *(terrain controlled)* | +0.074 | +0.009 m/deg | [-0.021, +0.038] | 0.555 |

Every AOI appears at several angles, so subtracting each region's own mean removes the
terrain confound entirely and leaves only "this same ground, viewed more obliquely". The
95% interval bounds the cost at 0.038 m per degree, i.e. **at most ~1 m of RMSE across the
whole 25-degree span** against a 6.45 m headline. Sun elevation tells the same story and
shows why the control matters: naive r +0.212 collapses to -0.033 within-region, so the
apparent effect was which AOIs happened to be imaged at low sun.

This bounds sensitivity to *view geometry only*. A different sensor, GSD and radiometry is
the Cartosat gap proper, and this says nothing about it. `tools/view_angle.py`,
`out/view_angle_run02/`.

## 6. Our four differentiators

The research for this task exists — IM2HEIGHT, TSE-Net, THE Benchmark, a decade of monocular height estimation. **ISRO knows.** What does not exist is a *product*: those are checkpoints and eval scripts producing a number on a benchmark, not something an analyst can open and use. The 50/50 split is ISRO saying so out loud.

So uniqueness is **not algorithmic**. Attempting novel architecture in four weeks yields a half-finished novel thing instead of a complete solid one plus a sharp differentiator.

### 6.1 Per-pixel uncertainty — *the strongest, do this first*

Single-view height is mathematically ill-posed: no unique 3D solution exists for a 2D image. Every other team ships one DSM and one RMSE, implying certainty they don't have. We ship a DSM **plus a calibrated confidence map**.

- **Implementation:** variance head + Gaussian NLL loss instead of MSE. ~20 lines.
- **Must be in the architecture from the first training run**, not bolted on in week 3.
- **Why it wins:** a jury of remote-sensing scientists works in error budgets. Nobody operationalises a product without knowing where it fails. This is their dialect.
- **Scores in both halves:** shade the flythrough by confidence and judges *see* the model's doubt over water, shadow, dense canopy.
- Also defuses the "don't oversell accuracy" risk in §7.4.

**The error bar is honest only inside a window, and we now know which one.** The viewer
quotes `dh +/- hypot(sigma_a, sigma_b)`, but every calibration number we had was per pixel,
while the thing a user reads is a difference. Measured on run02 over 20 val tiles, with
bootstrap intervals over tiles:

| separation | error corr | +/-1 sigma coverage | 95% CI | median abs z |
|---|---|---|---|---|
| *Gaussian expectation* | 0.000 | 68.3% | - | 0.674 |
| single pixel | - | 64.5% | - | 0.62 |
| 1-3 m | +0.944 | 88.4% | [86.8, 90.3] | 0.19 |
| **3-15 m** | +0.802 | **71.4%** | **[67.9, 74.8]** | 0.47 |
| 15-60 m | +0.312 | 56.5% | [51.6, 61.3] | 0.78 |
| 60-300 m | +0.051 | 51.7% | [46.1, 57.8] | 0.94 |

68.3% sits inside the interval at 3-15 m and outside it everywhere else. Below that the
two errors are nearly the same error (corr +0.94), they cancel in the difference, and the
quoted bar is far too wide. Above it independence arrives but coverage keeps falling, so
the bar is too narrow -- the dangerous direction. Tails are heavy throughout: +/-3 sigma
covers 84.8% on single pixels, not 99.7%, so a 3-sigma bound from this model must never be
presented as one.

Two consequences. Measuring a single roof against adjacent ground is the *most* reliable
use of the tool and the bar there is conservative. And the bar wants an empirical
separation-dependent scale factor, roughly 0.28x at 1 m rising to 1.39x at 100 m, refit per
checkpoint on val. `tools/pair_calibration.py`, `out/pair_calibration_run02/`.

### 6.2 Shadow as a physical prior — *the novel-ish one*

`height = shadow_length × tan(sun_elevation)`. This is how photointerpreters measured building heights before computers. Sun azimuth and elevation ship in satellite metadata.

The clever version is not an input channel — it is a **self-supervised signal on unlabelled Cartosat data.** We cannot fine-tune on Indian imagery because we have no height labels there; shadow geometry yields free physics-derived pseudo-labels on exactly the imagery we otherwise cannot supervise. **This turns our single biggest risk (§7.1) into our best story.**

- **Gate: PASSED (26 Aug 2026).** `Track 3 / Metadata` holds 67 WorldView-3 `.IMD` files with full solar and viewing geometry: `meanSunAz`, `meanSunEl`, `meanSatAz`, `meanSatEl`, `meanOffNadirViewAngle`. Parse with `tools/parse_imd.py`.

**Sun elevation spans 23.4°–74.5°**, and that spread is the substance, not the mere presence of the field. A 20 m building casts 46 m of shadow at 23.4° and 5.6 m at 74.5°. So shadow reliability is scene-dependent and *we know the scene's geometry*, which means pseudo-label confidence can be weighted by solar elevation instead of trusted uniformly — feeding straight into the uncertainty head (§6.1).

That gives a real method rather than a trick:

1. Estimate heights from shadows on DFC2019, where ground-truth AGL exists.
2. Measure how shadow-derived error scales with sun elevation → a calibrated confidence model.
3. Apply to unlabelled Cartosat with that confidence attached.

**Bonus:** `meanOffNadirViewAngle` spans 4.8°–28.9°, so we can measure accuracy degradation vs look angle — an empirical handle on the domain gap (§7.1) that predicts Cartosat behaviour before we touch Cartosat.

**Open follow-up:** this metadata is Track *3*. Track 1 tiles come from the same WV-3 collection over JAX/OMA, but we must confirm each Track 1 tile traces back to a source scene (filename or TIFF tags). If it doesn't, the prior still works on Cartosat (Bhoonidhi ships sun angles) — we just lose DFC ground-truth validation of it.

### 6.3 A measurement instrument, not a flythrough

The PS explicitly asks that a user *"validates estimated heights against reference data."* Most teams will skim that on the way to making something pretty.

- Click two points → height delta **with an error bar** (from 6.1)
- Drag a line → terrain cross-section profile
- Load a reference DSM → instant difference map + per-terrain-class error breakdown
- Report against the four terrain classes ISRO named: urban, sparse, hilly, forested

A flythrough is a screensaver. This is an analyst's tool.

### 6.4 Deployability as a scored feature — *this is the "optimisation" answer*

ONNX export, quantised, **CPU-only inference**, air-gapped Docker image. "Runs in N seconds per tile on a laptop, no GPU required" is a deployment story, and government facilities are frequently air-gapped — a demo needing internet is a liability. "Successful standalone deployment" is written into their criteria.

## 7. Risks

### 7.1 Domain gap — the one that matters

DFC2019 is aerial. ISRO evaluates on Cartosat-class imagery with different GSD and look angles. A model tuned only on DFC degrades on their held-out data.

**Mitigation:** pull Cartosat samples from Bhoonidhi in week 1 and validate continuously. Hold out our own test set from day one. Pursue §6.2 as the structural fix.

### 7.2 The reference repo is empty

`IMG-PROCESS-SAC/SIH-DepthWizard-2026` holds only a README ("This repository contains dataset regarding the problem-statement of DepthWizard"). It gained that README after 22 Aug, so data is coming — timing unknown. **It has 5 stars: other teams are watching.**

**Mitigation:** our public data path is fully independent. Keep the output layer modular so a changed file format is a config edit, not a rewrite. Everyone watches the repo.

### 7.3 They are format-strict

ISRO's sibling hackathon repo mandated exact output directory structure, exact file naming, explicit band ordering, plus code, model weights and a technical report.

**Assume the same. Teams with better models lose to teams that follow the spec.**

### 7.4 Do not oversell accuracy

The jury are SAC image-processing specialists. *"Relative DSM is solid, absolute scaling is ±8% against SRTM, here is the error map"* reads as competent. A centimetre-accuracy claim gets dismantled in questions.

### 7.5 Build what you can explain

At the finale you answer live questions — *"why is your error higher in forested terrain?"* Nothing helps but having built it and understood it. **Any component nobody on the team can explain from memory is a liability**, however well it performs.

### 7.6 What actually beats us

Not a team with a better model. A team with a **mediocre model and a beautiful viewer**, because they collected the half we neglected. Or a team that followed the output spec while we didn't.

## 8. Data

### Training / benchmark — DFC2019 Track 1

[IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019), free account. Single-view RGB paired with ground-truth AGL height — *literally this task*, with published baselines and metrics, so we report against an established benchmark instead of hand-waving.

| File | Size | Contents |
|---|---|---|
| `Track 1 / Training data / RGB images 1/1` | 6.59 GB | `*_RGB.tif` inputs |
| `Track 1 / Training data / Reference` | 8.17 GB | `*_AGL.tif` heights (m), `*_CLS.tif` classes |
| `Track 1 / Validation data` | 683 MB | imagery only, truth withheld |
| `Track 3 / Metadata` | 138 KB | **gate for §6.2** — check for sun angles |

Skip all MSI (30 GB) and all of Tracks 2/3/4. Cities: JAX (Jacksonville), OMA (Omaha).

Metrics + baselines: [pubgeo/dfc2019](https://github.com/pubgeo/dfc2019), MIT.
SOTA reference line: [TSE-Net](https://github.com/zhu-xlab/tse-net) reports on DFC2019.

#### Measured facts — downloaded, extracted and probed 26 Aug 2026

The DFC2019 documentation specifies none of the following, so `tools/prepare_data.py probe` measures them and every later stage reads the report rather than a hardcoded guess.

| Property | Measured value | Why it matters |
|---|---|---|
| Tiles | **2 783** RGB+AGL+CLS triples, **108 regions** | 108 regions is enough for a clean geographic split |
| Tile size | **1024 × 1024** | decides the crop geometry, below |
| dtypes | RGB `uint8`, AGL `float32` | |
| **Georeferencing** | **none — no CRS, no transform** | ⚠️ see below |
| GSD | **not in the files**; US3D/DFC2019 Track 1 is **0.3 m/px** | must be passed as `--gsd 0.3` |
| Height range | −2.98 m to 204.15 m | |
| Height distribution | p50 **0.0**, p75 4.6, p95 **13.2**, p99 21.1 | p95 sets the model's output scale |
| Void | **NaN only, and negligible** — 6/200 tiles contain any, then <0.005% of pixels | void handling is nearly a non-issue |
| Split | **76 train / 16 val / 16 test regions**, seed 1337 | region-level, so no tile neighbourhood spans two splits |

**⚠️ The tiles carry no georeferencing at all.** They were rewritten by `tifffile.py`, which stripped CRS and affine transform — `rasterio` returns the identity matrix and warns. Nothing in the files says a pixel is 30 cm. Every distance the viewer reports is therefore derived from an explicitly supplied `--gsd`, and `export_terrain.py` refuses to invent one: with no GSD it labels the axes *pixels* rather than printing confident nonsense. Sanity check: 1024 px × 0.3 m = 307 m per tile, which is the right order for a city block.

**Class distribution** (60 random tiles, 62.9 M px): ground 66.3%, building 15.8%, vegetation 13.2%, water 2.4%, bridge 1.2%, and **65 = "unlabeled" 1.1%** — an undocumented code. Its heights are finite and valid, so those pixels stay in the regression but are excluded from per-class breakdowns.

Median height by class: vegetation **7.97 m**, building **6.83 m**. *Vegetation is taller than buildings in this dataset* — Jacksonville tree canopy — which is worth remembering before reading too much into any single aggregate number.

#### Crop geometry — 518, not 256

The Depth Any Canopy recipe uses 256×256 crops. **We cannot.** DA-V2's patch size is 14 and its head computes `patch_h = H // 14`, then interpolates its output to `patch_h × 14`. 256 is not a multiple of 14, so a 256×256 input silently returns **252×252** — a size mismatch that reaches the loss as a confusing broadcast error, or worse, quietly works after an accidental resize.

So: **crop 518** (= 14 × 37, and DA-V2's native training resolution), **stride 506**. Since 1024 = 506 + 518, that tiles each tile exactly 2×2 with 12 px of overlap and full coverage. The ground truth is never resampled, so no interpolation error is baked into the labels. `model.check_input_size()` raises rather than let a bad size through.

Result: **~11 000 crops, ~9.5 GB of compressed shards.**

### Domain imagery — Bhoonidhi (Cartosat)

ISRO's own portal, free registration. The API is a **STAC** interface (use `pystac-client`), requested via `bhoonidhi@nrsc.gov.in` — **lead-time item, send in week 1.** The API is optional; the *account* is what matters, and browser download works without it.

**Access tiers, per Indian Space Policy 2023 (from the NRSC Bhoonidhi brochure):**

| Data | Tier |
|---|---|
| 5 m resolution **and coarser** | Free and open to everyone |
| **Finer than 5 m** | Free to Government Entities *on declaration*; **priced for Non-Government Entities** |
| 30 m Carto DSM | Open for GE, priced for NGE |

| Sensor | GSD | Us as an NGE |
|---|---|---|
| Cartosat-1 / 2 / 3 | 2.5 → 0.25 m | **priced** |
| Resourcesat LISS-IV | 5.8 m | **free** |
| LISS-III / AWiFS | 23.5 / 56 m | free |

LISS-IV is free and genuinely Indian, but at 5.8 m a building spans 2–3 pixels — adequate for terrain relief, weak for building height, and a DSM needs both.

**Worth pursuing:** GEs get finer-than-5 m data free on declaration. SIH is a Government of India initiative under the Ministry of Education, this is an ISRO problem statement, and many Indian engineering colleges are government institutions. Ask Bhoonidhi directly whether that qualifies — worst case is a no.

**Not a blocker either way.** `meanOffNadirViewAngle` spans 4.8°–28.9° across the 67 DFC scenes (§6.2), so we can already measure accuracy degradation vs look angle on data we hold. Cartosat strengthens the domain story; its absence does not sink it.

### Metric anchor — SRTM 30 m / Copernicus GLO-30

Public, free, named in the problem statement. *Not* CartoDEM (see §5 Stage 02).

## 9. Schedule

Build the baseline first: once it runs end to end we always have something demoable, and every later improvement becomes optional rather than critical-path.

### Week 1 — The ugly vertical slice

**Goal: a complete end-to-end path, however bad the output looks.**

- [x] Track 3 metadata checked for sun angles → §6.2 gate passed (26 Aug)
- [x] DFC2019 Track 1 extracted, probed, sharded; held-out test split carved out (26 Aug)
- [x] **Zero-shot DA-V2 → heightmap → GeoTIFF → Three.js flythrough, working end to end** (26 Aug)
- [x] Honest zero-shot baseline on 80 held-out tiles → **4.68 m RMSE** (test split; the val bar is 9.31 m) (§5)
- [x] Training loop verified end to end: resumable, bf16/fp16 autodetect, per-class + calibration eval
- [x] Loss suite under test — 19 tests, incl. recovery of known heteroscedastic σ
- [x] Email Bhoonidhi (§8) — **sent 26 Aug**
- [ ] Resolve SIH team registration (§11) ← *the only unrecoverable open item*
- [ ] Verify Azure GPU quota (expect it blocked; nothing lost if so)

*The output will be mediocre. Irrelevant. After this week there is always something to
demo, and nothing left that is critical-path.*

**Closed 26 Aug.** The slice runs: image → sliding-window inference with cosine blending
→ GeoTIFF → browser bundle → Three.js flythrough with texture / height / uncertainty /
slope surfaces and a two-point measurement tool. Ground truth is exported as a second
scene so the two can be flown side by side.

**Local compute turned out to be far less of a constraint than assumed.** Measured on the
3060: batch 12 fits in 7.6 GB and throughput plateaus near **25 crops/s** at batch 8, so
one full 3-epoch pass over ~7 800 training crops is roughly **15 minutes**, not the
5–9 hours §5 estimated from the A6000 figure. That estimate assumed a much larger crop
count. The consequence is strategic: we can afford many training runs and real ablations
locally, and Kaggle becomes a parallel-experiments resource rather than a dependency.

### Week 2 — Make the model good

- [x] DA-V2 Small fine-tuned on DFC2019 AGL, uncertainty head in from the first run (§6.1)
- [ ] Metric calibration against SRTM 30 m
- [x] **The honest table:** zero-shot vs fine-tuned vs TSE-Net published, same held-out tiles
- [ ] Benchmark remaining backbones on Kaggle (parallel sessions, 30 GPU-h/week)
- [x] Calibration curves — is the predicted sigma actually right?

*Slow down here and understand the calibration and the NLL properly (§10). These are what
the jury probes hardest.*

### Week 3 — Make the viewer good, and measure the domain

- [ ] Confidence shading on the mesh (§6.1) — judges *see* the model's doubt
- [ ] Point-to-point height measurement with error bar (§6.3)
- [ ] Error maps + per-terrain breakdown: urban / sparse / hilly / forested
- [x] Accuracy vs `meanOffNadirViewAngle` (4.8°–28.9°) — the domain-gap number we can get
      without Bhoonidhi (§7.1)
- [ ] Cartosat check if access came through

### Week 4 — Package and submit

- [ ] ONNX export, quantisation, CPU-only inference timing (§6.4)
- [ ] Air-gapped Docker image; deploy demo to Azure
- [ ] Output format compliance pass (§7.3)
- [ ] Technical report + benchmark write-up
- [ ] Idea submission written **against the evaluation criteria, in their language**
- [ ] Re-check live submission counts on the portal (§3)

## 10. Execution — solo build

**Decided 26 Aug 2026: this is a solo build.** The six-person split is withdrawn. That
invalidates the original competitive premise — *"both halves build in parallel from hour
one, so nobody is idle"* — because one person cannot parallelise. Everything is serial now.

### The sequencing inverts

For a team: model baseline first, viewer second. **Solo that is the failure mode** — three
weeks perfecting a model, a rushed viewer, half the marks forfeited.

**Vertical slice first.** Week 1 produces a complete, ugly, end-to-end path: image in →
*some* heightmap (zero-shot is fine) → mesh → flythrough in the browser. It will look
mediocre. Irrelevant. Once it exists, every later improvement is optional rather than
critical-path, and there is no world where we hold an excellent model and nothing to show.

Improve the model *behind a viewer that already works*.

### Two timelines — this is what makes solo viable

| Date | Deliverable |
|---|---|
| **20 Sep 2026** | **Idea submission** — a proposal judged on approach. Needs *evidence*: benchmark table, error maps, calibration curve, demo video. Not a finished product. |
| Grand finale (later, if shortlisted) | 36 hours of building. The real runway. |

25 days from 26 Aug. Solo, that is enough for evidence and a demo. It is not enough for
four differentiators plus a polished system — hence the scope call below.

### Scope, revised for solo

| | Verdict |
|---|---|
| **6.1 Uncertainty** | **Committed.** Already in `losses.py`. Cheapest, best return. |
| **6.3 Measurement tooling** | **Committed, trimmed.** Point-to-point height + error bar for submission. Cross-sections and difference maps → finale phase. |
| **6.4 Deployability** | **Committed.** Mostly packaging, explicitly scored. |
| **6.2 Shadow prior** | **Relocated to finale phase.** Not cut — solo, shadow detection + geometry + validation is a week we do not have before 20 Sep. It is excellent *submission* material as a stated method with evidence behind it (54 of 67 scenes carry usable solar geometry). Full credit for the insight in the proposal; build it on the finale runway. |

### The explain-it-yourself risk is now concentrated

§7.5 applies with force: solo, with AI-assisted code, **every component that cannot be
explained from memory is a liability carried by one person.** Standing rule — readable over
clever, and the metric calibration and uncertainty head get walked through properly, because
those are what a SAC jury probes hardest.

## 11. Open decisions

- [x] ~~Who takes the two CV seats~~ — solo build, concept withdrawn (26 Aug)
- [x] Viewer framework → **Three.js** (26 Aug)
- [x] Does DFC2019 ship solar metadata → **yes**, gate passed (26 Aug), see §6.2
- [x] Differentiator scope → 6.1 / 6.3 / 6.4 committed, 6.2 to finale phase (26 Aug)
- [ ] **Six names registered for SIH?** Rules require a six-member team from one institution,
      normally including at least one female member. Carrying the build alone is fine and
      common; having no registered team means nothing gets submitted. **Resolve in week 1.**
- [ ] Commit to SIH26175 or hedge to SIH26143 — **decide by 10 Sep on live counts**
- [ ] Bhoonidhi access tier — awaiting reply (§8)

## 12. Standing rules

1. Never let the viewer be the last night's work — 50% of marks.
2. Report uncertainty, never bare precision (§7.4).
3. Follow their output spec exactly, even over a better model (§7.3).
4. Build what you can explain live (§7.5).
5. Hold out a test set from day one; never tune against it.
6. Preprocess locally, train on Kaggle, deploy on Azure.
7. No code inside notebooks — clone from git.
