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

## 5. Method

Three stages. Only the middle is genuinely novel work.

### Stage 01 — Relative depth · *CV pair*

Depth Anything V2 (DINOv2 encoder + DPT decoder) producing a scale-free depth map.

> **This is NOT zero-shot.** The team brief's claim that "the weights already exist, we train nothing" is wrong, and it is the plan's one load-bearing wrong assumption. DA-V2 degrades badly at nadir: at pitch ≈ -90° there are no horizon cues, and it overestimates tree height from straight down. Its priors are natural photographs with a ground plane and a vanishing point. A satellite tile has neither. Run zero-shot it correlates with rooftop albedo, not height.

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

### Week 1 — Baseline end to end

- [ ] DFC2019 Track 1 downloaded, extracted, sharded; held-out test split carved out
- [ ] **Check Track 3 metadata for sun angles** → decides §6.2
- [ ] Email Bhoonidhi for API access
- [ ] Verify Azure GPU quota (expect it to be blocked)
- [ ] DA-V2 Small fine-tuned on one city → relative depth → SRTM-anchored metric DSM
- [ ] **The honest table:** zero-shot vs fine-tuned vs TSE-Net published, same held-out tiles
- [ ] Uncertainty head + Gaussian NLL in from the first run (§6.1)

*One tile, one number. Proves the core and kills "is this too hard" empirically.*

### Week 2 — The viewer

- [ ] Three.js flythrough over week-1 heightmaps
- [ ] Upload → DSM → flythrough path working end to end
- [ ] Confidence shading on the mesh (§6.1)
- [ ] Measurement tools: point-to-point delta with error bar, cross-section (§6.3)

*The half the judges watch. Building it early de-risks the demo.*

### Week 3 — Accuracy and domain

- [ ] Benchmark all backbones on DFC2019 metrics (Kaggle sweep, parallel)
- [ ] Pull Cartosat imagery from Bhoonidhi, measure the drop (§7.1)
- [ ] Shadow prior as self-supervision on Cartosat, if gated in (§6.2)
- [ ] Error maps + per-terrain breakdown: urban / sparse / hilly / forested
- [ ] Calibration curves for the uncertainty head

### Week 4 — Package and submit

- [ ] ONNX export, quantisation, CPU-only inference timing (§6.4)
- [ ] Air-gapped Docker image; deploy demo to Azure
- [ ] Output format compliance pass (§7.3)
- [ ] Technical report + benchmark write-up
- [ ] Idea submission written **against the evaluation criteria, in their language**
- [ ] Re-check live submission counts on the portal (§3)

## 10. Team split — six people

| Seats | Owns |
|---|---|
| **Depth & calibration — 2** | Stages 01–02. Backbone benchmarking, relative→metric regression, uncertainty head, error analysis across terrain types. *Carries the half we cannot fake.* |
| **Viewer & deployment — 2** | Stage 03 + standalone build. Upload flow, mesh displacement, camera navigation, measurement tooling, ONNX/Docker. |
| **Data & geospatial — 1** | DFC2019 + Bhoonidhi acquisition, GeoTIFF handling, SRTM alignment, held-out test set, output format compliance. |
| **Report & demo — 1** | Technical documentation, benchmark write-up, demo script. *Half the marks are judged on what the jury sees and understands.* |

## 11. Open decisions

- [ ] Who takes the two CV seats
- [ ] Commit to SIH26175 or hedge to SIH26143 — **decide by 10 Sep on live counts**
- [ ] Viewer framework: Three.js (default) vs Babylon.js
- [ ] Does DFC2019 ship solar metadata → §6.2 headline or footnote

## 12. Standing rules

1. Never let the viewer be the last night's work — 50% of marks.
2. Report uncertainty, never bare precision (§7.4).
3. Follow their output spec exactly, even over a better model (§7.3).
4. Build what you can explain live (§7.5).
5. Hold out a test set from day one; never tune against it.
6. Preprocess locally, train on Kaggle, deploy on Azure.
7. No code inside notebooks — clone from git.
