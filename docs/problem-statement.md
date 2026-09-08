# SIH26175 — official problem statement (verbatim)

Scraped from https://sih.gov.in/sih2026PS on **2026-09-06**. This is ISRO's wording, not
our paraphrase. The team brief PDF and PLAN.md §1 are interpretations — when they
disagree with this file, this file wins.

| Field | Value |
|---|---|
| Problem Statement ID | 26175 |
| Title | DepthWizard - Single-View Height Estimation and 3D Flythrough |
| Organization | Indian Space Research Organisation (ISRO) |
| Department | Department of Space / Indian Space Research Organisation |
| Category | Software |
| Theme | Disaster Management |
| Submissions | 0/500 as of 2026-09-06 |
| **Deadline** | **30 September 2026** |
| Reference repo | https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026 |
| Recommended dataset | GAMUS — https://huggingface.co/datasets/earthflow/GAMUS |

## Background

Accurate Digital Elevation Models (DEMs) and Digital Surface Models (DSMs) are
fundamental to urban planning, disaster management, and military reconnaissance.
Traditionally, elevation data is acquired through stereo-imaging pairs, LiDAR, or
Interferometric Synthetic Aperture Radar (InSAR). These approaches can be
cost-prohibitive, dependent on specific sensor availability, and computationally
intensive. Single-view height estimation offers an agile alternative, but foundational
monocular depth models are trained largely on natural egocentric imagery and predict
relative depth. When applied to remote sensing, they face domain gaps, structural
variations, and a lack of absolute-scale mapping. Converting relative depth into metric
elevation remains a critical challenge, alongside the operational need to transform
static elevation profiles into interactive 3D assets that can be navigated in real time.

## Description

Develop an end-to-end software pipeline that transforms single-view optical RGB
remote-sensing images into high-precision elevation maps. The framework must support both
non-georeferenced and georeferenced imagery.

- **Non-Georeferenced RGB Imagery** (for example, PNG or JPG): Produce a Relative Digital
  Surface Model (rDSM) for images without spatial metadata.
- **Georeferenced RGB Imagery** (for example, GeoTIFF): Produce an Absolute Digital
  Surface Model (DSM) with metric height values for images containing coordinate-system
  metadata.

The solution should use a pre-trained monocular depth-estimation backbone to generate
initial relative-depth maps. For georeferenced imagery, a lower-resolution DEM source such
as SRTM or a limited set of Ground Control Points may be used to map scale-agnostic depth
features to absolute metric elevations. For non-georeferenced imagery, relative height may
be used directly in the visualization stage.

After computing the elevation map, the system should project the original optical image
onto a generated 3D terrain mesh and integrate the result with a rendering engine such as
Unity, Three.js, or Babylon.js. The interface should support seamless first-person
navigation and analysis of structural heights and slopes from arbitrary aerial
perspectives.

## Key Milestones

- **Elevation Extraction:** Use a robust pre-trained monocular depth model to extract
  geometric and structural representations from single-view optical imagery.
- **Scale Calibration:** Develop a module that converts relative depth to absolute height
  using scene-level statistics, low-resolution DEMs, semantic priors, or minimal Ground
  Control Points for georeferenced inputs.
- **Visualization Layer:** Build an immersive, preferably interactive, rendering pipeline
  that converts the optical texture and derived depth map into a navigable 3D environment
  deployable as a standalone application.

## Evaluation Criteria

- **DSM Estimation - Accuracy and Validation (50%):** Evaluate RMSE, MAE, and correlation
  against LiDAR or reference data, including performance stability across urban, sparse,
  hilly, and forested landscapes.
- **Visualization - Rendering Quality and User Experience (50%):** Assess projection
  accuracy, visual fidelity, navigability of the 3D flythrough, interface intuitiveness,
  software stability, and successful standalone deployment.

## Expected Solution

Deliver a fully integrated software suite with complete source code and technical
documentation. The solution must be deployable as a unified module containing the
following components:

- **Elevation Estimation Module:** Accept single-view optical satellite imagery in PNG,
  JPG, or TIFF format and output a high-fidelity DSM in a standard geospatial format.
- **Interactive Visualization Platform:** Provide a user-friendly 3D flythrough experience
  that lets users upload imagery, visualize reconstructed terrain, and validate estimated
  height values against reference datasets.

## Dataset Link (as printed on the portal)

> Any high-resolution remote-sensing dataset openly available on the internet may be used
> for development. Reference dataset: https://github.com/IMG-PROCESS-SAC/SIH2026/. A
> lower-resolution DEM source such as SRTM 30 m may be used to map scale-agnostic depth
> features to absolute metric elevations. During final evaluation, ISRO RGB-band optical
> satellite i

**The portal truncates this field mid-word** ("satellite i…"). The visible direction is
that final evaluation uses ISRO RGB-band optical satellite imagery — i.e. a different
sensor and GSD from any training set we have. Treat cross-sensor generalisation as a
scored risk, not a nicety.

## What ISRO's own repo adds

`IMG-PROCESS-SAC/SIH2026` redirects to **`IMG-PROCESS-SAC/SIH-DepthWizard-2026`**. Its
README restates the above and adds a dataset recommendation:

- **GAMUS** (provider Earthflow, CC-BY-4.0) is the "recommended open-source foundation."
- Structure: `images/`, `heights/` (`*_AGL.h5`), `classes/` — 5 004 train / 859 val /
  2 861 test = 8 724 tiles. Cities: **PHL 4 398, NYC 2 167, DC 2 159** — all dense US urban.
- Dataloader: https://github.com/EarthNets/RSI-MMSegmentation
- Explicit escape hatch: *"While GAMUS is recommended, you are free to utilize any
  open-source dataset containing remote-sensing depth data, provided it supports both
  relative depth training and metric scale calibration."*
- Stated purpose: *"Use this data to overcome the domain gap between natural egocentric
  imagery … and top-down remote sensing imagery."*

## Deltas against what we had written down

1. **Deadline is 30 September 2026, not 20 September.** PLAN.md, README.md and the team
   brief all say 20 Sep. The portal prints 30 September 2026 in all 233 rows, with no
   other date anywhere on the page.
2. **GAMUS is the sponsor-recommended dataset; we built on DFC2019.** DFC2019 remains
   permitted, but GAMUS is 3× the tiles and carries the sponsor's endorsement.
3. **Structural height and slope analysis is required, not a differentiator.** "analysis
   of structural heights and slopes from arbitrary aerial perspectives" is in the
   Description. PLAN §6.3 treats measurement tooling as an optional trim.
4. **"Standalone application" / "standalone deployment" appears three times**, including
   inside the 50% rendering score.
5. **Theme is "Disaster Management."** Portal themes are frequently mislabelled, but this
   is the label a jury sees.
