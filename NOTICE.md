# Attribution and third-party notices

DepthWizard is licensed under the Apache License 2.0 (see `LICENSE`).

This project builds on other people's work. Where a licence requires attribution, it is
given here; where a dataset is *not* redistributed by us, that is said plainly so nobody
assumes a licence we do not have.

## Model

**Depth Anything V2 — Small** (`depth-anything/Depth-Anything-V2-Small-hf`)
Yang et al., *Depth Anything V2*, NeurIPS 2024. Licensed **Apache-2.0**, which is why it is
the size we ship: the Base, Large and Giant checkpoints are **CC-BY-NC-4.0** and cannot go
into a deliverable that anyone might use commercially. Our fine-tuned weights are derived
from the Small checkpoint and inherit Apache-2.0.

## Software

**three.js** — MIT. Vendored at `viewer/vendor/three.module.js` and inlined into
`viewer_standalone.html`.

**pubgeo/dfc2019** — MIT. Reference baselines and metric definitions for the Data Fusion
Contest, used to check our own scoring against the field's.

Python dependencies keep their own licences; see `requirements.txt` and
`deploy/requirements-serve.txt`.

## Imagery and reference data

**Maxar Open Data Program** — imagery over Sikkim, India, used for the two `hilly_sikkim_*`
viewer scenes and baked into `viewer_standalone.html` as `texture.jpg`.
Released under **CC-BY-4.0**; © Maxar Technologies, via the Maxar Open Data Program.
Attribution also travels *inside* the product: each affected scene manifest carries an
`imagery_source` field, so the credit survives someone opening the standalone file alone
with no repository and no README.

**GAMUS** (Earthflow / `earthflow/GAMUS` on Hugging Face) — **CC-BY-4.0**. Aerial imagery
and LiDAR-derived nDSM over Washington DC and Philadelphia; one of our two training corpora
and the dataset recommended by the problem statement. Not redistributed here.

**IEEE GRSS Data Fusion Contest 2019, Track 1 (US3D)** — Jacksonville and Omaha imagery with
airborne LiDAR truth; our other training corpus and our frozen evaluation set. **Not
redistributed here** — obtain it from IEEE DataPort under their terms:
<https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019>

**Google Open Buildings 2.5D Temporal** — building heights over South Asia, used *only* as an
independent cross-check over India and never as training data or ground truth. Published by
Google under **CC-BY-4.0**; see the dataset page for the current terms.

**Copernicus DEM GLO-30** — ESA / Copernicus Programme, used as the absolute elevation anchor
and as the terrain base under the Sikkim scenes. Free to access and use under the Copernicus
DEM licence; consult ESA's terms for the authoritative conditions.

**ISRO Bhoonidhi** (NRSC) — CartoDEM and LISS-4 products were evaluated during the project.
Nothing from Bhoonidhi is redistributed in this repository.

## What is not in this repository

Model weights, GeoTIFFs, `.npz` shards and baked viewer scenes are excluded by `.gitignore`
because they are large and, in the case of the datasets above, not ours to redistribute. A
fresh clone therefore builds and runs but opens an empty scene picker until scenes are
regenerated — `tools/build_scenes.py` does that, given the data.
