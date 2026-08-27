# Viewer design notes

Half the marks. ISRO scores it as *"projection accuracy, visual fidelity, navigability of
the flythrough, interface intuitiveness, stability, and successful standalone deployment."*

## The two rules that outrank every feature below

**1. It has to be readable and usable without explanation.** Zaid's constraint, and it is
the right one: *"the main thing is that the viewer is easy to read and use, it shouldn't be
jargon."* A juror should understand what they are looking at in five seconds, without a
caption, a legend they have to study, or a term they have to already know. If a control
needs explaining, the label is wrong.

This does **not** mean hiding the rigour. It means the plain word goes on the button and the
precise number goes in the readout. "How sure we are, ±2.1 m" beats "sigma (m)" and carries
strictly more information.

**2. No decoration that carries no information.** This is what makes something look
generated rather than built. Purple gradients, glassmorphism cards, emoji headings, a hero
section, controls that exist to look busy. The jury are Space Applications Centre
image-processing specialists who have used QGIS for twenty years.

What reads as an instrument instead:

- The data is the only saturated thing on screen. UI stays dark and near-greyscale; all the
  colour belongs to the height ramp. **Never a rainbow/jet colormap** — it is the classic
  amateur signal in remote sensing, and the current ramps are already perceptually ordered.
- Monospace for numbers. Every number carries a unit.
- One accent colour, used only to show what is interactive.
- Restraint. A viewer that looks slightly austere reads as competent; one that looks
  *designed* reads as compensating.

## Label discipline — plain word outside, precision inside

| avoid | use |
|---|---|
| sigma / σ (m) | **How sure we are** — "±2.1 m" |
| Signed error vs ground truth | **Where we're wrong** — "blue = we said too low, red = too high" |
| Vertical exaggeration ×3 | **Height boost ×3** (say it is exaggerated, so nobody thinks the terrain is that steep) |
| nDSM / AGL | **Height above ground** |
| GSD 0.31 m | **Each pixel is 31 cm on the ground** |
| Hillshade azimuth | **Sun direction** |

## What already exists

Four display modes (satellite drape, height ramp, uncertainty, slope), perceptually ordered
ramps, vertical exaggeration, a tour path, wireframe, reset, a legend, a stats panel, and the
click-two-points measurement with a **calibrated** error bar. The tool is not the problem.

## What is missing, in order of return

**0. Fresh scenes.** The viewer currently ships `zeroshot` and `truth` from week one — it is
demonstrating our *worst* model. Regenerate from run02+TTA (or run05). This changes more
than any new feature.

**1. Truth comparison, drag-to-compare.** DFC2019 ships LiDAR ground truth. A divider the
user drags between our surface and real LiDAR is the most convincing five seconds available
to us, and it is literally the "validation" half of the accuracy criterion. Nothing to
explain: the two pictures either match or they do not.

**2. "Where we're wrong" map.** Diverging ramp centred at zero against the truth. Specialists
read error maps instinctively, and it converts our known weakness into evidence of rigour —
the tall buildings glow, and we explain exactly why (see PLAN §9).

**3. Scene picker = the four terrain types ISRO named.** Urban, sparse, hilly, forested.
Answering their stated criterion in the interface itself. We already compute the per-terrain
breakdown in `evaluate.py`.

**4. A Sikkim scene.** Real Indian mountains at 0.31 m, for an ISRO jury. Even without truth
it says "this works on our terrain", and the Open Buildings cross-check backs it
(`docs/evaluation-protocol.md`).

**5. Sun direction slider (hillshade).** Every serious DEM viewer has one and its absence is
noticeable. It also makes the surface legible in the demo video.

**6. Hover readout** — position, height, and confidence, plus a real scale bar.

## Demo video

The tour path already exists. The video should show, in order: the satellite image, the
surface rising out of it, the drag-to-compare against LiDAR, one measurement with its ±, and
one honest look at where it fails. That last beat is what a specialist jury remembers.
