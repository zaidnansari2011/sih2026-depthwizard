# What the literature actually says — and what it changes

Findings from a targeted read on 27 Aug 2026, written down because two of them
contradict decisions already taken in this repo.

## 1. Our failure mode is *the* known failure mode of this field

We measured, on our own region-disjoint split, that the model **lifts ground and
flattens buildings** — ground bias +2.53 m, building bias −3.30 m zero-shot,
degrading to −8.56 m after training. I had treated this as a quirk of our setup.

It is the central, named problem of monocular height estimation. From
[HTC-DC Net (Chen et al., IEEE TGRS 2023)](https://arxiv.org/abs/2309.16486):

> "The distribution of height values is long-tailed with the low-height pixels,
> e.g. the background, as the head... trained networks are usually biased and tend
> to **underestimate building heights**... predictions can include many fatal cases
> for higher objects with incredibly large errors."

That is our error profile verbatim: 91.3% of squared error concentrated in the
12.8% of pixels that are buildings.

**Consequence:** the fix is not a loss-weighting tweak. Every paper that moves the
needle on this reformulates the *output space*.

## 2. The reformulation that works: classification–regression, not regression

Published DFC19 results (Table I of the same paper), pixelwise RMSE in metres:

| Method | RMSE | building px | non-building px | building-wise |
|---|---|---|---|---|
| SegNet | 7.1397 | 14.6632 | 4.9302 | 5.0540 |
| FCN-8s | 3.8070 | 7.5295 | 2.7626 | 3.7290 |
| **U-Net** (best plain regression) | **2.9776** | 5.7762 | 2.2098 | 3.3759 |
| IM2HEIGHT | 4.8139 | 10.5586 | 2.9693 | 4.1436 |
| Amirkolaee & Arefi | 2.8709 | 5.7860 | 2.0345 | 3.2337 |
| DORN (classification) | 3.5755 | 6.4091 | 1.9402 | 3.2285 |
| PLNet | 3.2484 | 6.6785 | 2.2397 | 3.5126 |
| **HTC-DC Net B5** | **2.1813** | 3.8725 | 1.7588 | 2.7850 |
| **HTC-DC Net B7** | **2.1184** | 1.5173 | 2.1785 | 2.3236 |

Note B7's building RMSE (1.5173) is *lower* than its non-building RMSE and less
than half B5's — buildings being easier than flat ground is not credible. Treat
B5's 3.87 m as the defensible reference for building pixels and flag B7's cell as
suspect rather than quoting it.

The method, in four pieces, all portable to our DPT head:

1. **Adaptive bins.** A ViT branch predicts `N=256` relative bin widths via
   `softmax(fc(e₁))`; edges are the cumulative sum from `h_min`. Bins adapt per image.
2. **Soft-argmax output.** `H = Σᵢ Pᵢcᵢ` over bin centres — continuous output from a
   classification head, no discretisation artefacts.
3. **Head-tail cut (HTC).** Trivially simple: **foreground = height > 1 m**. Two
   separate token sets produce FG and BG bin probabilities; a sigmoid head on the FG
   range-attention map picks between them, supervised by cross-entropy against
   `H̃ > 1`. Worth **0.12 m RMSE / 0.05 m on building pixels** on its own (Table VI) —
   real but small.
4. **Distribution-based constraint (DC).** Assume the per-pixel height is Gaussian
   about the ground truth; solve σ analytically from the predicted probability of the
   bin containing the truth; integrate that Gaussian over every bin to get reference
   bin probabilities; apply **KL divergence** against the predicted ones. This forces
   the expectation (what soft-argmax computes) to sit at the mode.

Losses: L1 on height + **Chamfer** between bin edges and flattened ground-truth
heights (so edges land on the truth quantiles) + HTC cross-entropy + DC KL.
AdamW, lr 1e-4, early stop after 10 epochs without improvement.

**Note the DC is a close cousin of what we already have.** Our heteroscedastic head
already predicts a per-pixel σ. HTC-DC Net *derives* σ analytically and uses it to
shape a distribution over bins. We are one architectural step away, not a rewrite.

## 3. The published numbers are NOT comparable to ours — and this matters

Their DFC19 protocol, quoted exactly:

> "the patches are cropped into 44 258 smaller patches of size 256 × 256... **randomly
> split** into training, validation, and test sets, with 31 152, 4432, and 8944 data
> samples"

DFC19 covers **two cities** — Jacksonville FL and Omaha NE, ~100 km² total. Randomly
splitting 256×256 crops means the crop immediately adjacent to a test crop — same
building, same rooftop, same shadow, same street — is in the training set. That is
textbook spatial leakage.

Our split is region-disjoint: no region in test appears in train. **This is why our
4.68 m zero-shot and their 2.12 m are not the same number**, and it would be wrong
to present our result as losing to theirs, or to quietly adopt their protocol to
close the gap.

**What we do about it:** report the region-disjoint number as our headline, and say
plainly why it is not directly comparable to published DFC19 figures. If time allows,
also report a random-crop-split number as a second row — clearly labelled — so a
reader can place us against the literature without us having to fudge the primary
metric. Two honest numbers beat one flattering one, and the gap between them is
itself a result worth showing.

## 4. What this changes in the plan

- The β-NLL ablation (run02) is a **weaker lever than assumed**. Epoch 0 already
  showed β=0.5 improving global RMSE while making building bias *worse*
  (−7.76 vs −4.78). The literature explains why: reweighting a regression loss does
  not fix a long-tailed *output space*. Finish run02 for the record; do not expect it
  to be the answer.
- The next architectural experiment should be a **binned classification–regression
  head with soft-argmax**, not a further β sweep.
- Keep the heteroscedastic σ. It is a differentiator, it is compatible with the
  binned head, and the DC shows the two ideas compose.

Sources: [HTC-DC Net (arXiv 2309.16486)](https://arxiv.org/abs/2309.16486) ·
[code](https://github.com/zhu-xlab/HTC-DC-Net) ·
[TSE-Net (arXiv 2511.13552)](https://arxiv.org/pdf/2511.13552) ·
[AdaBins lineage / survey context](https://arxiv.org/html/2603.29245)

---

## 5. Soft-argmax needs distribution supervision, or it will not help us

Added 27 Aug 2026, before implementing the binned head. This contradicts the plan in
section 4 above, which said to build soft-argmax first and treat the distribution
constraint as a later refinement.

Soft-argmax computes `H = sum_i P_i * c_i` - the **mean** of the predicted distribution.
That is only close to the mode when the distribution is unimodal and peaked. The stereo
and depth literature documents what happens when it is not:

> "3D stereo architectures typically employ a soft argmax operation to obtain the final
> disparity estimate by calculating the mean of the often multimodal predicted
> distribution, which leads to disparity estimates falling between the foreground and
> background modes, resulting in erroneous predictions and over-smoothed depth
> discontinuities."

and, on why nothing prevents this by default:

> "Networks trained with soft-argmax lack explicit supervision for the distribution,
> resulting in unconstrained probability shape."

**A roof edge is exactly a foreground/background bimodal pixel** - the truth is either
roof height or ground, never the average. Supervising only the mean lets the network put
mass on both modes and score well on the mean while bleeding the edge. Our error is
already concentrated on buildings (91.8% of squared error on 12.8% of pixels), and
building edges are where a large share of that lives.

**Consequence for the build:** bins + soft-argmax + Chamfer alone is not the safe
increment it looked like. The distribution has to be supervised in the same run, or the
experiment does not test what we think it tests. HTC-DC Net's distribution constraint
(KL against a Gaussian reference over the bins) exists for this reason, and its
head-tail cut - separate token sets for foreground and background, split at 1 m - attacks
the same bimodality from the architecture side.

**What we implement:** adaptive bins normalised the AdaBins way, soft-argmax, the
bi-directional Chamfer term on bin centres (AdaBins uses beta = 0.1), **and** a
cross-entropy term against a Gaussian centred on the truth, which is the distribution
constraint in its simplest defensible form. The existing heteroscedastic sigma head and
its NLL stay - run02 showed that head is working (ECE 0.083, rank rho +0.836) and it is
a committed differentiator.

Bin count: the ablations do not support paying for many bins. Gains saturate early
(4 -> 8 bins is large, 16 -> 32 marginal), 256 is convention rather than a measured
optimum, and one recent method reports N = 128 as the best accuracy/robustness
trade-off. We use N = 128, which is also what fits: bin logits at 518x518 cost ~550 MB
per 4 images in bf16, on a card that run02 already had at 10.2 of 12 GB.

Sources: [AdaBins (arXiv 2011.14141)](https://arxiv.org/abs/2011.14141) ·
[BinsFormer (arXiv 2204.00987)](https://arxiv.org/pdf/2204.00987) ·
[LocalBins (arXiv 2203.15132)](https://arxiv.org/pdf/2203.15132) ·
[IEBins (arXiv 2309.14137)](https://arxiv.org/html/2309.14137) ·
[Sampling-Gaussian for stereo matching (arXiv 2410.06527)](https://arxiv.org/pdf/2410.06527) ·
[Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching](https://arxiv.org/pdf/2407.07816)
