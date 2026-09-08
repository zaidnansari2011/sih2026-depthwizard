"""DepthWizard: Depth Anything V2 adapted to absolute height with per-pixel uncertainty.

What this changes about the stock model
---------------------------------------
1. Two outputs instead of one. The stock head ends in conv3(32 -> 1); we keep that conv
   (it is pretrained and we want its weights) as the mean branch, and add a second
   1x1 conv for log-variance. Both share conv1/conv2, which is the standard Kendall &
   Gal arrangement and costs almost nothing.

2. No final ReLU. The stock head clamps output at zero because relative depth is
   non-negative. AGL is *almost* non-negative, but a hard ReLU has zero gradient
   everywhere it is active, so any pixel the model pushes below zero early in training
   is dead for the rest of it. We let the loss handle plausibility instead.

3. Output scaled to metres. The pretrained head emits values around unity; AGL runs to
   tens or hundreds of metres. Rather than force the head to learn a large gain through
   a 5e-6 learning rate, we predict in normalised units and multiply by `height_scale`.
   The loss therefore sees metres -- so reported RMSE is directly the number ISRO scores
   -- while the network's internal range stays O(1).

A useful accident of polarity
-----------------------------
DA-V2 predicts *inverse* relative depth: nearer is larger. Viewed from orbit, nearer
means taller. So the pretrained head's sign convention already matches height, and we
are fine-tuning a sensible initialisation rather than fighting one.

Input size constraint
---------------------
The backbone has patch size 14 and the head computes `patch_h = H // 14`, then
interpolates its output to `patch_h * 14`. Feed it 256 and you get 252 back with no
warning. `check_input_size` refuses non-multiples rather than let that mismatch reach
the loss, where it would surface as a confusing broadcast error or, worse, silently
work after an accidental resize.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"   # Apache-2.0. Base/Large are CC-BY-NC.
PATCH = 14


def check_input_size(h: int, w: int, patch: int = PATCH):
    if h % patch or w % patch:
        near_h, near_w = (h // patch) * patch, (w // patch) * patch
        raise ValueError(
            f"input {h}x{w} is not a multiple of the patch size {patch}. The head would "
            f"silently return {near_h}x{near_w}. Use a crop size divisible by {patch} "
            f"(e.g. {near_h} or {near_h + patch})."
        )


class DepthWizard(nn.Module):
    """DA-V2 backbone + DPT neck + twin (mean, log-variance) head.

    Args:
        model_id: HF id of the DA-V2 checkpoint. Small is Apache-2.0; the larger
            variants are CC-BY-NC-4.0, which we cannot ship in a deliverable.
        height_scale: metres per unit of normalised output. Set it near the upper bulk
            of the height distribution (the probe reports one) -- not the maximum, which
            is a lone skyscraper and would squash everything else toward zero.
        init_sigma_m: sigma the variance head starts at, in metres. Starting somewhere
            plausible keeps the NLL well-behaved in the first few hundred steps.
        freeze_backbone: train the neck and head only. Cheap sanity baseline, and a
            useful ablation for the report.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        height_scale: float = 30.0,
        init_sigma_m: float = 5.0,
        freeze_backbone: bool = False,
        predict_uncertainty: bool = True,
        bins: int = 0,
        bin_min: float = -3.0,
        bin_max: float = 120.0,
        htc: bool = True,
        init_mu: str = "pretrained",
        init_mu_m: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        base = AutoModelForDepthEstimation.from_pretrained(model_id)
        cfg = base.config

        self.backbone = base.backbone
        self.neck = base.neck
        self.patch_size = int(getattr(cfg, "patch_size", PATCH))
        self.head_in_index = int(getattr(cfg, "head_in_index", -1))
        self.height_scale = float(height_scale)
        self.predict_uncertainty = predict_uncertainty
        self.model_id = model_id

        # Reuse the pretrained head convs.
        self.conv1 = base.head.conv1
        self.conv2 = base.head.conv2
        self.activation1 = base.head.activation1
        self.conv_mu = base.head.conv3                      # pretrained 32 -> 1

        # The pretrained readout emits DISPARITY, and its output scale is a property of
        # the checkpoint's feature magnitudes, not of our task. Measured on real crops
        # with tools/preflight.py, median |truth - mu| at init is 16.5 m for DA-V2-Small
        # but 666 m for DA-V1-Large: the two families' necks do not emit features of the
        # same size, even though their conv3 weights are comparable (|w| 0.088 vs 0.100).
        # A 666 m residual drives log_var into log_var_max on the first NLL step, and the
        # clamp then zeroes its gradient while suppressing mu's by ~1100x
        # (tools/nll_deadlock_probe.py). That is what killed run05.
        #
        # "pretrained" reproduces the historical behaviour exactly, so run01-run04 remain
        # bit-comparable. "constant" applies to conv_mu the same argument the variance
        # head below already makes for itself: begin as a uniform prediction and let the
        # data add structure. It discards a 32->1 linear readout of disparity, which has
        # to be relearned as height in any case; backbone, neck and conv1/conv2 are kept.
        if init_mu == "constant":
            nn.init.zeros_(self.conv_mu.weight)
            nn.init.constant_(self.conv_mu.bias, float(init_mu_m) / self.height_scale)
        elif init_mu != "pretrained":
            raise ValueError(f"init_mu must be 'pretrained' or 'constant', got {init_mu!r}")
        self.init_mu = init_mu

        if predict_uncertainty:
            hidden = self.conv_mu.in_channels
            self.conv_log_var = nn.Conv2d(hidden, 1, kernel_size=1)
            # Zero weights so the head begins as a constant: sigma is uniform and equal
            # to init_sigma_m everywhere, and only becomes spatially varying once the
            # data gives it a reason to. Random init here lets the variance term shove
            # the mean head around before the mean head knows anything.
            nn.init.zeros_(self.conv_log_var.weight)
            nn.init.constant_(
                self.conv_log_var.bias,
                2.0 * math.log(max(init_sigma_m, 1e-6) / self.height_scale),
            )
        else:
            self.conv_log_var = None

        # -------------------------------------------------------- binned height head
        self.bins = int(bins)
        self.bin_min, self.bin_max = float(bin_min), float(bin_max)
        self.htc = bool(htc) and self.bins > 0
        if self.bins > 0:
            hidden = self.conv_mu.in_channels
            # Head-tail cut (HTC-DC Net, TGRS 2023, Eqns 8-9). Foreground and background
            # get their OWN distribution over bins, selected per pixel by a binary
            # classifier at 1 m. One shared head has to express both a near-delta at
            # ground level and a broad spread over roof heights using the same weights,
            # and run03 showed what that costs: bins and soft-argmax with a loss-only
            # constraint left buildings untouched. This is structural work a loss term
            # cannot do. Measured on our shards, 35.6% of valid pixels sit above 1 m, so
            # the binary problem is close to balanced.
            self.conv_bins = nn.Conv2d(hidden, self.bins, kernel_size=1)
            self.conv_bins_bg = (nn.Conv2d(hidden, self.bins, kernel_size=1)
                                 if self.htc else None)
            self.conv_htc = nn.Conv2d(hidden, 1, kernel_size=1) if self.htc else None
            self.bin_width = nn.Sequential(
                nn.Linear(2 * self.conv1.in_channels, 256), nn.GELU(),
                nn.Linear(256, self.bins))
            # Start the head where the data is. Randomly initialised, the bin logits give
            # a near-uniform distribution whose expectation is the MIDDLE of the height
            # range against a true median of 0, and the first epoch goes on climbing back
            # down. Zero the weights and shape each bias into a Gaussian over the (now
            # uniform) bins, the same trick conv_log_var uses above.
            nn.init.zeros_(self.bin_width[-1].weight)
            nn.init.zeros_(self.bin_width[-1].bias)      # -> uniform widths
            step = (self.bin_max - self.bin_min) / self.bins
            c0 = self.bin_min + (torch.arange(self.bins) + 0.5) * step
            tau = max(self.height_scale / 4.0, 1.0)
            nn.init.zeros_(self.conv_bins.weight)
            if self.htc:
                # The foreground head only ever sees pixels above 1 m, so starting it at
                # ground level would waste the head-tail split from the first step. p90 of
                # our height distribution is 9.85 m; start it there.
                nn.init.zeros_(self.conv_bins_bg.weight)
                self.conv_bins.bias.data.copy_(-0.5 * ((c0 - 9.85) / tau) ** 2)
                self.conv_bins_bg.bias.data.copy_(-0.5 * (c0 / tau) ** 2)
                # The gate keeps its default random weights and a zero bias, on purpose.
                # Zeroing them the way the bin heads are zeroed makes the logit a spatial
                # CONSTANT, the hard threshold then sends every pixel down one branch, and
                # the other head receives exactly zero gradient until the bias drifts
                # across. Measured: with a base-rate bias of -0.594 the gate fired on 0.0%
                # of pixels and the foreground head was dead at initialisation. A zero
                # bias splits it near 50/50 with real spatial variation, and the head-tail
                # cross-entropy pulls it to the measured 35.6% within a few hundred steps.
                nn.init.zeros_(self.conv_htc.bias)
            else:
                self.conv_bins.bias.data.copy_(-0.5 * (c0 / tau) ** 2)
        else:
            self.conv_bins = None
            self.conv_bins_bg = None
            self.conv_htc = None
            self.bin_width = None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        del base

    # ------------------------------------------------------------------ forward

    def forward(self, pixel_values: torch.Tensor, return_bins: bool = False,
                route_mask: torch.Tensor | None = None):
        """pixel_values: (B, 3, H, W), ImageNet-normalised. Returns (mu, log_var) in metres.

        return_bins additionally yields per-pixel bin probabilities and per-image bin
        centres, which the Chamfer and distribution losses need. It defaults to False
        so every existing caller (inference, ONNX export, the viewer) is untouched.
        """
        _, _, H, W = pixel_values.shape
        check_input_size(H, W, self.patch_size)
        ph, pw = H // self.patch_size, W // self.patch_size

        out = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=False, output_attentions=False
        )
        hidden = self.neck(out.feature_maps, ph, pw)
        feat = hidden[self.head_in_index]
        x = feat

        # Mirrors the stock head: upsample after conv1, exactly as in the DPT paper.
        x = self.conv1(x)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size),
                          mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))

        if self.bins:
            mu, aux = self._binned_height(feat, x, route_mask)
        else:
            mu = self.conv_mu(x) * self.height_scale
            aux = None

        if self.conv_log_var is None:
            log_var = None
        else:
            # Variance is predicted in normalised units, so converting to metres is a
            # shift of 2*log(scale) in log space. Doing it here rather than in the loss
            # keeps the loss unit-agnostic and the reported sigma in metres.
            log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)

        if return_bins:
            return mu, log_var, aux
        return mu, log_var

    def _binned_height(self, feat, x, route_mask=None):
        """Adaptive-bin classification head with soft-argmax, plus the head-tail cut.

        One regressed scalar has to span a long-tailed height distribution and
        systematically loses the tail. Predicting a distribution over bins and taking its
        expectation keeps a continuous output while letting the network put its
        resolution where the scene heights actually are.

        Widths are normalised the AdaBins way: a softmax, so they are strictly positive
        and always sum to the full height range. Edges are the cumulative sum, and the
        Chamfer term in losses.py pulls those edges onto the quantiles of the truth --
        the edges, not the centres, which is what HTC-DC Net supervises (Eqn 13).

        The expectation is the MEAN of the distribution, which sits between the modes
        wherever the truth is bimodal, and a roof edge is roof or ground and never the
        average. Two things fight that: the head-tail cut here, which gives roof and
        ground separate distributions rather than making one head straddle both, and
        DistributionConstraint in losses.py, which supervises the shape.
        """
        g = torch.cat([feat.mean(dim=(2, 3)), feat.std(dim=(2, 3))], dim=1)
        widths = F.softmax(self.bin_width(g), dim=1) * (self.bin_max - self.bin_min)
        edges = torch.cat([widths.new_full((widths.shape[0], 1), self.bin_min),
                           self.bin_min + torch.cumsum(widths, dim=1)], dim=1)  # (B, N+1)
        centres = 0.5 * (edges[:, :-1] + edges[:, 1:])                          # (B, N)

        probs = self.conv_bins(x).softmax(dim=1)                    # (B, N, H, W)
        htc_logit = None
        if self.htc:
            p_bg = self.conv_bins_bg(x).softmax(dim=1)
            htc_logit = self.conv_htc(x)                            # (B, 1, H, W)
            # Hard selection, as in the paper (Eqn 9). The gate is not differentiable
            # here on purpose: it is trained directly by its own cross-entropy against
            # (truth > 1 m), and letting height gradients also pull on it would trade
            # classification accuracy for regression convenience.
            # Routing by ground truth for the first steps, by the gate thereafter.
            #
            # The hard threshold is not optional -- a soft blend would mix a roof-peaked
            # and a ground-peaked distribution, and the expectation of that bimodal
            # mixture lands between the two modes, which is the exact failure the
            # head-tail cut exists to prevent. But a hard gate driven by a single 1x1 conv
            # is an initialisation lottery: one random projection decides the sign for
            # nearly every pixel, and whichever branch loses gets no gradient at all.
            # Measured both ways at init here, 0.0% and then 99.4% of pixels.
            #
            # So during warmup the truth does the routing. Both heads then see the right
            # pixels from step one while the head-tail cross-entropy trains the gate on
            # its own terms, and the handover costs nothing because the gate is already
            # accurate by then.
            if route_mask is not None:
                gate = route_mask.to(probs.dtype)
            else:
                gate = (htc_logit.detach() > 0.0).to(probs.dtype)
            probs = gate * probs + (1.0 - gate) * p_bg

        mu = (probs * centres[:, :, None, None]).sum(dim=1, keepdim=True)
        return mu, {"probs": probs, "centres": centres, "edges": edges,
                    "htc_logit": htc_logit}


    # ------------------------------------------------------------------ convenience

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor):
        """Returns (height_m, sigma_m) as (B, 1, H, W) tensors."""
        was_training = self.training
        self.eval()
        mu, log_var = self(pixel_values)
        sigma = torch.exp(0.5 * log_var) if log_var is not None else None
        if was_training:
            self.train()
        return mu, sigma

    def param_groups(self, lr_backbone: float = 5e-6, lr_head: float | None = None,
                     weight_decay: float = 0.01):
        """Two learning rates.

        The backbone is a well-conditioned pretrained representation and wants the small
        rate the Depth Any Canopy recipe uses. The head is doing something the pretrained
        weights were never asked to do -- emit metres, and emit a variance from a
        freshly initialised conv -- so it needs a materially larger rate or it spends the
        whole budget catching up. 50x is a starting point, not a law; it is in the
        ablation list.

        Biases and norms are excluded from weight decay, which is standard for ViTs and
        worth roughly nothing on a short run but costs nothing to get right.
        """
        lr_head = lr_head if lr_head is not None else lr_backbone * 50

        def split(module):
            decay, no_decay = [], []
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
            return decay, no_decay

        head = nn.ModuleList([self.neck, self.conv1, self.conv2, self.conv_mu]
                             + ([self.conv_log_var] if self.conv_log_var is not None else [])
                             + ([self.conv_bins, self.bin_width] if self.bins else [])
                             + ([self.conv_bins_bg, self.conv_htc] if self.htc else []))
        bb_d, bb_n = split(self.backbone)
        hd_d, hd_n = split(head)
        groups = [
            {"params": bb_d, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": bb_n, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_nodecay"},
            {"params": hd_d, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
            {"params": hd_n, "lr": lr_head, "weight_decay": 0.0, "name": "head_nodecay"},
        ]
        return [g for g in groups if g["params"]]

    def enable_gradient_checkpointing(self):
        """Trade compute for memory. Lets the 12 GB card hold a larger batch."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        elif hasattr(self.backbone, "gradient_checkpointing"):
            self.backbone.gradient_checkpointing = True

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train


    def config(self) -> dict:
        """Everything from_checkpoint needs to rebuild this model.

        Kept in one place because three separate tools reconstruct a model from a
        checkpoint and each used to hardcode the config keys it knew about. Adding the
        binned head would have broken all three silently.
        """
        return {"model_id": self.model_id, "height_scale": self.height_scale,
                "predict_uncertainty": self.conv_log_var is not None,
                "bins": self.bins, "bin_min": self.bin_min, "bin_max": self.bin_max,
                "htc": self.htc}


def build(model_id: str = DEFAULT_MODEL, **kw) -> DepthWizard:
    return DepthWizard(model_id=model_id, **kw)


def from_checkpoint(ck: dict, **overrides) -> DepthWizard:
    """Rebuild the model a checkpoint was saved from and load its weights.

    Checkpoints written before config() existed carry only model_id and height_scale at
    the top level. Those default to the direct-regression head, which is what they are.
    """
    cfg = dict(ck.get("model_config") or {})
    cfg.setdefault("model_id", ck.get("model_id") or DEFAULT_MODEL)
    cfg.setdefault("height_scale", ck.get("height_scale", 30.0))
    # Any checkpoint written before the head-tail cut existed was trained without it, so
    # this has to default to False rather than to the constructor default of True.
    cfg.setdefault("htc", False)
    cfg.update(overrides)
    model = DepthWizard(**cfg)
    if "model" in ck:
        model.load_state_dict(ck["model"])
    return model
