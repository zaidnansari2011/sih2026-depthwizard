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

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        del base

    # ------------------------------------------------------------------ forward

    def forward(self, pixel_values: torch.Tensor):
        """pixel_values: (B, 3, H, W), ImageNet-normalised. Returns (mu, log_var) in metres."""
        _, _, H, W = pixel_values.shape
        check_input_size(H, W, self.patch_size)
        ph, pw = H // self.patch_size, W // self.patch_size

        out = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=False, output_attentions=False
        )
        hidden = self.neck(out.feature_maps, ph, pw)
        x = hidden[self.head_in_index]

        # Mirrors the stock head: upsample after conv1, exactly as in the DPT paper.
        x = self.conv1(x)
        x = F.interpolate(x, (ph * self.patch_size, pw * self.patch_size),
                          mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))

        mu = self.conv_mu(x) * self.height_scale
        if self.conv_log_var is None:
            return mu, None
        # Variance is predicted in normalised units, so converting to metres is a shift
        # of 2*log(scale) in log space. Doing it here rather than in the loss keeps the
        # loss unit-agnostic and the reported sigma in metres.
        log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)
        return mu, log_var

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
                             + ([self.conv_log_var] if self.conv_log_var is not None else []))
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


def build(model_id: str = DEFAULT_MODEL, **kw) -> DepthWizard:
    return DepthWizard(model_id=model_id, **kw)
