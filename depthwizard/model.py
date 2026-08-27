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

        # -------------------------------------------------------- binned height head
        self.bins = int(bins)
        self.bin_min, self.bin_max = float(bin_min), float(bin_max)
        if self.bins > 0:
            self.conv_bins = nn.Conv2d(self.conv_mu.in_channels, self.bins,
                                       kernel_size=1)
            # Bin widths are a global, per-image quantity, so they come from pooled
            # features rather than per-pixel ones. Mean and std together carry how
            # high the scene is and how spread out it is, which is what should decide
            # where the bins are dense.
            self.bin_width = nn.Sequential(
                nn.Linear(2 * self.conv1.in_channels, 256), nn.GELU(),
                nn.Linear(256, self.bins))
            # Start the head where the data is. Randomly initialised, the bin
            # logits give a near-uniform distribution whose expectation is the
            # MIDDLE of the height range -- about 99 m for [-2, 200] -- against a
            # true median of 0. The whole first epoch would go on climbing back
            # down. So zero the weights and shape the bias into a Gaussian over the
            # (now uniform) bins centred on the ground, the same trick conv_log_var
            # uses above to begin at a plausible constant.
            nn.init.zeros_(self.bin_width[-1].weight)
            nn.init.zeros_(self.bin_width[-1].bias)      # -> uniform widths
            nn.init.zeros_(self.conv_bins.weight)
            step = (self.bin_max - self.bin_min) / self.bins
            c0 = self.bin_min + (torch.arange(self.bins) + 0.5) * step
            tau = max(self.height_scale / 4.0, 1.0)
            self.conv_bins.bias.data.copy_(-0.5 * (c0 / tau) ** 2)
        else:
            self.conv_bins = None
            self.bin_width = None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        del base

    # ------------------------------------------------------------------ forward

    def forward(self, pixel_values: torch.Tensor, return_bins: bool = False):
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
            mu, probs, centres = self._binned_height(feat, x)
        else:
            mu = self.conv_mu(x) * self.height_scale
            probs = centres = None

        if self.conv_log_var is None:
            log_var = None
        else:
            # Variance is predicted in normalised units, so converting to metres is a
            # shift of 2*log(scale) in log space. Doing it here rather than in the loss
            # keeps the loss unit-agnostic and the reported sigma in metres.
            log_var = self.conv_log_var(x) + 2.0 * math.log(self.height_scale)

        if return_bins:
            return mu, log_var, probs, centres
        return mu, log_var

    def _binned_height(self, feat, x):
        """Adaptive-bin classification head with soft-argmax.

        One regressed scalar has to span a long-tailed height distribution and
        systematically loses the tail. Predicting a distribution over bins and taking its
        expectation keeps a continuous output while letting the network put its
        resolution where the scene heights actually are.

        Widths are normalised the AdaBins way: a softmax, so they are strictly positive
        and always sum to the full height range.

        Note the expectation is the MEAN of the distribution, which sits between the
        modes wherever the truth is bimodal (a roof edge is roof or ground, never the
        average). Nothing here prevents that. BinDistributionLoss supervises the shape.
        See docs/literature.md section 5.
        """
        g = torch.cat([feat.mean(dim=(2, 3)), feat.std(dim=(2, 3))], dim=1)
        widths = F.softmax(self.bin_width(g), dim=1) * (self.bin_max - self.bin_min)
        edges = self.bin_min + torch.cumsum(widths, dim=1)      # right edge of each bin
        centres = edges - 0.5 * widths                          # (B, N), metres
        probs = self.conv_bins(x).softmax(dim=1)                # (B, N, H, W)
        mu = (probs * centres[:, :, None, None]).sum(dim=1, keepdim=True)
        return mu, probs, centres


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
                             + ([self.conv_bins, self.bin_width] if self.bins else []))
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
                "bins": self.bins, "bin_min": self.bin_min, "bin_max": self.bin_max}


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
    cfg.update(overrides)
    model = DepthWizard(**cfg)
    if "model" in ck:
        model.load_state_dict(ck["model"])
    return model
