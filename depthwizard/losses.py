"""Losses for height regression with per-pixel uncertainty.

Differentiator 6.1. Single-view height is ill-posed -- there is no unique 3D solution
for a 2D image -- so the model predicts a distribution per pixel, not a point estimate.
We train with Gaussian negative log-likelihood and let the model say where it is unsure.

The network emits two channels: mu (height) and log_var (log predictive variance).
Predicting log-variance rather than variance keeps it positive without a clamp and
keeps gradients well-scaled.

    L = 0.5 * ( log_var + (y - mu)^2 / exp(log_var) )

Read the two terms as a trade: the model can lower the residual term by admitting
uncertainty, but pays log_var for doing so. It cannot buy a free pass by predicting
huge variance everywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """Heteroscedastic Gaussian NLL with an optional validity mask.

    Args:
        beta: detach-weight in the residual term, per Kendall & Gal. 0 = standard NLL.
            Small positive values stabilise very early training, when a randomly
            initialised variance head can dominate the loss.
        log_var_min/max: clamp range. Guards against exp() overflow and against the
            model escaping into "infinitely uncertain everywhere".
        warmup_mse: if set, the first N steps use plain MSE so mu becomes reasonable
            before the variance head starts shaping gradients.
    """

    def __init__(
        self,
        beta: float = 0.0,
        log_var_min: float = -7.0,
        log_var_max: float = 7.0,
        warmup_mse: int = 0,
    ):
        super().__init__()
        self.beta = beta
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.warmup_mse = warmup_mse
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_var = log_var.clamp(self.log_var_min, self.log_var_max)
        sq_err = (target - mu) ** 2

        if self.training:
            self._step += 1
        if self.warmup_mse and int(self._step) <= self.warmup_mse:
            loss_map = sq_err
        else:
            inv_var = torch.exp(-log_var)
            loss_map = 0.5 * (log_var + sq_err * inv_var)
            if self.beta > 0:
                # beta-NLL, Seitzer et al. 2022: weight each pixel's NLL by
                # stopgrad(sigma^2)^beta. The weight covers the WHOLE term, not just
                # the residual. At beta=1 it cancels the 1/sigma^2 so the mean head's
                # gradient stops depending on sigma -- which is the point: plain NLL
                # lets mu abandon noisy regions, because down-weighting them is
                # cheaper than fitting them.
                loss_map = loss_map * torch.exp(log_var.detach() * self.beta)

        if mask is not None:
            n = mask.sum().clamp(min=1.0)
            loss = (loss_map * mask).sum() / n
            rmse = ((sq_err * mask).sum() / n).sqrt()
        else:
            loss = loss_map.mean()
            rmse = sq_err.mean().sqrt()

        stats = {
            "loss": float(loss.detach()),
            "rmse": float(rmse.detach()),
            "sigma_mean": float(torch.exp(0.5 * log_var).mean().detach()),
        }
        return loss, stats


class MaskedL1Loss(nn.Module):
    """Plain masked L1. Baseline comparator -- what a team without 6.1 would train."""

    def forward(self, pred, target, mask=None):
        err = (pred - target).abs()
        if mask is None:
            return err.mean()
        return (err * mask).sum() / mask.sum().clamp(min=1.0)


class GradientMatchingLoss(nn.Module):
    """Penalise disagreement in first differences.

    Height maps are piecewise smooth with hard edges at building footprints. Pure
    per-pixel losses produce blurred roof edges, which look bad in the flythrough
    (differentiator 6.3) even when RMSE is respectable. This sharpens them.
    """

    def forward(self, pred, target, mask=None):
        def diffs(t):
            return t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :]

        px, py = diffs(pred)
        tx, ty = diffs(target)
        lx, ly = (px - tx).abs(), (py - ty).abs()
        if mask is None:
            return lx.mean() + ly.mean()
        mx, my = diffs(mask)
        mx, my = (mx == 0).float() * mask[..., :, 1:], (my == 0).float() * mask[..., 1:, :]
        return ((lx * mx).sum() / mx.sum().clamp(min=1.0)
                + (ly * my).sum() / my.sum().clamp(min=1.0))


class ChamferBinLoss(nn.Module):
    """Bi-directional Chamfer between bin centres and the heights actually present.

    Without it the adaptive widths are free to park anywhere. This pulls them onto the
    quantiles of the scene height distribution, which is the point of making them
    adaptive at all. AdaBins weights this term at 0.1.
    """

    def __init__(self, max_points: int = 4096):
        super().__init__()
        self.max_points = int(max_points)

    def forward(self, centres, target, mask=None):
        total, n = centres.new_zeros(()), 0
        for b in range(centres.shape[0]):
            t = target[b][mask[b] > 0] if mask is not None else target[b].flatten()
            if t.numel() == 0:
                continue
            if t.numel() > self.max_points:
                t = t[torch.randint(0, t.numel(), (self.max_points,), device=t.device)]
            d = (t[:, None] - centres[b][None, :]).abs()
            total = total + d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
            n += 1
        return total / max(n, 1)


class BinDistributionLoss(nn.Module):
    """Supervise the SHAPE of the per-pixel bin distribution, not only its mean.

    Soft-argmax returns the mean. Trained on the mean alone the distribution shape is
    unconstrained, so at a roof edge, where the truth is bimodal (roof or ground), the
    mean lands between the two modes and the edge bleeds. That is the documented
    over-smoothing failure of soft-argmax, and it is exactly where our building error
    lives. Cross-entropy against a Gaussian centred on the truth forces the mass onto one
    mode. See docs/literature.md section 5.

    Scored on a random subsample of valid pixels: a full (B, N, H, W) reference would be
    another half-gigabyte on a card that is already tight, and a few thousand pixels
    estimate this term perfectly well.
    """

    def __init__(self, sigma_bins: float = 1.0, n_points: int = 8192):
        super().__init__()
        self.sigma_bins = float(sigma_bins)
        self.n_points = int(n_points)

    def forward(self, probs, centres, target, mask=None):
        B, N = centres.shape
        p = probs.flatten(2)                                  # (B, N, P)
        t = target.flatten(2).squeeze(1)                      # (B, P)
        m = (mask.flatten(2).squeeze(1) > 0) if mask is not None else None
        total, n = p.new_zeros(()), 0
        for b in range(B):
            idx = (m[b].nonzero(as_tuple=False).squeeze(1) if m is not None
                   else torch.arange(t.shape[1], device=t.device))
            if idx.numel() == 0:
                continue
            if idx.numel() > self.n_points:
                idx = idx[torch.randint(0, idx.numel(), (self.n_points,),
                                        device=idx.device)]
            # Reference width tracks the average bin spacing, so the target stays about as
            # sharp as the bins can actually represent.
            width = ((centres[b, -1] - centres[b, 0]).abs() / N).clamp(min=1e-3)
            ref = torch.exp(-0.5 * (((t[b][idx][None, :] - centres[b][:, None])
                                     / (self.sigma_bins * width)) ** 2))
            ref = ref / ref.sum(dim=0, keepdim=True).clamp(min=1e-12)
            total = total - (ref * p[b][:, idx].clamp(min=1e-12).log()).sum(dim=0).mean()
            n += 1
        return total / max(n, 1)


class CompositeLoss(nn.Module):
    """NLL + optional gradient matching. The default training objective."""

    def __init__(self, grad_weight: float = 0.5, chamfer_weight: float = 0.0,
                 dist_weight: float = 0.0, sigma_bins: float = 1.0, **nll_kwargs):
        super().__init__()
        self.nll = GaussianNLLLoss(**nll_kwargs)
        self.grad = GradientMatchingLoss()
        self.chamfer = ChamferBinLoss()
        self.dist = BinDistributionLoss(sigma_bins=sigma_bins)
        self.grad_weight = grad_weight
        self.chamfer_weight = chamfer_weight
        self.dist_weight = dist_weight

    def forward(self, mu, log_var, target, mask=None, probs=None, centres=None):
        loss, stats = self.nll(mu, log_var, target, mask)
        if self.grad_weight > 0:
            g = self.grad(mu, target, mask)
            loss = loss + self.grad_weight * g
            stats["grad"] = float(g.detach())
        if self.chamfer_weight > 0 and centres is not None:
            c = self.chamfer(centres, target, mask)
            loss = loss + self.chamfer_weight * c
            stats["chamfer"] = float(c.detach())
        if self.dist_weight > 0 and probs is not None:
            d = self.dist(probs, centres, target, mask)
            loss = loss + self.dist_weight * d
            stats["dist"] = float(d.detach())
        # Outside the grad_weight branch. With --grad-weight 0 the logged loss used to
        # be whatever the NLL alone reported, which went stale the moment any other
        # term existed.
        stats["loss"] = float(loss.detach())
        return loss, stats
