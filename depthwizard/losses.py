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
            if self.beta > 0:
                inv_var = inv_var * (inv_var.detach() ** -self.beta)
            loss_map = 0.5 * (log_var + sq_err * inv_var)

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


class CompositeLoss(nn.Module):
    """NLL + optional gradient matching. The default training objective."""

    def __init__(self, grad_weight: float = 0.5, **nll_kwargs):
        super().__init__()
        self.nll = GaussianNLLLoss(**nll_kwargs)
        self.grad = GradientMatchingLoss()
        self.grad_weight = grad_weight

    def forward(self, mu, log_var, target, mask=None):
        loss, stats = self.nll(mu, log_var, target, mask)
        if self.grad_weight > 0:
            g = self.grad(mu, target, mask)
            loss = loss + self.grad_weight * g
            stats["grad"] = float(g.detach())
            stats["loss"] = float(loss.detach())
        return loss, stats
