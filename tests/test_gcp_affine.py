"""Tests for scale-aware ground control point anchoring.

`fit_gcp_affine` exists because the model's heights are compressed -- measured over 3,090
region-disjoint validation buildings as `ours = 0.483 * truth + 2.49 m` -- and an offset
cannot correct a slope. The PS names this as its own milestone, so the arithmetic had
better be right and the guards had better hold.

Two kinds of check:

  1. Recovery -- given a surface compressed by a KNOWN slope and intercept, the fit must
                 invert it exactly. If it cannot do that on clean data it will certainly
                 not help on real data.
  2. Guards   -- the thresholds are not taste. Measured in tools/gcp_calibration_probe.py,
                 2 control points made 10 of 13 tall tiles WORSE (+17.7% RMSE) while 5
                 well-spread points helped (-21.3%). So too few points, or points that do
                 not span a height range, must REFUSE rather than return a bad fit.

Run:  python -m tests.test_gcp_affine
"""
from __future__ import annotations

import numpy as np
from affine import Affine

import sys
from pathlib import Path

# Same idiom as tools/evaluate.py and tools/gsd_probe.py: make the repo importable
# when this file is run directly. Without it `python tests/test_losses.py` from a
# fresh clone dies on ModuleNotFoundError before a single assertion runs, which is
# the first thing someone checking out the source tries.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.dem import fit_gcp_affine, fit_gcp_power

# 1 m pixels, north-up, origin at the top-left. Sampling uses +0.3 of a pixel rather than
# +0.5 on purpose: Python's round() is banker's rounding, so a .5 tie lands on the even
# neighbour and the test would read the wrong pixel. That bit the first version of this.
TR = Affine(1.0, 0, 0.0, 0, -1.0, 100.0)
HEIGHTS = [2.0, 4.0, 6.0, 10.0, 25.0, 40.0, 60.0]
SLOPE, INTERCEPT = 0.483, 2.49          # the measured compression


def _scene(heights=HEIGHTS, block=9):
    """A surface holding `heights` on square footprints, compressed as the model compresses.

    Footprints, not single pixels. `fit_gcp_affine` samples the median of a 5x5 window
    because a control point marks a structure and one pixel of a noisy height field is a
    poor estimate of it. A single-pixel fixture would read the background through that
    window and refuse every fit -- which is exactly what happened when this test was first
    written against 1 px buildings, and is a property of the fixture, not of the code.
    """
    truth = np.zeros((100, 100), np.float32)
    cells = [(8 + i * 12, 20) for i in range(len(heights))]
    h_ = block // 2
    for (r, c), h in zip(cells, heights):
        truth[r - h_:r + h_ + 1, c - h_:c + h_ + 1] = h
    pred = (SLOPE * truth + INTERCEPT).astype(np.float32)
    pts = [[c + 0.3, 100.0 - r - 0.3, float(h)] for (r, c), h in zip(cells, heights)]
    return truth, pred, pts, cells


def test_recovers_known_compression():
    truth, pred, pts, cells = _scene()
    got = fit_gcp_affine(pred, TR, pts, verbose=False)
    assert got is not None, "a clean 7-point fit must not be refused"
    scale, offset, n, rms = got
    assert n == 7, n
    # Inverting y = a*x + b is x = (y - b)/a, i.e. scale 1/a and offset -b/a.
    assert abs(scale - 1.0 / SLOPE) < 1e-3, scale
    assert abs(offset - (-INTERCEPT / SLOPE)) < 1e-2, offset
    assert rms < 1e-3, rms
    # And applying it must actually undo the compression, not merely report good numbers.
    corrected = pred * scale + offset
    m = truth > 0
    assert np.abs(corrected[m] - truth[m]).max() < 1e-3
    print("  recovery: scale %.4f offset %+.4f, residual RMS %.2e m — OK" % (scale, offset, rms))


def test_refuses_too_few_points():
    _, pred, pts, _ = _scene()
    assert fit_gcp_affine(pred, TR, pts[:2], verbose=False) is None
    assert fit_gcp_affine(pred, TR, pts[:4], verbose=False) is None
    # Five points ALONE are no longer enough: they must also constrain the tall end.
    # HEIGHTS[:5] is 2,4,6,10,25 -- only one lands in the upper half of its own span.
    assert fit_gcp_affine(pred, TR, pts[:5], verbose=False) is None
    _, pred2, pts2, _ = _scene(heights=[2.0, 4.0, 6.0, 40.0, 60.0])
    assert fit_gcp_affine(pred2, TR, pts2, verbose=False) is not None
    print("  guard: refuses <5 points; 5 accepted only if they reach the tall end — OK")


def test_refuses_when_only_short_structures_constrain_the_fit():
    """The failure this guard exists for, reproduced.

    Measured end to end on OMA_288_012: 7 control points at 3.6-30.7 m, six below 11 m,
    produced scale 0.830 -- compressing further -- and took the tile from RMSE 23.62 to
    25.09 m. Least squares was dominated by the over-called short end. A fit with no
    support at height must refuse.
    """
    _, pred, pts, _ = _scene(heights=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 40.0])
    assert fit_gcp_affine(pred, TR, pts, verbose=False) is None
    print("  guard: refuses a fit supported only by short structures — OK")


def test_refuses_scale_below_one():
    """A corrective scale below 1 would compress further, and our error is under-call."""
    truth = np.zeros((100, 100), np.float32)
    cells = [(8 + i * 12, 20) for i in range(6)]
    for (r, c), h in zip(cells, [10.0, 12.0, 14.0, 30.0, 34.0, 38.0]):
        truth[r - 4:r + 5, c - 4:c + 5] = h
    # Predictions deliberately ABOVE truth and steeper, so the fitted scale comes out < 1.
    pred = (truth * 1.6).astype(np.float32)
    pts = [[c + 0.3, 100.0 - r - 0.3, float(truth[r, c])] for r, c in cells]
    assert fit_gcp_affine(pred, TR, pts, verbose=False) is None
    print("  guard: refuses a scale below 1.0 — OK")


def test_refuses_degenerate_span():
    """Five points at the same height define no slope, and fitting one is harmful."""
    truth, pred, pts, _ = _scene(heights=[4.0] * 5)  # five identical footprints
    assert fit_gcp_affine(pred, TR, pts, verbose=False) is None
    print("  guard: refuses points spanning no height range — OK")


def test_refuses_implausible_scale():
    """Control points that disagree must not silently produce a wild scale."""
    _, pred, pts, _ = _scene()
    # Same predictions, elevations scrambled to nonsense far outside 0.2-5.0.
    bad = [[x, y, z * 400.0] for x, y, z in pts]
    assert fit_gcp_affine(pred, TR, bad, verbose=False) is None
    print("  guard: refuses a scale outside the plausible range — OK")


def test_noise_does_not_break_the_fit():
    """A little measurement noise should degrade the fit, not invert or reject it."""
    rng = np.random.default_rng(1337)
    truth, pred, pts, _ = _scene()
    noisy = [[x, y, z + float(rng.normal(0, 0.5))] for x, y, z in pts]
    got = fit_gcp_affine(pred, TR, noisy, verbose=False)
    assert got is not None
    scale, offset, n, rms = got
    assert abs(scale - 1.0 / SLOPE) < 0.25, scale     # near, not exact
    print("  robustness: scale %.4f under 0.5 m point noise — OK" % scale)


def test_power_recovers_a_known_ratio_error():
    """The form that ships. A ratio error is what the model actually has."""
    truth = np.zeros((100, 100), np.float32)
    cells = [(8 + i * 12, 20) for i in range(7)]
    hs = [2.0, 4.0, 6.0, 10.0, 25.0, 40.0, 60.0]
    A, B = 1.9, 0.72                       # pred = A * truth**B: over-call short, under tall
    for (r, c), h in zip(cells, hs):
        truth[r - 4:r + 5, c - 4:c + 5] = h
    pred = (A * np.maximum(truth, 0.05) ** B).astype(np.float32)
    pts = [[c + 0.3, 100.0 - r - 0.3, float(h)] for (r, c), h in zip(cells, hs)]
    got = fit_gcp_power(pred, TR, pts, verbose=False)
    assert got is not None, "a clean 7-point power fit must not be refused"
    a, b, n, rms = got
    # Inverting y = A*x**B is x = (1/A)**(1/B) * y**(1/B).
    assert abs(b - 1.0 / B) < 1e-2, b
    assert rms < 1e-2, rms
    corrected = a * np.maximum(pred, 0.05) ** b
    m = truth > 0
    assert np.abs(corrected[m] - truth[m]).max() < 0.05
    print("  power: recovered exponent %.4f (true %.4f), residual RMS %.2e m — OK"
          % (b, 1.0 / B, rms))


def test_power_refuses_the_same_bad_inputs():
    """Guards must not be weaker than the affine path's."""
    _, pred, pts, _ = _scene()
    assert fit_gcp_power(pred, TR, pts[:3], verbose=False) is None
    _, pred2, pts2, _ = _scene(heights=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 40.0])
    assert fit_gcp_power(pred2, TR, pts2, verbose=False) is None
    print("  power: refuses too-few and short-only control points — OK")


if __name__ == "__main__":
    for fn in (test_recovers_known_compression, test_refuses_too_few_points,
               test_refuses_when_only_short_structures_constrain_the_fit,
               test_refuses_scale_below_one,
               test_refuses_degenerate_span, test_refuses_implausible_scale,
               test_noise_does_not_break_the_fit,
               test_power_recovers_a_known_ratio_error,
               test_power_refuses_the_same_bad_inputs):
        fn()
    print("all gcp affine tests passed")
