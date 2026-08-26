"""Export a checkpoint to ONNX, and verify the export actually agrees with PyTorch.

    python tools/export_onnx.py --ckpt checkpoints/run01/best.pt --out out/depthwizard.onnx
    python tools/export_onnx.py --ckpt ... --out ... --quantize   # int8 dynamic, CPU

Differentiator 6.4. The problem statement asks for a *standalone deployment*, and ISRO
scores optimisation. "It runs on my machine with a 12 GB GPU and a conda environment" is
not a deployment. One file, no Python, no CUDA, is.

Verification is the point of this script
----------------------------------------
Exporting is three lines. Exporting something that silently computes a slightly different
function is the actual risk -- opset differences around `interpolate` with
`align_corners=True` are a classic source of quiet drift, and DPT heads are full of
interpolation. So this script always re-runs the ONNX graph against PyTorch on real-shaped
input and reports the maximum divergence **in metres**, which is the unit that matters.
An export that disagrees by centimetres is fine; one that disagrees by metres is a bug,
and it would be invisible without checking.

Dynamic axes cover height and width so one file serves any tile size that is a multiple
of 14, rather than baking in 518.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.model import build, PATCH  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Emit metres and sigma directly.

    The training model returns log-variance, which is the right parameterisation for the
    loss and the wrong one for a consumer. Anything downstream wants sigma in metres, so
    the conversion belongs inside the exported graph -- otherwise every caller has to
    remember to do `exp(0.5 * x)` and one of them eventually will not.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        mu, log_var = self.model(pixel_values)
        if log_var is None:
            return mu, torch.zeros_like(mu)
        return mu, torch.exp(0.5 * log_var.clamp(-20.0, 20.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="out/depthwizard.onnx")
    ap.add_argument("--size", type=int, default=518, help="tracing size; must be a multiple of 14")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize", action="store_true", help="also write an int8 dynamic-quantised copy")
    ap.add_argument("--tolerance", type=float, default=0.05, help="max allowed divergence, metres")
    args = ap.parse_args()

    if args.size % PATCH:
        raise SystemExit(f"--size must be a multiple of {PATCH}; {args.size} is not")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build(model_id=ck.get("model_id"), height_scale=ck.get("height_scale", 30.0))
    model.load_state_dict(ck["model"])
    model.eval()
    wrapper = ExportWrapper(model).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size)

    print(f"exporting {args.ckpt} (epoch {ck.get('epoch')}, height scale "
          f"{ck.get('height_scale', 30.0):.1f} m) at {args.size}x{args.size}, opset {args.opset}")

    torch.onnx.export(
        wrapper, (dummy,), str(out),
        input_names=["image"], output_names=["height_m", "sigma_m"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "height_m": {0: "batch", 2: "height", 3: "width"},
                      "sigma_m": {0: "batch", 2: "height", 3: "width"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    mb = out.stat().st_size / 1e6
    print(f"  wrote {out}  ({mb:.1f} MB)")

    # ------------------------------------------------------------------ verify
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n  ! onnxruntime not installed -- export NOT verified.")
        print("    pip install onnxruntime   then re-run. Do not ship an unverified export.")
        return

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        th_mu, th_sg = wrapper(dummy)
    on_mu, on_sg = sess.run(None, {"image": dummy.numpy()})

    d_mu = float(np.abs(th_mu.numpy() - on_mu).max())
    d_sg = float(np.abs(th_sg.numpy() - on_sg).max())
    print(f"\n  agreement with PyTorch (max abs difference over {dummy.numel():,} inputs):")
    print(f"    height  {d_mu*100:.4f} cm")
    print(f"    sigma   {d_sg*100:.4f} cm")
    ok = max(d_mu, d_sg) <= args.tolerance
    print(f"    -> {'PASS' if ok else 'FAIL'} against a {args.tolerance*100:.0f} cm tolerance")

    # A different input size proves the dynamic axes are real rather than nominal.
    alt = args.size + PATCH * 2
    try:
        alt_out = sess.run(None, {"image": torch.randn(1, 3, alt, alt).numpy()})
        print(f"    dynamic shape {alt}x{alt}: OK, returned {alt_out[0].shape}")
    except Exception as e:
        print(f"    dynamic shape {alt}x{alt}: FAILED -- {str(e)[:120]}")

    t0 = time.time()
    n = 3
    for _ in range(n):
        sess.run(None, {"image": dummy.numpy()})
    cpu_s = (time.time() - t0) / n
    px = args.size * args.size
    print(f"\n  CPU inference: {cpu_s*1000:.0f} ms per {args.size}x{args.size} tile "
          f"({px/cpu_s/1e6:.2f} Mpx/s, single ONNX Runtime session)")

    if args.quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("\n  ! quantization tools unavailable; skipping int8")
            return
        q = out.with_name(out.stem + ".int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
        qmb = q.stat().st_size / 1e6
        qs = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
        q_mu, q_sg = qs.run(None, {"image": dummy.numpy()})
        qd = float(np.abs(th_mu.numpy() - q_mu).max())
        print(f"\n  int8 -> {q}  ({qmb:.1f} MB, {100*(1-qmb/mb):.0f}% smaller)")
        print(f"    height divergence vs PyTorch: {qd:.3f} m")
        print("    Quantisation is lossy by construction. Report this number honestly in")
        print("    the submission -- an int8 model that is 0.3 m worse is a real trade,")
        print("    and pretending otherwise is the kind of claim a jury checks.")


if __name__ == "__main__":
    main()
