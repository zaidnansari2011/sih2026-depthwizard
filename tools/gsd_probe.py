"""Does the model survive the resolution ISRO will actually evaluate at?

We train and infer at DFC2019's 0.3 m. The problem statement says final evaluation uses
"ISRO RGB-band optical satellite imagery", which in practice means Cartosat -- roughly
0.6-1.0 m. Probe 04 established that what governs our detail is the ground area a token
covers, so halving the resolution doubles that footprint and is a live domain gap.

Method: downsample the RGB to the target GSD, run inference, upsample the prediction back
to the truth grid, and score against the SAME LiDAR reference. Truth never moves, so the
only thing changing is what the model was allowed to see.
"""
import argparse, sys, subprocess, json
from pathlib import Path
import numpy as np, rasterio
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.metrics import building_instances, building_wise_metrics

ROOT = Path("D:/sih2026")
TRUTH = ROOT / "data" / "extracted" / "Track1-Truth"
RGB = ROOT / "data" / "extracted" / "Track1-RGB"
TMP = ROOT / "out" / "gsdprobe"

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", nargs="+", required=True)
ap.add_argument("--gsd", type=float, nargs="+", default=[0.3, 0.6, 1.0])
ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/run02/best.pt"))
a = ap.parse_args()
TMP.mkdir(parents=True, exist_ok=True)

acc = {}
for tile in a.tiles:
    agl = rasterio.open(TRUTH / f"{tile}_AGL.tif").read(1).astype("float32")
    cls = rasterio.open(TRUTH / f"{tile}_CLS.tif").read(1)
    H, W = agl.shape
    src = np.array(Image.open(RGB / f"{tile}_RGB.tif").convert("RGB")) \
        if False else rasterio.open(RGB / f"{tile}_RGB.tif").read()[:3].transpose(1, 2, 0)
    for g in a.gsd:
        f = 0.3 / g
        nw, nh = max(64, int(W * f)), max(64, int(H * f))
        # Multiple of 14 keeps the backbone from silently resizing under us.
        nw, nh = nw - nw % 14, nh - nh % 14
        tag = f"g{int(round(g*100)):03d}"   # cm, not metres: a dot in the stem is eaten as a suffix
        p = TMP / f"{tile}_{tag}.png"
        Image.fromarray(src).resize((nw, nh), Image.LANCZOS).save(p)
        out = TMP / f"{tile}_{tag}"
        # The window cannot exceed the image: at 1 m a 1024 px tile is only ~307 px wide,
        # and a 518 px window would be cropped short and blow up the cosine taper.
        win = min(518, (min(nh, nw) // 14) * 14)
        ov = max(14, (min(140, win // 3) // 14) * 14)
        subprocess.run([sys.executable, "infer.py", "--image", str(p), "--ckpt", a.ckpt,
                        "--out", str(out), "--tta", "--tile", str(win),
                        "--overlap", str(ov)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pred = rasterio.open(str(out) + ".height.tif").read(1).astype("float32")
        # Back onto the truth grid so every setting is scored identically.
        pred = np.array(Image.fromarray(pred).resize((W, H), Image.BILINEAR))
        m = np.isfinite(agl); b = (cls == 6) & m; gr = (cls == 2) & m
        o, t = building_instances(pred, agl, cls); bw = building_wise_metrics(o, t)
        acc.setdefault(g, []).append((
            float(np.sqrt(np.mean((pred[m] - agl[m]) ** 2))),
            float(np.sqrt(np.mean((pred[gr] - agl[gr]) ** 2))),
            bw["rmse"], bw["bias"], bw["corr"], bw["n_buildings"]))
        print(f"  {tile} @ {g:.1f} m ({nw}x{nh}): per-building {bw['rmse']:.3f} "
              f"bias {bw['bias']:+.2f} r {bw['corr']:+.3f}")

print(f"\n{'GSD':>6s} {'per-pixel':>10s} {'ground':>8s} {'per-building':>13s} {'bias':>7s} {'corr':>7s}")
base = None
for g, v in sorted(acc.items()):
    r = [float(np.mean([x[i] for x in v])) for i in range(5)]
    d = "" if base is None else f"   [{r[2]-base:+.3f} per-building]"
    if base is None: base = r[2]
    print(f"{g:6.1f} {r[0]:10.3f} {r[1]:8.3f} {r[2]:13.3f} {r[3]:+7.2f} {r[4]:+7.3f}{d}")
print("\n0.3 m is what we train on. Cartosat is nearer 0.6-1.0 m.")
