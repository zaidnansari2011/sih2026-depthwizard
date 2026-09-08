"""Bake the viewer into a single double-clickable HTML file.

Why this exists: index.html loads as an ES module and fetches its scene binaries, and a
browser blocks both under file://. A judge who unzips the submission and double-clicks
index.html gets an empty page -- against a criterion that says "successful standalone
deployment" in as many words. run_viewer.bat solves it for anyone with Python installed;
this removes the Python dependency entirely.

The trick is that nothing in main.js changes. Three things get rewritten on the way in:

  * three.module.js is minified and ends in one `export{a as Vector3,...}` statement.
    Rewriting that to `const THREE={Vector3:a,...}` turns the module into plain inline
    script, so main.js's import line can simply be dropped.
  * fetch() is shimmed to serve from an embedded asset table. Because the shim controls
    the bytes main.js receives, the heights can be stored quantised as uint16 and
    dequantised on the way out -- halving the file for a precision loss of a few
    millimetres against a model whose error is metres.
  * Textures go through THREE.DefaultLoadingManager.setURLModifier, which exists for
    exactly this and keeps TextureLoader untouched.

    python tools/build_standalone.py                 # -> viewer_standalone.html
    python tools/build_standalone.py --grid 384      # smaller file, coarser mesh
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def inline_three(src: str) -> str:
    """Turn the minified ES module into a plain script defining a THREE namespace."""
    i = src.rindex("export{")
    j = src.index("}", i)
    body = src[i + len("export{"):j]
    pairs = []
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        if " as " in item:
            local, name = (s.strip() for s in item.split(" as "))
        else:
            local = name = item
        pairs.append(f"{name}:{local}")
    return src[:i] + "const THREE={" + ",".join(pairs) + "};" + src[j + 1:]


def resample(a: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """Stride-sample, matching export_terrain.py.

    Averaging would round off exactly the rooftops the model is judged on, and the four
    arrays in a scene must be sampled identically or they stop being registered.
    """
    ys = np.linspace(0, a.shape[0] - 1, gh).astype(np.int32)
    xs = np.linspace(0, a.shape[1] - 1, gw).astype(np.int32)
    return a[np.ix_(ys, xs)]


def quantise(a: np.ndarray):
    """float32 -> uint16 + [offset, scale]. Precision is (range / 65535) metres."""
    a = np.asarray(a, np.float64)
    good = np.isfinite(a)
    lo = float(a[good].min()) if good.any() else 0.0
    hi = float(a[good].max()) if good.any() else 1.0
    scale = (hi - lo) / 65535.0 or 1e-9
    q = np.clip(np.round((np.where(good, a, lo) - lo) / scale), 0, 65535).astype("<u2")
    return q.tobytes(), [lo, scale]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=512,
                    help="longest edge of the baked mesh grid (default 512)")
    ap.add_argument("--tex", type=int, default=1024, help="baked texture size")
    ap.add_argument("--quality", type=int, default=82, help="baked texture JPEG quality")
    ap.add_argument("--out", default=str(ROOT / "viewer_standalone.html"))
    a = ap.parse_args()

    from PIL import Image

    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    main_js = (VIEWER / "main.js").read_text(encoding="utf-8")
    three = (VIEWER / "vendor" / "three.module.js").read_text(encoding="utf-8")

    main_js = re.sub(r"^import \* as THREE.*$", "", main_js, count=1, flags=re.M)
    three = inline_three(three)

    # Give main.js its own scope. Inlining puts it in the same scope as minified three.js,
    # which declares plenty of short identifiers of its own -- `$` collides immediately and
    # the whole bundle then fails to parse, showing a blank page with nothing in the
    # console but a SyntaxError. Renaming the clash would fix today's collision and leave
    # the next one waiting; a block removes the entire class of them.
    main_js = "(() => {\n" + main_js + "\n})();"

    assets: dict[str, dict] = {}
    urls: dict[str, str] = {}

    cal = VIEWER / "calibration.json"
    if cal.exists():
        assets["calibration.json"] = {"t": "text", "v": cal.read_text(encoding="utf-8")}

    index = json.loads((VIEWER / "scenes" / "index.json").read_text(encoding="utf-8"))
    baked_index = []
    total = 0

    for entry in index:
        d = VIEWER / "scenes" / entry["dir"]
        # Skip entries whose directory is gone. deploy/stage.py already does this via
        # prune_index(), and without the same rule here the hosted site and the standalone
        # file are built from different scene lists -- which is exactly the drift these two
        # artefacts must never have. An upload scene deleted by hand leaves its index entry
        # behind, and this crashed the build with a bare FileNotFoundError.
        if not (d / "manifest.json").exists():
            print(f"  skipping {entry['dir']}: no manifest.json (stale index entry)")
            continue
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        W, H = m["width"], m["height"]
        scale = min(1.0, a.grid / max(W, H))
        gw, gh = max(2, int(round(W * scale))), max(2, int(round(H * scale)))

        for key, fname in list(m["files"].items()):
            p = d / fname
            if not p.exists():
                continue
            path = f"scenes/{entry['dir']}/{fname}"
            if fname.endswith(".jpg"):
                im = Image.open(p).convert("RGB")
                if max(im.size) > a.tex:
                    # Scale by the LONG edge. Fixing width alone leaves a portrait image
                    # taller than the cap, so the budget silently does not hold.
                    k = a.tex / max(im.size)
                    im = im.resize((max(1, int(im.size[0] * k)), max(1, int(im.size[1] * k))),
                                   Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=a.quality, optimize=True)
                urls[path] = "data:image/jpeg;base64," + b64(buf.getvalue())
                total += len(urls[path])
            elif fname.endswith(".json"):
                # Not a raster. buildings.json is a small list of per-building statistics
                # in NORMALISED coordinates, so unlike every band below it survives the
                # mesh downsample untouched -- resampling or quantising it would corrupt it.
                assets[path] = {"t": "text", "v": p.read_text(encoding="utf-8")}
                total += len(assets[path]["v"])
            elif fname.endswith("_valid.bin"):
                arr = resample(np.fromfile(p, "<u1").reshape(H, W), gw, gh)
                assets[path] = {"t": "u8", "v": b64(arr.tobytes())}
                total += len(assets[path]["v"])
            else:
                arr = resample(np.fromfile(p, "<f4").reshape(H, W), gw, gh)
                raw, q = quantise(arr)
                assets[path] = {"t": "f32", "q": q, "v": b64(raw)}
                total += len(assets[path]["v"])

        # The mesh grid shrank, so the manifest must say so or main.js indexes past the
        # end of every array. Ground sample distance grows by the same factor, which keeps
        # the scene extent -- and therefore every measurement -- correct.
        m["width"], m["height"] = gw, gh
        if m.get("gsd_m"):
            m["gsd_m"] = m["gsd_m"] * W / gw
        m["baked_from"] = f"{W}x{H}"
        assets[f"scenes/{entry['dir']}/manifest.json"] = {
            "t": "text", "v": json.dumps(m, ensure_ascii=False)}
        e = dict(entry, width=gw, height=gh)
        baked_index.append(e)
        print(f"  {entry['name']:26s} {W}x{H} -> {gw}x{gh}   {len(m['files'])} files")

    assets["scenes/index.json"] = {"t": "text", "v": json.dumps(baked_index, ensure_ascii=False)}

    shim = """
const __A = %s, __U = %s;
function __d(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
const __key = (u) => String(u).replace(/^\\.\\//,'').replace(/[?#].*$/,'');
self.fetch = async (u) => {
  const a = __A[__key(u)];
  if (!a) return new Response(null, {status: 404, statusText: 'not baked'});
  if (a.t === 'text')
    return new Response(a.v, {status: 200, headers: {'Content-Type': 'application/json'}});
  if (a.t === 'u8') return new Response(__d(a.v).buffer, {status: 200});
  // uint16 -> float32, undoing the quantisation done at build time.
  const q = __d(a.v), n = q.byteLength / 2;
  const s = new Uint16Array(q.buffer, q.byteOffset, n), f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = a.q[0] + s[i] * a.q[1];
  return new Response(f.buffer, {status: 200});
};
THREE.DefaultLoadingManager.setURLModifier((u) => __U[__key(u)] || u);
""" % (json.dumps(assets, separators=(",", ":")), json.dumps(urls, separators=(",", ":")))

    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
    body = re.sub(r'<script type="module".*?</script>', "", body, flags=re.S)

    out = Path(a.out)
    out.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\" />\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "<title>DepthWizard — 3D terrain viewer</title>\n"
        "<!-- Single-file build. Everything below is embedded; no server, no network,\n"
        "     no install. Generated by tools/build_standalone.py. -->\n"
        f"<style>{style}</style>\n</head>\n<body>\n{body}\n"
        f"<script type=\"module\">\n{three}\n{shim}\n{main_js}\n</script>\n</body>\n</html>\n",
        encoding="utf-8")

    mb = out.stat().st_size / 1e6
    print(f"\n{out}  {mb:.1f} MB  ({len(assets)} assets, {len(urls)} textures)")
    if mb > 40:
        print("  WARNING: large enough that browsers will feel it. Try --grid 384.")


if __name__ == "__main__":
    main()
