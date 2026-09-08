"""Assemble the Docker build context for the public deployment.

Everything here exists to keep the context small, because it is uploaded from a home
connection to ACR before the image is built. Three things dominate the bytes and each is
handled rather than hoped about:

  * **The checkpoint is stripped.** run02/best.pt is 297 MB, but only 99 MB of that is
    the model. The rest is optimiser and scheduler state, RNG state and the training
    args -- everything needed to *resume* a run and nothing needed to serve one.
    from_checkpoint() reads model, model_id, height_scale and model_config, so those are
    what ship. epoch and val come along because infer.py prints them, and a served result
    that cannot say which checkpoint produced it is not evidence.
  * **The HF cache is baked in.** model.py builds the backbone with
    AutoModelForDepthEstimation.from_pretrained(), which reaches Hugging Face on a cache
    miss. A container that downloads 99 MB on first request is one rate-limit away from a
    failed demo, so the cache ships and HF_HUB_OFFLINE=1 turns a silent download into a
    loud error at build time.
  * **Only the runtime tree is copied.** No papers, no docs, no notebooks, no
    tools/_ortweb (which contains a whole Chrome profile).

    python deploy/stage.py                    # -> deploy/_context
    python deploy/stage.py --out D:/tmp/ctx
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Modules the serving path actually imports. infer.py pulls pick_precision from train.py,
# so train.py is runtime code here whether or not that reads oddly.
CODE = ["infer.py", "train.py"]
PKG = "depthwizard"
TOOLS = ["serve_app.py", "export_terrain.py"]

HF_MODEL = "models--depth-anything--Depth-Anything-V2-Small-hf"


def strip_checkpoint(src: Path, dst: Path) -> None:
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: ck[k] for k in ("model", "model_id", "height_scale", "model_config",
                               "epoch", "val") if k in ck}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, dst)
    print(f"  checkpoint {src.stat().st_size/1e6:6.1f} MB -> "
          f"{dst.stat().st_size/1e6:6.1f} MB   (kept: {', '.join(keep)})")


def prune_index(scenes_root: Path) -> None:
    """Drop index entries for scenes that were not staged.

    index.json is written by export_terrain.py and lists every scene the dev machine ever
    made, including the upload_* ones just excluded. An entry with no directory makes the
    viewer fetch a manifest that is not there, which presents as the scene picker being
    broken rather than as one missing scene.
    """
    index = scenes_root / "index.json"
    if not index.exists():
        return
    known = json.loads(index.read_text(encoding="utf-8"))
    live = [k for k in known if (scenes_root / k.get("dir", "")).is_dir()]
    if len(live) != len(known):
        index.write_text(json.dumps(live, indent=2), encoding="utf-8")
        print(f"  index      dropped {len(known)-len(live)} runtime scene entries")


def copy_tree(src: Path, dst: Path, *, ignore=None) -> int:
    if not src.exists():
        sys.exit(f"missing: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*(ignore or [])))
    return sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deploy" / "_context"))
    ap.add_argument("--ckpt", default=str(ROOT.parent / "checkpoints" / "run02" / "best.pt"))
    ap.add_argument("--hf-cache", default="D:/hf-cache/hub")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    app = out / "app"
    app.mkdir(parents=True)

    print("staging build context")

    for f in CODE:
        shutil.copy2(ROOT / f, app / f)
    total = copy_tree(ROOT / PKG, app / PKG, ignore=["__pycache__", "*.pyc"])
    print(f"  package    {total/1e6:6.1f} MB")

    (app / "tools").mkdir()
    for t in TOOLS:
        shutil.copy2(ROOT / "tools" / t, app / "tools" / t)

    # The viewer and its baked scenes are the site. Scenes are gitignored, so they exist
    # only on this machine -- if they are not copied the deployment serves an empty picker.
    # upload_* are scenes a visitor (or a local test) produced at runtime. They live in
    # the same directory as the baked ones and are indistinguishable to copytree, so
    # without this the image ships whatever happened to be on the dev machine.
    v = copy_tree(ROOT / "viewer", app / "viewer", ignore=["__pycache__", "upload_*"])
    prune_index(app / "viewer" / "scenes")
    n_scenes = len([d for d in (app / "viewer" / "scenes").iterdir() if d.is_dir()]) \
        if (app / "viewer" / "scenes").is_dir() else 0
    print(f"  viewer     {v/1e6:6.1f} MB   ({n_scenes} baked scenes)")
    if n_scenes == 0:
        sys.exit("  !! no baked scenes staged -- regenerate with tools/build_scenes.py")

    strip_checkpoint(Path(args.ckpt), app / "weights" / "best.pt")

    hf_src = Path(args.hf_cache) / HF_MODEL
    if not hf_src.exists():
        sys.exit(f"missing HF cache: {hf_src}")
    # A HF cache stores each file twice: once in blobs/ under its sha, and once in
    # snapshots/<rev>/ as a link to it. copytree dereferences those links, so copying the
    # whole tree ships 99 MB of weights twice. Only the snapshot is resolved at load time,
    # so blobs/ is dropped -- and the Dockerfile builds the model offline at image-build
    # time, which is what proves that claim rather than assuming it.
    h = copy_tree(hf_src, app / "hf" / "hub" / HF_MODEL, ignore=["blobs"])
    print(f"  hf cache   {h/1e6:6.1f} MB   (blobs/ dropped; snapshot only)")

    shutil.copy2(ROOT / "deploy" / "Dockerfile", out / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "requirements-serve.txt", out / "requirements-serve.txt")
    shutil.copy2(ROOT / "deploy" / "dockerignore", out / ".dockerignore")

    grand = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  context    {grand/1e6:6.1f} MB at {out}")


if __name__ == "__main__":
    main()
