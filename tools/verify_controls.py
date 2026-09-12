"""Do the mouse controls actually move the camera? Compare what the page renders.

    python tools/verify_controls.py
    python tools/verify_controls.py --keep   # leave the PNGs behind to look at

Why this is done with pixels
----------------------------
Nothing in the DOM reflects the camera. There is no readout of position, bearing or
distance to assert against, and adding a global just so a test can reach inside the module
would be shipping a test hook in the product. What the camera *does* is visible in exactly
one place -- the rendered image -- so that is what gets compared.

Each gesture is dispatched as a real event on the real canvas, on its own load of the page,
and the result is screenshotted. A gesture that works moves the surface and leaves the
panels alone, so the check is two-sided and both sides matter:

  * the canvas region must change, or the control did nothing;
  * the panel regions must NOT change, or the control disturbed the interface -- which is
    how a wheel handler bound to the window instead of the canvas shows up, scrolling the
    HUD out from under the pointer while it zooms.

A pure "did the image change" test would pass on a control that moved the camera the wrong
way, so --keep writes the frames out to be looked at. They were, on 12 Sep 2026.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]
W, H = 1366, 768

# Gestures, as JS run against the canvas 2.5 s after load. Each is what the browser itself
# would send: pointerdown, a few pointermoves, pointerup -- not a call into the viewer.
DRAG = """
  var r = c.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  function pe(type, px, py, button, buttons) {
    c.dispatchEvent(new PointerEvent(type, {
      clientX: px, clientY: py, button: button, buttons: buttons, bubbles: true,
      cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true
    }));
  }
  pe('pointerdown', x, y, __BTN__, __BTNS__);
  for (var i = 1; i <= 10; i++) {
    pe('pointermove', x + i * __DX__, y + i * __DY__, __BTN__, __BTNS__);
  }
  pe('pointerup', x + 10 * __DX__, y + 10 * __DY__, __BTN__, __BTNS__);
"""


def drag(button: int, buttons: int, dx: int, dy: int) -> str:
    """Fill in the drag template. The placeholders are delimited on purpose: a plain
    BUTTON/BUTTONS pair silently corrupts, because the first is a prefix of the second."""
    out = DRAG
    for k, v in (("__BTN__", button), ("__BTNS__", buttons), ("__DX__", dx), ("__DY__", dy)):
        assert k in out, k
        out = out.replace(k, str(v))
    assert "__" not in out, f"a placeholder survived: {out}"
    return out


GESTURES = {
    "baseline": "",
    "wheel-out": """
  for (var i = 0; i < 6; i++) {
    c.dispatchEvent(new WheelEvent('wheel', { deltaY: 120, bubbles: true, cancelable: true }));
  }""",
    "wheel-in": """
  for (var i = 0; i < 6; i++) {
    c.dispatchEvent(new WheelEvent('wheel', { deltaY: -120, bubbles: true, cancelable: true }));
  }""",
    "drag-orbit": drag(0, 1, 14, 0),
    "right-drag-pan": drag(2, 2, 10, 6),
}

# Two taps with the measure tool on. This exists because "a drag is not a click" is exactly
# the sort of rule that quietly breaks the thing clicks were already for: pick() runs from the
# click handler, behind the same travel threshold that stops an orbit from dropping a pin.
MEASURE = """
  var r = c.getBoundingClientRect();
  document.getElementById('measure').click();
  function tap(px, py) {
    ['pointerdown', 'pointerup'].forEach(function (t) {
      c.dispatchEvent(new PointerEvent(t, {
        clientX: px, clientY: py, button: 0, buttons: t === 'pointerdown' ? 1 : 0,
        bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true
      }));
    });
    c.dispatchEvent(new MouseEvent('click', {
      clientX: px, clientY: py, button: 0, bubbles: true, cancelable: true
    }));
  }
  tap(r.left + r.width * 0.42, r.top + r.height * 0.55);
  tap(r.left + r.width * 0.58, r.top + r.height * 0.62);
"""

PAGE_PROBE = """
<script>
(function () {
  addEventListener('load', function () {
    setTimeout(function () {
      var c = document.querySelector('canvas');
      if (!c) return;
      __GESTURE__
    }, 2500);
  });
})();
</script>
"""


def find_browser() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("no Chrome or Edge found; edit CHROME_CANDIDATES")


def shoot(browser: str, src: Path, gesture: str, out: Path, tmp: Path) -> str:
    """Load the page, run one gesture, screenshot it, and hand back the resulting DOM."""
    html = src.read_text(encoding="utf-8")
    i = html.rfind("</body>")
    page = tmp / f"{out.stem}.html"
    page.write_text(html[:i] + PAGE_PROBE.replace("__GESTURE__", gesture) + html[i:],
                    encoding="utf-8")
    r = subprocess.run([browser, "--headless=new", "--allow-file-access-from-files",
                        f"--user-data-dir={tmp / ('p-' + out.stem)}", f"--window-size={W},{H}",
                        "--virtual-time-budget=9000", f"--screenshot={out}", "--dump-dom",
                        page.resolve().as_uri()], capture_output=True, timeout=300)
    if not out.exists():
        raise SystemExit(f"no screenshot for {out.stem}")
    return r.stdout.decode("utf-8", "replace")


def regions(img: np.ndarray) -> dict[str, np.ndarray]:
    """Split a frame into the surface and the interiors of the two panels.

    The panel windows are the panels' own tops, not full-height columns beside them: the
    panels are opaque but short, and the terrain shows through below each one. A full-height
    strip therefore reports every camera move as the panel moving, which it did on the first
    run of this check.
    """
    _, w = img.shape[:2]
    return {
        # Clear of both columns and above the key hints: nothing here but the surface.
        "canvas": img[60:640, 320:w - 300],
        # Inside the HUD (274 px wide at a 14 px inset) and the stats panel (246 px), down
        # to a depth both are taller than in every scene.
        "hud": img[20:240, 20:280],
        "stats": img[20:240, w - 260:w - 30],
    }


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference, 0-255."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", default=str(ROOT / "viewer_standalone.html"))
    ap.add_argument("--keep", action="store_true", help="write the PNGs somewhere lasting")
    ap.add_argument("--out", default=None, help="where --keep puts them")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    src = Path(args.page)
    if not src.exists():
        raise SystemExit(f"{src} not found -- run tools/build_standalone.py first")
    browser = find_browser()
    tmp = Path(tempfile.mkdtemp(prefix="dwz-ctl-"))
    keep = Path(args.out) if args.out else (ROOT / "out_controls")
    if args.keep:
        keep.mkdir(exist_ok=True)

    print(f"\ncontrol check  {src}")
    print(f"  {len(GESTURES)} loads at {W}x{H}; each needs a real GPU and ~15s\n")
    try:
        frames: dict[str, np.ndarray] = {}
        for name, js in GESTURES.items():
            png = (keep if args.keep else tmp) / f"{name}.png"
            shoot(browser, src, js, png, tmp)
            frames[name] = np.asarray(Image.open(png).convert("RGB"))
            if args.keep:
                print(f"  wrote {png}")
        measure_dom = shoot(browser, src, MEASURE,
                            (keep if args.keep else tmp) / "measure.png", tmp)

        base = regions(frames["baseline"])
        # A gesture has to move the surface by more than the renderer's own frame-to-frame
        # wobble. Anti-aliasing and the FPS counter alone come in under 0.2.
        MOVED, STILL = 1.0, 0.35
        ok = True
        for name in GESTURES:
            if name == "baseline":
                continue
            got = regions(frames[name])
            d = {k: diff(base[k], got[k]) for k in base}
            surface_moved = d["canvas"] >= MOVED
            panels_still = max(d["hud"], d["stats"]) <= STILL
            good = surface_moved and panels_still
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  {name:<15} "
                  f"surface {d['canvas']:6.2f} ({'moved' if surface_moved else 'DID NOT MOVE'})"
                  f"   panels {d['hud']:.2f} / {d['stats']:.2f}"
                  f" ({'still' if panels_still else 'DISTURBED'})")

        # The measure tool reports into the DOM, so this one is read rather than seen.
        m = re.search(r'id="m-ground"[^>]*>(.*?)<', measure_dom, re.S)
        got = (m.group(1).strip() if m else "")
        measured = bool(got) and "—" not in got and got not in ("-", "")
        ok &= measured
        print(f"  {'PASS' if measured else 'FAIL'}  {'measure two taps':<15} "
              f"distance on ground reads {got!r}"
              f"{'' if measured else '  -- a tap no longer places a point'}")

        print(f"\n{'CONTROLS OK' if ok else 'CONTROLS BROKEN'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
