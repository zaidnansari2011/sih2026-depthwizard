"""Screenshot the real viewer, so the deck can show the half of the marks that is UX.

Half of SIH's score for this problem statement is rendering quality and interface, and a
deck that only quotes RMSE argues just the other half. These are captures of the actual
running viewer, not mock-ups.

The viewer has no URL parameters, so a throwaway copy gets a small shim appended that
reads ?scene=&mode= and drives the real controls. Nothing in viewer/ is modified.

Headless Chrome needs software WebGL (SwiftShader); without it the canvas exists but
renders nothing and the shots come back as empty sky, which is a silent failure worth
guarding against -- so each capture is checked for actual pixel variance afterwards.

    python tools/ppt/capture_viewer.py
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEWER = Path("D:/sih2026/depthwizard/viewer")
STAGE = Path("D:/sih2026/depthwizard/tools/ppt/_shotstage")
OUT = Path("D:/sih2026/depthwizard/tools/ppt/shots")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W, H = 1600, 1000

# scene index, view mode, output name, and camera framing.
#
# resetView() parks the camera at (0, 0.45e, 0.75e) looking slightly down, which is a
# safe default for exploring and a poor one for a still: the terrain sits edge-on with a
# third of the frame empty. Framing here is (azimuth deg, height/extent, distance/extent)
# and is converted to a position plus yaw/pitch aimed at the scene centre.
SHOTS = [
    # framing None keeps the viewer's own resetView(), which frames terrain well: it aims
    # slightly above centre so ridgelines sit against the sky. Overriding it for the
    # Sikkim valley produced either a flat top-down texture or a view from under the
    # ridge. The urban scene is the opposite case -- 34 m buildings in a 307 m tile are
    # lost at the default distance, so that one is framed explicitly.
    (2, "texture", "sikkim_valley", None),                # Hilly - Sikkim valley
    (0, "height", "urban_height", (-32, 0.30, 0.46)),     # Omaha, height colouring
    (0, "sigma", "urban_sigma", (-32, 0.30, 0.46)),       # the confidence layer
    (5, "texture", "sikkim_town", None),                  # spare
]

CAM_HOOK = """

// Appended for screenshot capture only. viewer/main.js is not modified.
window.__frame = (azDeg, hFrac, dFrac) => {
  const m = state.manifest;
  if (!m) return false;
  const e = Math.max(m.width, m.height) * (state.gsd || 1);
  const a = azDeg * Math.PI / 180;
  camera.position.set(Math.sin(a) * dFrac * e, hFrac * e, Math.cos(a) * dFrac * e);
  yaw = a;
  pitch = -Math.atan2(hFrac, dFrac);
  camera.updateProjectionMatrix();
  return true;
};
"""

SHIM = """
<script>
// Appended for screenshot capture only; viewer/ itself is untouched.
(function () {
  const q = new URLSearchParams(location.search);
  const want = parseInt(q.get('scene') || '0', 10);
  const mode = q.get('mode') || '';
  const fire = (el) => el.dispatchEvent(new Event('change', { bubbles: true }));
  let tries = 0;
  const t = setInterval(() => {
    const sc = document.getElementById('scene');
    if (!sc || sc.options.length === 0) { if (++tries > 400) clearInterval(t); return; }
    clearInterval(t);
    if (want < sc.options.length) { sc.selectedIndex = want; fire(sc); }
    if (mode) {
      setTimeout(() => {
        const m = document.getElementById('mode');
        if (!m) return;
        for (let i = 0; i < m.options.length; i++) {
          if (m.options[i].value === mode || m.options[i].text.toLowerCase().includes(mode)) {
            m.selectedIndex = i; fire(m); break;
          }
        }
      }, 2500);
    }
    if (q.has('az')) {
      const az = parseFloat(q.get('az'));
      const hf = parseFloat(q.get('h'));
      const df = parseFloat(q.get('d'));
      let framed = 0;
      const f = setInterval(() => {
        if (window.__frame && window.__frame(az, hf, df)) framed++;
        if (framed > 30) clearInterval(f);
      }, 120);
    }
    setTimeout(() => { document.title = 'SHOT-READY'; }, 9000);
  }, 50);
})();
</script>
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(VIEWER, STAGE)
    idx = STAGE / "index.html"
    html = idx.read_text(encoding="utf8")
    assert "</body>" in html, "viewer/index.html has no </body> to append to"
    idx.write_text(html.replace("</body>", SHIM + "</body>"), encoding="utf8")

    # main.js is an ES module, so code appended to it shares scope with camera, yaw and
    # pitch. This is the staged copy; viewer/main.js is never written to.
    mjs = STAGE / "main.js"
    src = mjs.read_text(encoding="utf8")
    for need in ("const camera = new THREE.PerspectiveCamera", "let yaw = 0"):
        assert need in src, f"main.js no longer contains {need!r}; update the hook"
    mjs.write_text(src + CAM_HOOK, encoding="utf8")


def looks_rendered(path: Path) -> tuple[bool, str]:
    """A blank WebGL canvas still produces a valid PNG. Check it actually has content."""
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    # Ignore the HUD strip on the left; judge the 3D area on the right.
    view = a[:, int(a.shape[1] * 0.35):, :]
    sd = float(view.std())
    colours = len(np.unique((view // 24).astype(np.uint8).reshape(-1, 3), axis=0))
    return sd > 12 and colours > 40, f"std {sd:.1f}, {colours} colour bins"


def main() -> int:
    if not Path(CHROME).exists():
        print(f"  Chrome not at {CHROME}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    stage()

    port = free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(STAGE))
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"  serving {STAGE} on :{port}")

    ok = 0
    try:
        for idx, mode, name, framing in SHOTS:
            dest = OUT / f"{name}.png"
            url = f"http://127.0.0.1:{port}/index.html?scene={idx}&mode={mode}"
            if framing:
                az, hf, df = framing
                url += f"&az={az}&h={hf}&d={df}"
            cmd = [CHROME, "--headless=new", "--disable-gpu-sandbox", "--no-sandbox",
                   "--hide-scrollbars", "--force-device-scale-factor=2",
                   "--enable-unsafe-swiftshader", "--use-gl=angle",
                   "--use-angle=swiftshader",
                   f"--window-size={W},{H}", "--virtual-time-budget=26000",
                   f"--screenshot={dest}", url]
            subprocess.run(cmd, capture_output=True, timeout=180)
            if not dest.exists():
                print(f"  [FAIL] {name}: no file produced")
                continue
            good, why = looks_rendered(dest)
            kb = dest.stat().st_size / 1024
            print(f"  [{'OK  ' if good else 'BLANK'}] {name:<16} {kb:6.0f} KB  {why}")
            ok += int(good)
    finally:
        srv.shutdown()

    print(f"\n  {ok}/{len(SHOTS)} captures have real content -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
