"""Record the demo video's scripted shots from the real viewer, frame by frame.

    python tools/video/record_shots.py            # every shot
    python tools/video/record_shots.py flood      # just the named ones

Each frame sets the camera (and any control) explicitly, waits for the render, takes a
screenshot and pipes it to ffmpeg. Nothing depends on wall-clock timing, so motion is
perfectly smooth however slow the capture runs. Rendered on the real GPU through the
installed Chrome (Playwright channel="chrome"); the shot list follows
docs/demo-video-script.md, and each shot is its own file so it can be trimmed to the
voice-over.

Needs: pip install playwright imageio-ffmpeg   (no browser download; uses installed Chrome)
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ppt"))
import capture_viewer as cv                       # noqa: E402  stage() copies viewer/ + hooks

OUT = Path("D:/sih2026/video")
W, H, FPS = 1920, 1080, 30
SCENE = {"urban": 0, "sparse": 1, "valley": 2, "forest": 3, "mixed": 4, "town": 5}

cv.STAGE = ROOT / "tools" / "video" / "_stage"
cv.SHIM = "<script></script>"
# __look aims the camera at a real point (cv's __frame aims at y = 0, the lowest ground,
# which on a mountain means looking under it). __box gives the surface's vertical span.
cv.CAM_HOOK += """
window.__look = (px, py, pz, tx, ty, tz) => {
  camera.position.set(px, py, pz);
  const dx = tx - px, dy = ty - py, dz = tz - pz;
  yaw = Math.atan2(-dx, -dz);
  pitch = Math.atan2(dy, Math.hypot(dx, dz));
  return true;
};
window.__box = () => {
  mesh.geometry.computeBoundingBox();
  const b = mesh.geometry.boundingBox, s = mesh.scale.y;
  return { e: sceneExtent(), lo: b.min.y * s, hi: b.max.y * s };
};
window.__split = (x) => { state.split = x; layoutDivider(); };
"""

HIDE_UI = "#hud,#stats,#help,#compass,#scalebar,#hover,#readout,#toast{display:none!important}"


def ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


class Rec:
    def __init__(self, page):
        self.page = page

    # -- plumbing
    def settle(self):
        self.page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")

    def load(self, name: str):
        self.page.evaluate(f"""() => {{ const s = document.getElementById('scene');
            s.selectedIndex = {SCENE[name]}; s.dispatchEvent(new Event('change')); }}""")
        self.page.wait_for_timeout(400)
        self.page.wait_for_function("document.getElementById('loading').style.display === 'none'",
                                    timeout=120000)
        for _ in range(10):
            self.settle()
        return self.page.evaluate("window.__box()")

    def ui(self, visible: bool):
        self.page.evaluate("document.getElementById('__hide')?.remove()")
        if not visible:
            self.page.add_style_tag(content=HIDE_UI)
            self.page.evaluate("document.querySelector('style:last-of-type').id = '__hide'")

    def mode(self, value: str):
        self.page.evaluate(f"""() => {{ const m = document.getElementById('mode');
            m.value = '{value}'; m.dispatchEvent(new Event('change')); }}""")

    def look(self, p, t):
        self.page.evaluate(f"window.__look({p[0]}, {p[1]}, {p[2]}, {t[0]}, {t[1]}, {t[2]})")

    def record(self, name: str, seconds: float, frame):
        """Call frame(t in 0..1) once per output frame, capture, encode to OUT/name.mp4."""
        OUT.mkdir(parents=True, exist_ok=True)
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                               "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                               str(OUT / f"{name}.mp4")], stdin=subprocess.PIPE)
        n = round(seconds * FPS)
        t0 = time.time()
        for f in range(n):
            frame(f / (n - 1))
            self.settle()
            ff.stdin.write(self.page.screenshot(type="jpeg", quality=93))
        ff.stdin.close()
        ff.wait()
        print(f"  {name:<22} {n} frames, {seconds:.0f} s   ({time.time() - t0:.0f} s to record)")


# ------------------------------------------------------------------ the shots
def orbit(r: Rec, bx, *, a0, a1, r0, r1, h0, h1, aim=0.35):
    e, lo, hi = bx["e"], bx["lo"], bx["hi"]
    mid = lo + aim * (hi - lo)

    def frame(t):
        k = ease(t)
        a = math.radians(a0 + (a1 - a0) * k)
        rr = e * (r0 + (r1 - r0) * k)
        y = hi + e * (h0 + (h1 - h0) * k)
        r.look((math.sin(a) * rr, y, math.cos(a) * rr), (0, mid, 0))
    return frame


def shot_hook(r: Rec):
    """1 - Hook: slow orbit and push-in over the Sikkim valley, panels hidden."""
    bx = r.load("valley"); r.ui(False); r.mode("texture")
    r.record("c1_hook_sikkim", 15, orbit(r, bx, a0=-30, a1=30, r0=0.66, r1=0.55, h0=0.26, h1=0.20))


def shot_problem(r: Rec):
    """2 - The problem: Omaha straight down (it reads as a photo), then tilt into 3D."""
    bx = r.load("urban"); r.ui(False); r.mode("texture")
    e, lo, hi = bx["e"], bx["lo"], bx["hi"]
    tgt = (0, lo + 0.25 * (hi - lo), 0)

    def frame(t):
        # Hold top-down for the first 20%, then tilt over the remaining 80%.
        k = ease((t - 0.2) / 0.8)
        a = math.radians(-35 * k)
        horiz = e * (0.001 + 0.62 * k)
        y = hi + e * (0.95 - 0.62 * k)
        r.look((math.sin(a) * horiz, y, math.cos(a) * horiz), tgt)
    r.record("c2_problem_omaha", 20, frame)


def shot_flythrough(r: Rec):
    """4 - Fly low over Mixed Jacksonville along an arc, interface visible."""
    bx = r.load("mixed"); r.ui(True); r.mode("texture")
    e, lo = bx["e"], bx["lo"]

    def frame(t):
        k = ease(t)
        a = math.radians(210 + 120 * k)             # sweep round the park
        rad = e * 0.46
        p = (math.sin(a) * rad, lo + e * 0.11, math.cos(a) * rad)
        ahead = math.radians(210 + 120 * k + 55)    # look across, ahead of travel
        q = (math.sin(ahead) * rad * 0.35, lo + e * 0.01, math.cos(ahead) * rad * 0.35)
        r.look(p, q)
    r.record("c4_flythrough_jacksonville", 15, frame)


def shot_sigma(r: Rec):
    """5a - Mixed Jacksonville: satellite, then switch Show to 'How sure we are'."""
    bx = r.load("mixed"); r.ui(True); r.mode("texture")
    fr = orbit(r, bx, a0=-15, a1=15, r0=0.62, r1=0.56, h0=0.28, h1=0.24, aim=0.2)
    state = {"switched": False}

    def frame(t):
        if t >= 0.35 and not state["switched"]:
            r.mode("sigma"); state["switched"] = True
        fr(t)
    r.record("c5a_confidence_jacksonville", 12, frame)


def shot_compare(r: Rec):
    """5b - Omaha: drag-to-compare with LiDAR, the divider sweeping across."""
    bx = r.load("urban"); r.ui(True); r.mode("texture")
    r.page.click("#compare")
    e, lo, hi = bx["e"], bx["lo"], bx["hi"]
    r.look((e * 0.08, hi + e * 0.30, e * 0.58), (0, lo + 0.2 * (hi - lo), 0))

    def frame(t):
        # 0.5 -> 0.15 -> 0.85 -> 0.5, eased between each
        keys = [(0, .5), (.2, .15), (.7, .85), (1, .5)]
        for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
            if t <= t1:
                x = x0 + (x1 - x0) * ease((t - t0) / (t1 - t0)); break
        r.page.evaluate(f"window.__split({x})")
        r.look((e * 0.08, hi + e * 0.30, e * 0.58), (0, lo + 0.2 * (hi - lo), 0))
    r.record("c5b_lidar_compare_omaha", 14, frame)
    r.page.click("#compare")


def shot_error(r: Rec):
    """5c - Omaha: 'Where we're wrong', slow orbit."""
    bx = r.load("urban"); r.ui(True); r.mode("error")
    r.record("c5c_where_wrong_omaha", 10,
             orbit(r, bx, a0=-20, a1=10, r0=0.60, r1=0.52, h0=0.30, h1=0.26, aim=0.15))
    r.mode("texture")


def shot_flood(r: Rec):
    """6 - Sikkim: 'Flood the valley', water rising slowly to 40% of the range, then held."""
    r.load("valley"); r.ui(True); r.mode("texture")
    r.page.evaluate("window.__frame(0, 0.70, 0.45)")
    r.page.click("#flood")
    lo, hi = r.page.evaluate("[+document.getElementById('water').min, +document.getElementById('water').max]")
    target = round(lo + (hi - lo) * 0.40)

    def frame(t):
        v = round(lo + (target - lo) * ease(t / 0.8))       # rise over 80%, hold the rest
        r.page.evaluate(f"""() => {{ const w = document.getElementById('water');
            w.value = {v}; w.dispatchEvent(new Event('input', {{ bubbles: true }}));
            window.__frame(0, 0.70, 0.45); }}""")
        r.page.wait_for_timeout(110)     # the slider applies the flood after a 90 ms pause
    r.record("c6_flood_sikkim", 18, frame)
    r.page.click("#flood")


def title_card():
    """8 - Close: a static end card, 8 s."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (W, H), (14, 32, 48))
    d = ImageDraw.Draw(im)
    def font(sz, bold=False):
        for f in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
            try:
                return ImageFont.truetype(f, sz)
            except OSError:
                pass
        return ImageFont.load_default()
    d.text((160, 330), "Depth", font=font(120, True), fill=(255, 255, 255))
    x = 160 + d.textlength("Depth", font=font(120, True))
    d.text((x, 330), "Wizard", font=font(120, True), fill=(217, 160, 60))
    d.text((164, 480), "Height from one satellite image, with the evidence to trust it.",
           font=font(44), fill=(207, 224, 238))
    d.text((164, 600), "Team Dev Up  ·  SIH26175  ·  Team ID 129655", font=font(36, True),
           fill=(255, 255, 255))
    for i, (k, v) in enumerate([("LIVE", "project5.zaidansari.tech"),
                                ("DOCS", "project5.zaidansari.tech/documentation"),
                                ("CODE", "github.com/zaidnansari2011/sih2026-depthwizard")]):
        d.text((164, 690 + i * 52), k, font=font(30, True), fill=(217, 160, 60))
        d.text((290, 690 + i * 52), v, font=font(30), fill=(207, 224, 238))
    d.text((164, 980), "Sikkim imagery © Maxar Technologies, Maxar Open Data, CC-BY-4.0.",
           font=font(24), fill=(140, 160, 178))
    png = OUT / "c8_end_card.png"
    im.save(png)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-loop", "1",
                    "-framerate", str(FPS), "-t", "8", "-i", str(png), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(OUT / "c8_end_card.mp4")], check=True)
    print("  c8_end_card            8 s")


def intro_card():
    """0 - Intro: title, problem statement and team, 9 s with a fade in and out."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (W, H), (14, 32, 48))
    d = ImageDraw.Draw(im)
    def font(sz, bold=False):
        for f in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
            try:
                return ImageFont.truetype(f, sz)
            except OSError:
                pass
        return ImageFont.load_default()
    d.text((164, 250), "SMART INDIA HACKATHON 2026  ·  ISRO  ·  PROBLEM STATEMENT SIH26175",
           font=font(30, True), fill=(217, 160, 60))
    d.text((160, 320), "Depth", font=font(140, True), fill=(255, 255, 255))
    x = 160 + d.textlength("Depth", font=font(140, True))
    d.text((x, 320), "Wizard", font=font(140, True), fill=(217, 160, 60))
    d.text((164, 500), "Single-view height estimation and 3D flythrough", font=font(48),
           fill=(207, 224, 238))
    d.line([(164, 610), (900, 610)], fill=(60, 90, 115), width=2)
    d.text((164, 640), "Team Dev Up", font=font(40, True), fill=(255, 255, 255))
    d.text((164, 700), "Team ID 129655  ·  Theme: Disaster Management", font=font(32),
           fill=(207, 224, 238))
    png = OUT / "c0_intro_card.png"
    im.save(png)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-loop", "1",
                    "-framerate", str(FPS), "-t", "9", "-i", str(png),
                    "-vf", "fade=t=in:st=0:d=0.6,fade=t=out:st=8.4:d=0.6", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(OUT / "c0_intro_card.mp4")], check=True)
    print("  c0_intro_card          9 s")


SHOTS = {"hook": shot_hook, "problem": shot_problem, "flythrough": shot_flythrough,
         "sigma": shot_sigma, "compare": shot_compare, "error": shot_error, "flood": shot_flood}


def main() -> None:
    want = sys.argv[1:] or ["intro"] + list(SHOTS) + ["card"]
    cv.stage()

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    port = cv.free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Quiet, directory=str(cv.STAGE)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        if any(w in SHOTS for w in want):
            with sync_playwright() as p:
                b = p.chromium.launch(channel="chrome", headless=True,
                                      args=["--enable-gpu", "--use-angle=d3d11",
                                            "--ignore-gpu-blocklist"])
                page = b.new_page(viewport={"width": W, "height": H})
                page.goto(f"http://127.0.0.1:{port}/index.html")
                page.wait_for_function(
                    "document.getElementById('loading').style.display === 'none'", timeout=120000)
                r = Rec(page)
                for w in want:
                    if w in SHOTS:
                        SHOTS[w](r)
                b.close()
        if "intro" in want:
            intro_card()
        if "card" in want:
            title_card()
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
