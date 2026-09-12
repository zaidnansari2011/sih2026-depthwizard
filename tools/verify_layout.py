"""Does the viewer's chrome still fit on the screen it will be judged on?

    python tools/verify_layout.py                 # the built standalone, four screen sizes
    python tools/verify_layout.py --page viewer/index.html
    python tools/verify_layout.py --json          # machine-readable, for a diff

It measures a LOCAL page only, and says so rather than guessing: the probe has to be
appended to the page's own markup, and the deployed site sends no CORS headers, so a copy
of it loaded from disk cannot fetch its own scene data. That costs nothing, because the
geometry comes from the stylesheet both copies share -- and the one live-only difference,
the upload card that appears when a server is behind the page, is force-shown by the probe
anyway. Use verify_viewer.py --url to check that the deployment renders.

Why this exists
---------------
`verify_viewer.py` proves the scene renders. It says nothing about whether the panels that
explain the scene are reachable, and those are two different failures. A demo hall projector
is 1024x768 or 1366x768, not the 1400x880 the viewer was built on, and `html, body` carry
`overflow: hidden` -- so a panel taller than the viewport does not get a scrollbar, it gets
silently truncated. Controls below the fold simply do not exist for the person at the
keyboard, and nothing on screen hints that they are missing.

What it measures, rather than assumes
-------------------------------------
A probe script is appended to a COPY of the page (the original is never touched) which,
after the render loop has run, reads `getBoundingClientRect()` and `scrollHeight` off every
panel and writes them into the DOM as JSON. Chrome is then asked to dump the DOM. Two
states are captured at each size:

  * `as-loaded` -- what a judge sees on arrival.
  * `all-panels-open` -- every block the HUD can ever reveal at once: the upload card (shown
    only when a server is behind the page, so absent from the standalone but present on the
    live site), the flood controls, the legend and the measurement readout. This is the
    state that actually overflows, and it is reachable by clicking two buttons.

Three failures are checked:

  1. **off-screen** -- a panel extends past an edge. Unreachable: no scrollbar exists.
  2. **clipped** -- `scrollHeight` exceeds `clientHeight` and the element cannot scroll, so
     its own content is cut off inside it.
  3. **overlap** -- two panels cover each other. The HUD growing down into the scale bar and
     the key hints is the one that happens.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

# Sizes that matter, with why. 1366x768 is the single most common laptop panel and what a
# hall machine usually is; 1024x768 is the 4:3 projector fallback a hall switches to when
# it cannot negotiate; 1280x720 is the common HDMI mirror mode.
SIZES = [
    (1024, 768, "4:3 projector fallback"),
    (1280, 720, "720p HDMI mirror"),
    (1366, 768, "most common laptop panel"),
    (1920, 1080, "full HD, the happy case"),
]

PANELS = ["hud", "stats", "help", "scalebar", "readout", "hover", "toast"]
# Pairs that can realistically collide. Panels pinned to opposite corners are listed only
# where one of them grows: #hud grows downward into the bottom-left furniture, and #toast
# is centred at the bottom, which is where the key hints and the readout already live.
PAIRS = [("hud", "scalebar"), ("hud", "help"), ("stats", "readout"), ("hud", "stats"),
         ("toast", "help"), ("toast", "scalebar"), ("toast", "readout"),
         ("toast", "hud"), ("toast", "stats")]

PROBE = """
<script>
/* Appended by tools/verify_layout.py. Measures, never changes behaviour. */
(function () {
  function box(el) {
    var r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    return {
      x: r.x, y: r.y, w: r.width, h: r.height,
      scrollH: el.scrollHeight, clientH: el.clientHeight,
      overflowY: cs.overflowY, shown: cs.display !== 'none'
    };
  }
  function snap(label) {
    var o = { label: label, vw: innerWidth, vh: innerHeight, panels: {} };
    ['hud', 'stats', 'help', 'scalebar', 'readout', 'hover', 'toast'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) o.panels[id] = box(el);
    });
    return o;
  }
  function openEverything() {
    /* The legend is populated by applyMode, so drive the real control rather than
       unhiding an empty box -- an empty legend would understate the height. */
    var mode = document.getElementById('mode');
    if (mode) { mode.value = 'height'; mode.dispatchEvent(new Event('change')); }
    ['upload-block', 'flood-ctl', 'readout', 'legend', 'scalebar'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.style.display = 'block';
    });
    /* The scale bar is shown by updateScaleBar(), which runs on every tenth animation
       frame -- and headless Chrome under --virtual-time-budget produces about six frames
       for the whole session (measured 12 Sep 2026), so it never runs and the panel would
       be skipped as hidden. Forcing it visible measures the geometry that matters; whether
       it *becomes* visible is verify_viewer.py's business, not this tool's. */
    var sb = document.getElementById('sb-bar');
    if (sb && !sb.style.width) sb.style.width = '120px';
    /* Same for the toast: it only appears when something goes wrong, and its whole job is
       to appear over a running scene without covering anything that matters. */
    var t = document.getElementById('toast');
    if (t && getComputedStyle(t).display === 'none') {
      t.innerHTML = '<b>Mouse-look did not engage.</b><div class="note">Browsers block '
        + 're-locking the mouse for about a second after Esc, and refuse it while the '
        + 'window is not focused. Click the scene again.</div>';
      t.style.display = 'block';
    }
  }
  function report() {
    var out = [snap('as-loaded')];
    openEverything();
    out.push(snap('all-panels-open'));
    var pre = document.createElement('pre');
    pre.id = 'layout-report';
    pre.textContent = JSON.stringify(out);
    document.body.appendChild(pre);
  }
  addEventListener('load', function () { setTimeout(report, 5000); });
})();
</script>
"""


def find_browser() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("no Chrome or Edge found; edit CHROME_CANDIDATES")


def instrument(src: Path, dst: Path) -> None:
    """Copy the page and append the probe. The original is never modified."""
    html = src.read_text(encoding="utf-8")
    i = html.rfind("</body>")
    if i < 0:
        raise SystemExit(f"{src} has no </body> to append the probe before")
    dst.write_text(html[:i] + PROBE + html[i:], encoding="utf-8")
    # Scene binaries are embedded in the standalone, but an un-built viewer/index.html
    # fetches ./main.js and ./scenes/*, so the copy has to sit beside them.
    for extra in ("main.js", "vendor", "scenes", "calibration.json"):
        p = src.parent / extra
        if not p.exists():
            continue
        t = dst.parent / extra
        if t.exists():
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, t)


def measure(browser: str, url: str, w: int, h: int, budget: int, timeout: int):
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser, "--headless=new", "--allow-file-access-from-files",
            f"--user-data-dir={profile}", f"--window-size={w},{h}",
            f"--virtual-time-budget={budget}", "--dump-dom", url,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"browser did not finish within {timeout}s"
    dom = r.stdout.decode("utf-8", "replace")
    m = re.search(r'id="layout-report"[^>]*>(.*?)</pre>', dom, re.S)
    if not m:
        return None, "probe never reported (the page may not have finished loading)"
    raw = m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"probe output was not JSON: {e}"


def faults(snap: dict) -> list[str]:
    """Every way this snapshot is unusable, in plain words."""
    out: list[str] = []
    vw, vh = snap["vw"], snap["vh"]
    p = {k: v for k, v in snap["panels"].items() if v["shown"] and v["h"] > 0}

    for name, b in sorted(p.items()):
        over = []
        if b["y"] < -0.5:
            over.append(f"{-b['y']:.0f}px above the top")
        if b["y"] + b["h"] > vh + 0.5:
            over.append(f"{b['y'] + b['h'] - vh:.0f}px below the bottom")
        if b["x"] < -0.5:
            over.append(f"{-b['x']:.0f}px off the left")
        if b["x"] + b["w"] > vw + 0.5:
            over.append(f"{b['x'] + b['w'] - vw:.0f}px off the right")
        if over:
            out.append(f"off-screen  #{name} runs {' and '.join(over)} "
                       f"(panel is {b['h']:.0f}px tall, viewport {vh}px)")
        # A panel that can scroll is fine; one that cannot is silently cut.
        if b["scrollH"] > b["clientH"] + 1 and b["overflowY"] in ("visible", "hidden", "clip"):
            out.append(f"clipped     #{name} holds {b['scrollH']:.0f}px of content in a "
                       f"{b['clientH']:.0f}px box with overflow-y: {b['overflowY']}")

    for a, b in PAIRS:
        if a not in p or b not in p:
            continue
        ra, rb = p[a], p[b]
        ox = min(ra["x"] + ra["w"], rb["x"] + rb["w"]) - max(ra["x"], rb["x"])
        oy = min(ra["y"] + ra["h"], rb["y"] + rb["h"]) - max(ra["y"], rb["y"])
        if ox > 1 and oy > 1:
            out.append(f"overlap     #{a} and #{b} cover each other over "
                       f"{ox:.0f}x{oy:.0f}px")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Deliberately not type=Path: Path() mangles "https://host" into "https:\host", which
    # would slip past the guard below and fail as a confusing missing-file error instead.
    ap.add_argument("--page", default=None,
                    help="local html to instrument (default: viewer_standalone.html)")
    ap.add_argument("--budget-ms", type=int, default=14000)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--only", default=None, help="one size as WxH, e.g. 1366x768")
    ap.add_argument("--json", action="store_true", help="dump the raw measurements too")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    browser = find_browser()
    sizes = SIZES
    if args.only:
        w, h = (int(v) for v in args.only.lower().split("x"))
        sizes = [(w, h, "requested")]

    tmp = Path(tempfile.mkdtemp(prefix="dwz-layout-"))
    try:
        if args.page and args.page.lower().startswith(("http:", "https:")):
            raise SystemExit(
                "this tool measures a local page only. The probe has to be appended to the\n"
                "page's own markup, and the deployed site sends no CORS headers, so a copy of\n"
                "it loaded from disk cannot fetch its scene data. The geometry is identical\n"
                "either way -- it comes from the stylesheet both copies share. Run it on\n"
                "viewer_standalone.html or viewer/index.html, and use\n"
                "  python tools/verify_viewer.py --url <site>\n"
                "to check that the deployment renders.")
        src = Path(args.page) if args.page else (ROOT / "viewer_standalone.html")
        if not src.exists():
            raise SystemExit(f"{src} not found -- run tools/build_standalone.py first")
        probed = tmp / src.name
        instrument(src, probed)
        url = probed.resolve().as_uri()

        print(f"\nlayout check  {src}")
        print(f"  probing {len(sizes)} screen size(s); each needs a real GPU and ~15s\n")

        bad = 0
        raw_all = []
        for w, h, why in sizes:
            snaps, err = measure(browser, url, w, h, args.budget_ms, args.timeout)
            print(f"{w}x{h}  ({why})")
            if err:
                print(f"    ERROR  {err}")
                bad += 1
                continue
            raw_all.append({"size": [w, h], "snaps": snaps})
            for snap in snaps:
                f = faults(snap)
                tag = "ok" if not f else f"{len(f)} problem(s)"
                print(f"    {snap['label']:<17} {tag}")
                for line in f:
                    print(f"      {line}")
                bad += len(f)
            print()

        if args.json:
            print(json.dumps(raw_all, indent=1))

        print("LAYOUT OK" if not bad else f"LAYOUT BROKEN  {bad} problem(s)")
        return 0 if not bad else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
