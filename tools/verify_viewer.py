"""Load the built viewer in a real browser and assert it actually rendered.

    python tools/verify_viewer.py                    # checks viewer_standalone.html
    python tools/verify_viewer.py --url http://localhost:8080/

Why this exists
---------------
"It parses" and "it renders" are different claims, and only the second one matters for a
demo. The check that separates them is the canvas: the source ships ZERO `<canvas>` tags
and Three.js creates one at runtime, so a canvas in the rendered DOM is proof the WebGL
context came up and the render loop ran. The stats panel is the second proof -- those
fields start as em-dashes and are only filled once the height data is fetched, decoded
and measured.

Do NOT pass --use-gl=swiftshader. Measured 29 Aug 2026: software WebGL crashes the
renderer on the 15 MB standalone ("Abnormal renderer termination", GPU process
exit_code=-1073741819 / 0xC0000005 access violation) and the run hangs until killed.
The real GPU renders the same page in about eight seconds. This is why the check is not
in CI as-is: it needs a display-capable GPU.
"""
from __future__ import annotations

import argparse
import re
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


def find_browser() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("no Chrome or Edge found; edit CHROME_CANDIDATES")


def text_of(dom: str, element_id: str) -> str | None:
    """The text inside the element with this id, or None if it is not in the DOM."""
    m = re.search(rf'id="{re.escape(element_id)}"[^>]*>(.*?)<', dom, re.S)
    return m.group(1).strip() if m else None


def loading_overlay(dom: str) -> tuple[bool, str]:
    """Is the #loading overlay on screen, and if so what does it say?

    Scoped by counting div tags from the element, because in the standalone build the page
    also *contains* main.js as text, and that text mentions the error markup.
    """
    i = dom.find('id="loading"')
    if i < 0:
        return False, ""
    tag_end = dom.find(">", i)
    hidden = re.search(r'style="[^"]*display:\s*none', dom[i:tag_end]) is not None
    # Walk to the matching close so the inlined script below it is never read as content.
    j, depth = tag_end, 0
    while True:
        nxt_open, nxt_close = dom.find("<div", j + 1), dom.find("</div>", j + 1)
        if nxt_close < 0:
            break
        if 0 <= nxt_open < nxt_close:
            depth, j = depth + 1, nxt_open
        elif depth:
            depth, j = depth - 1, nxt_close
        else:
            j = nxt_close
            break
    inner = re.sub(r"<[^>]+>", " ", dom[tag_end + 1:j])
    return (not hidden), " ".join(inner.split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None,
                    help="page to load (default: the built viewer_standalone.html)")
    ap.add_argument("--budget-ms", type=int, default=9000,
                    help="virtual time to let the render loop run before dumping")
    ap.add_argument("--timeout", type=int, default=180, help="hard timeout, seconds")
    args = ap.parse_args()

    # The panel text carries en-dashes and multiplication signs; a cp1252 console turns
    # those into replacement characters and makes a passing check look broken.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    url = args.url or (ROOT / "viewer_standalone.html").resolve().as_uri()
    browser = find_browser()

    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser, "--headless=new", "--allow-file-access-from-files",
            f"--user-data-dir={profile}", "--window-size=1400,880",
            f"--virtual-time-budget={args.budget_ms}", "--dump-dom", url,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"FAIL  browser did not finish within {args.timeout}s")
            return 1
    dom = r.stdout.decode("utf-8", "replace")

    checks: list[tuple[bool, str]] = []

    n_canvas = dom.count("<canvas")
    checks.append((n_canvas >= 1,
                   f"WebGL context came up: {n_canvas} <canvas> in the rendered DOM "
                   f"(the source ships none)"))

    extent = text_of(dom, "s-extent")
    checks.append((bool(extent) and extent != "&mdash;" and "—" not in (extent or ""),
                   f"scene data loaded and measured: area = {extent!r}"))

    rng = text_of(dom, "s-range")
    checks.append((bool(rng) and "—" not in (rng or "") or (rng or "").count("–") == 1,
                   f"height range read from height.bin: {rng!r}"))

    model = text_of(dom, "s-model")
    checks.append((bool(model) and "produced by" in (model or ""),
                   f"model provenance is on screen: {model!r}"))

    errb = text_of(dom, "s-errb")
    checks.append((bool(errb) and errb.endswith("m"),
                   f"per-building error is the headline figure: {errb!r}"))

    # Which scene it opened on. Read from body[data-scene], which loadScene() sets: the
    # <option> carries no `selected` attribute because the value was assigned in JS, so
    # parsing the <select> reports the first option regardless of what is on screen.
    m = re.search(r'<body[^>]*\sdata-scene="([^"]+)"', dom)
    opened = m.group(1) if m else "?"
    checks.append((opened not in ("?", "urban_oma_288_042"),
                   f"landing scene is not our worst case: opened on {opened!r}"))

    # Every failure path -- fatal(), explainNoWebGL(), the no-scenes message -- works by
    # showing #loading with an `.err` inside it, and loadScene() hides #loading on success.
    # So "is #loading on screen" is the whole question.
    #
    # This check used to look for `id="fatal"` and "Something went wrong", neither of which
    # the viewer has ever emitted, so it passed unconditionally -- including on a page showing
    # nothing but an error. Found 12 Sep 2026. The replacement must also not be fooled by the
    # standalone, which inlines main.js into the page: the *source* of fatal() contains the
    # literal `<div class="err">`, so the search has to be scoped to the element itself.
    shown, detail = loading_overlay(dom)
    checks.append((not shown,
                   "#loading is dismissed, so no error panel is on screen" if not shown
                   else f"ERROR PANEL ON SCREEN: {detail[:110]!r}"))

    # A visible toast means something failed after the scene came up (see trouble()). Not
    # fatal to the demo by design, but it must never be the state a verification run passes.
    toast = re.search(r'id="toast"[^>]*style="[^"]*display:\s*block', dom)
    checks.append((toast is None, "no late-failure toast showing"))

    ok = True
    for passed, msg in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {msg}")
        ok &= passed

    print(f"\n{'VIEWER OK' if ok else 'VIEWER BROKEN'}  ({url})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
