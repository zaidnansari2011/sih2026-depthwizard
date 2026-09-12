"""Does the viewer report the right place on Earth -- and admit when it cannot?

    python tools/verify_geo.py

Why this exists
---------------
A3 puts a coordinate system, a datum, a centre position, a per-pixel longitude and latitude
and a north arrow into the viewer. Every one of those is a claim about the world that a
geospatial jury can check against their own tools, so a viewer that shows a plausible-looking
wrong number is worse than one that shows nothing.

The expected values are not hard-coded here. They are recomputed from each scene's own
GeoTIFF transform through rasterio's projection library, and compared against what the page
actually renders. The viewer reaches its answer a different way -- it interpolates between
four corner positions baked in at export time, so that no projection library has to ship to
the browser -- so agreement between the two is a real check on the approximation, not a
tautology.

The other half matters as much: four of the six demo scenes are DFC2019 tiles that carry no
CRS and no transform at all. On those the viewer must say so plainly rather than inventing a
position, and must not draw a north arrow, because there is no north to point at.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "viewer" / "scenes"
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]
GEOREFERENCED = "hilly_sikkim_valley"
PLAIN = "mixed_jax_020_020"

PROBE = """
<script>
/* Appended by tools/verify_geo.py. Reads the page; never reaches into the module. */
(function () {
  function txt(id) { var e = document.getElementById(id); return e ? e.textContent.trim() : null; }
  function disp(id) {
    var e = document.getElementById(id);
    return e ? getComputedStyle(e).display : 'absent';
  }
  /* Waiting on animation frames was tried and abandoned: with the frame-rate flags this
     page produced no rAF ticks at all after the first scene loaded, so an rAF chain never
     finished. Wall time it is -- generously, because the hover readout is throttled to one
     raycast per frame and frames here are scarce. */
  function wait(ms, then) { setTimeout(then, ms); }
  function read(label) {
    return {
      scene: label, centre: txt('s-centre'), en: txt('s-en'), crs: txt('s-crs'),
      datum: txt('s-datum'), note: txt('s-geonote'), compass: disp('compass'),
      needle: (document.getElementById('needle') || {}).getAttribute
        ? document.getElementById('needle').getAttribute('transform') : null,
      hover: txt('h-pos'), hover_row: disp('h-pos-row')
    };
  }
  function hover(fy) {
    var c = document.querySelector('canvas'), r = c.getBoundingClientRect();
    c.dispatchEvent(new PointerEvent('pointermove', {
      clientX: r.left + r.width / 2, clientY: r.top + r.height * fy, bubbles: true,
      cancelable: true, pointerId: 1, pointerType: 'mouse', isPrimary: true
    }));
  }
  /* Turn the camera by a known amount: 10 moves of 14px at 0.005 rad/px = -0.7 rad. */
  function orbit() {
    var c = document.querySelector('canvas'), r = c.getBoundingClientRect();
    var x = r.left + r.width / 2, y = r.top + r.height / 2;
    function pe(t, px, py, b) {
      c.dispatchEvent(new PointerEvent(t, {
        clientX: px, clientY: py, button: 0, buttons: b, bubbles: true, cancelable: true,
        pointerId: 1, pointerType: 'mouse', isPrimary: true
      }));
    }
    pe('pointerdown', x, y, 1);
    for (var i = 1; i <= 10; i++) pe('pointermove', x + i * 14, y, 1);
    pe('pointerup', x + 140, y, 0);
  }
  function sample(fy, then) { hover(fy); wait(700, then); }

  /* Written after every stage, not only at the end: if the page runs out of animation
     frames mid-chain the report still says how far it got, which is the difference
     between a diagnosis and a shrug. */
  var stage = 'start', ticks = 0;
  (function count() { ticks++; requestAnimationFrame(count); })();
  function done(out) {
    var pre = document.getElementById('geo-report');
    if (!pre) {
      pre = document.createElement('pre');
      pre.id = 'geo-report';
      document.body.appendChild(pre);
    }
    pre.textContent = JSON.stringify({ stage: stage, frames: ticks, rows: out });
  }

  addEventListener('load', function () {
    setTimeout(function () {                       // the first scene has to finish loading
      sample(0.5, function () {
        var out = [read('__PLAIN__')];
        stage = 'plain-read';
        done(out);
        var sel = document.getElementById('scene');
        sel.value = '__GEO__';
        sel.dispatchEvent(new Event('change'));
        setTimeout(function () {                   // and so does the second
          sample(0.5, function () {
            var r = read('__GEO__');
            stage = 'geo-read';
            done(out.concat([r]));
            orbit();                             // the needle updates on the gesture
            r.needle_turned = document.getElementById('needle').getAttribute('transform');
            stage = 'complete';
            out.push(r);
            done(out);
          });
        }, 3000);
      });
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


def expected(scene: str) -> dict | None:
    """What the centre of this scene really is, straight from its own transform."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp

    m = json.loads((SCENES / scene / "manifest.json").read_text(encoding="utf-8"))
    g = m.get("geo") or {}
    if not g.get("crs") or not g.get("transform"):
        return None
    a = g["transform"]
    c, r = (m["width"] - 1) / 2 + 0.5, (m["height"] - 1) / 2 + 0.5
    e, n = a[0] * c + a[1] * r + a[2], a[3] * c + a[4] * r + a[5]
    lon, lat = warp(CRS.from_string(g["crs"]), CRS.from_epsg(4326), [e], [n])
    return {"crs": g["crs"], "e": e, "n": n, "lon": lon[0], "lat": lat[0]}


def parse_lonlat(s: str) -> tuple[float, float] | None:
    m = re.match(r"([\d.]+)°\s*([NS]),\s*([\d.]+)°\s*([EW])", s or "")
    if not m:
        return None
    lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
    lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
    return lon, lat


def metres_apart(lon1, lat1, lon2, lat2) -> float:
    return math.hypot((lat1 - lat2) * 111320.0,
                      (lon1 - lon2) * 111320.0 * math.cos(math.radians(lat1)))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    src = ROOT / "viewer_standalone.html"
    if not src.exists():
        raise SystemExit(f"{src} not found -- run tools/build_standalone.py first")

    html = src.read_text(encoding="utf-8")
    i = html.rfind("</body>")
    tmp = Path(tempfile.mkdtemp(prefix="dwz-geo-"))
    page = tmp / "geo.html"
    page.write_text(
        html[:i] + PROBE.replace("__GEO__", GEOREFERENCED).replace("__PLAIN__", PLAIN)
        + html[i:], encoding="utf-8")

    dom = subprocess.run(
        [find_browser(), "--headless=new", "--allow-file-access-from-files",
         f"--user-data-dir={tmp / 'prof'}", "--window-size=1366,768",
         # No frame-rate flags: --disable-frame-rate-limit and
         # --run-all-compositor-stages-before-draw were measured to leave this page with no
         # animation frames at all once the first scene had loaded. See the plan's gotcha 6.
         "--virtual-time-budget=16000", "--dump-dom", page.resolve().as_uri()],
        capture_output=True, timeout=300).stdout.decode("utf-8", "replace")
    m = re.search(r'id="geo-report"[^>]*>(.*?)</pre>', dom, re.S)
    if not m:
        raise SystemExit("probe never reported")
    raw = m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    report = json.loads(raw)
    rows = report["rows"] if isinstance(report, dict) else report
    got = {r["scene"]: r for r in rows}
    if isinstance(report, dict):
        print(f"  probe reached stage {report.get('stage')!r} "
              f"after {report.get('frames')} animation frames\n")

    checks: list[tuple[bool, str]] = []
    skipped: list[str] = []

    # --- the georeferenced scene must agree with its own projection ------------------
    exp, saw = expected(GEOREFERENCED), got.get(GEOREFERENCED, {})
    assert exp, f"{GEOREFERENCED} has no CRS -- this check is pointed at the wrong scene"
    shown = parse_lonlat(saw.get("centre", ""))
    if shown is None:
        checks.append((False, f"centre unreadable: {saw.get('centre')!r}"))
    else:
        off = metres_apart(shown[0], shown[1], exp["lon"], exp["lat"])
        # 4 decimal places is ~11 m of rounding; anything beyond that is a real error.
        checks.append((off < 12.0,
                       f"centre agrees with rasterio to {off:.1f} m "
                       f"(shown {saw['centre']}, exact {exp['lat']:.5f} N {exp['lon']:.5f} E)"))
    checks.append((exp["crs"] in (saw.get("crs") or ""),
                   f"coordinate system named: {saw.get('crs')!r}"))
    checks.append(((saw.get("datum") or "").startswith("WGS 84"),
                   f"datum named: {saw.get('datum')!r}"))
    en = re.match(r"(-?\d+) E\s+(-?\d+) N", saw.get("en") or "")
    checks.append((bool(en) and abs(int(en.group(1)) - exp["e"]) < 2
                   and abs(int(en.group(2)) - exp["n"]) < 2,
                   f"easting/northing agrees: {saw.get('en')!r} vs "
                   f"{exp['e']:.0f} E {exp['n']:.0f} N"))
    checks.append((saw.get("compass") == "block", "north arrow shown on a north-up scene"))

    # Which way is north. A needle 180 degrees out would pass the rotation check below, so
    # the direction is verified separately -- and from the data and the source rather than
    # from the screen. Sampling the readout at two screen heights was tried first and
    # abandoned: the hover is throttled to one raycast per animation frame, and this page
    # gets about eleven frames for a whole headless session, so both samples came back
    # holding the same stale value and the test reported a 0 m difference either way.
    #
    # The arrow rests on exactly one claim -- that increasing raster row goes south, while
    # the mesh lays increasing row along +Z, so north is -Z. Both halves are checkable.
    for scene in sorted(p.parent.name for p in SCENES.glob("*/manifest.json")):
        e = expected(scene)
        if not e:
            continue
        g = json.loads((SCENES / scene / "manifest.json").read_text(encoding="utf-8"))["geo"]
        tl, bl = g["corners_lonlat"][0], g["corners_lonlat"][2]
        checks.append((tl[1] > bl[1],
                       f"{scene}: raster rows run south, so -Z is north "
                       f"(top {tl[1]:.5f} N > bottom {bl[1]:.5f} N)"))

    src_js = (ROOT / "viewer" / "main.js").read_text(encoding="utf-8")
    checks.append(("pos[k * 3 + 2] = y * gsd - cz;" in src_js,
                   "the mesh really lays raster rows along +Z (surfaceGeometry)"))
    checks.append(("Math.abs(g.transform[1]) > 1e-9" in src_js,
                   "a rotated transform is refused rather than pointed at wrongly"))

    # The needle is one rotation of yaw, and the drag above turns yaw by a known amount:
    # 10 moves x 14 px x 0.005 rad/px = -0.7 rad = -40.1 degrees.
    turned = re.search(r"rotate\((-?[\d.]+)\)", saw.get("needle_turned") or "")
    want = -math.degrees(10 * 14 * 0.005)
    if not turned:
        checks.append((False, f"needle unreadable after a turn: {saw.get('needle_turned')!r}"))
    else:
        checks.append((abs(float(turned.group(1)) - want) < 1.0,
                       f"needle follows the camera: {float(turned.group(1)):.1f}° after a "
                       f"drag that turns the view {want:.1f}°"))

    hov = parse_lonlat(saw.get("hover") or "")
    if hov is None:
        # Not a failure of the viewer. The readout is throttled to one raycast per animation
        # frame and this page gets roughly a dozen frames for a whole headless session, so
        # the sample sometimes lands before any frame has run. Saying "not run" is honest;
        # calling it a pass would be a lie and calling it a failure would be a false alarm.
        skipped.append("cursor position: no animation frame ran during the sample, so the "
                       "hover readout never updated. Re-run, or check it in a browser.")
    else:
        # The cursor is not at the centre, but it is on the tile: a 2 km tile means any
        # point on it is within ~1.5 km of the middle.
        off = metres_apart(hov[0], hov[1], exp["lon"], exp["lat"])
        checks.append((off < 1600.0,
                       f"cursor position is on the tile: {saw.get('hover')} "
                       f"({off:.0f} m from centre)"))

    # --- the unreferenced scene must admit it ----------------------------------------
    saw = got.get(PLAIN, {})
    checks.append((expected(PLAIN) is None, f"{PLAIN} really has no CRS (precondition)"))
    checks.append((saw.get("compass") == "none", "no north arrow where there is no north"))
    checks.append((parse_lonlat(saw.get("centre") or "") is None,
                   f"no position invented: centre reads {saw.get('centre')!r}"))
    checks.append(("without georeferencing" in (saw.get("note") or ""),
                   f"says so in words: {(saw.get('note') or '')[:72]!r}..."))
    checks.append((saw.get("hover_row") == "none",
                   "cursor position hidden rather than blank"))

    ok = True
    for passed, msg in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {msg}")
        ok &= passed
    for msg in skipped:
        print(f"  SKIP  {msg}")
    tail = f"  ({len(skipped)} check could not run)" if skipped else ""
    print(f"\n{'GEO OK' if ok else 'GEO BROKEN'}{tail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
