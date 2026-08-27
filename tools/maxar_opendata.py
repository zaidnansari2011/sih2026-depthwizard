"""Maxar Open Data: free 0.3-0.6 m imagery, including India, under CC-BY-4.0.

Why this matters to us. Bhoonidhi's best optical is 5.8 m against our 0.3 m training data,
a 19x gap that makes it a different problem rather than a transfer test. Maxar release
disaster-response imagery at 0.3-0.6 m, which is within 2x of our training GSD, and the
catalogue includes India-Floods-Oct-2023 over the Teesta valley in Sikkim.

Sikkim is mountains. That reads like a drawback and is close to the opposite: PLAN section
2 requires stability across urban, sparse, hilly and forested terrain, and DFC2019 is
Jacksonville and Omaha -- two flat American cities. We have no hilly terrain at all, in
training or in test. This is the only free source that fills that hole at a resolution our
model can actually consume.

No height truth comes with it, so it cannot score accuracy. It is a qualitative robustness
demonstration on terrain we otherwise cannot show, which is worth saying plainly rather
than dressing up.

    python tools/maxar_opendata.py events
    python tools/maxar_opendata.py items --event India-Floods-Oct-2023
    python tools/maxar_opendata.py download --event India-Floods-Oct-2023 --top 4
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[2]
CATALOG = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
UA = "DepthWizard/0.1 (SIH2026; research)"
OUT_DIR = ROOT / "data" / "maxar"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def event_url(event: str) -> str:
    return f"https://maxar-opendata.s3.amazonaws.com/events/{event}/collection.json"


def iter_items(event: str, limit_acq: int = 0):
    """Every ARD item in an event, following the relative hrefs the catalogue uses."""
    cu = event_url(event)
    coll = get_json(cu)
    acqs = [urljoin(cu, l["href"]) for l in coll.get("links", []) if l.get("rel") == "child"]
    if limit_acq:
        acqs = acqs[:limit_acq]
    for au in acqs:
        try:
            ac = get_json(au)
        except urllib.error.HTTPError:
            continue
        for l in ac.get("links", []):
            if l.get("rel") != "item":
                continue
            iu = urljoin(au, l["href"])
            try:
                yield iu, get_json(iu)
            except urllib.error.HTTPError:
                continue


def score(props: dict) -> tuple:
    """Rank items for our purposes: near-nadir, cloud-free, and covering real ground.

    Off-nadir first because it is the only one of the three we have actually measured a
    sensitivity to, and least-oblique is closest to how DFC2019 was captured.
    """
    return (
        float(props.get("view:off_nadir", 99)),
        float(props.get("eo:cloud_cover", 100) or 100),
        -float(props.get("tile:data_area", 0) or 0),
    )


def cmd_events(args):
    d = get_json(CATALOG)
    names = []
    for l in d.get("links", []):
        if l.get("rel") == "child" and l.get("href"):
            names.append(l["href"].strip("./").split("/")[0])
    if args.grep:
        names = [n for n in names if args.grep.lower() in n.lower()]
    print(f"{len(names)} events")
    for n in sorted(names):
        print("  ", n)


def cmd_items(args):
    rows = []
    for iu, it in iter_items(args.event, args.max_acquisitions):
        p = it.get("properties", {})
        rows.append({
            "url": iu, "id": it.get("id"), "bbox": it.get("bbox"),
            "gsd": p.get("gsd"), "off_nadir": p.get("view:off_nadir"),
            "cloud": p.get("eo:cloud_cover"), "area_km2": p.get("tile:data_area"),
            "datetime": p.get("datetime"), "platform": p.get("platform"),
            "epsg": p.get("proj:epsg"),
            "visual": urljoin(iu, (it.get("assets", {}).get("visual") or {}).get("href", "")),
        })
    rows.sort(key=lambda r: score({"view:off_nadir": r["off_nadir"] or 99,
                                   "eo:cloud_cover": r["cloud"],
                                   "tile:data_area": r["area_km2"]}))
    print(f"{len(rows)} items in {args.event}, best first "
          f"(near-nadir, cloud-free, most ground covered)")
    print(f"  {'gsd':>5} {'off-nadir':>9} {'cloud':>6} {'km2':>6}  {'platform':8} {'date':10}  id")
    for r in rows[: args.limit]:
        print(f"  {r['gsd'] or 0:5.2f} {r['off_nadir'] or 0:9.1f} "
              f"{(r['cloud'] if r['cloud'] is not None else -1):6.1f} "
              f"{r['area_km2'] or 0:6.1f}  {str(r['platform']):8} "
              f"{str(r['datetime'])[:10]:10}  {r['id']}")
    out = Path(args.out) if args.out else OUT_DIR / f"{args.event}_items.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as fh:
        total, got = int(r.headers.get("Content-Length") or 0), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got/1e6:6.1f} / {total/1e6:.1f} MB", end="", flush=True)
    print()
    return got


def cmd_download(args):
    cache = Path(args.items) if args.items else OUT_DIR / f"{args.event}_items.json"
    if not cache.exists():
        raise SystemExit(f"{cache} missing. Run `items --event {args.event}` first.")
    rows = json.loads(cache.read_text())
    rows = [r for r in rows if r.get("visual")]
    if args.max_off_nadir:
        rows = [r for r in rows if (r.get("off_nadir") or 99) <= args.max_off_nadir]
    if args.id:
        rows = [r for r in rows if any(i in r["id"] for i in args.id)]
        if not rows:
            raise SystemExit(f"no item matched {args.id}")
    elif args.min_lat is not None:
        rows = [r for r in rows if r["bbox"][1] >= args.min_lat]
    if args.max_lat is not None:
        rows = [r for r in rows if r["bbox"][1] <= args.max_lat]
    rows = rows[: args.top]
    print(f"downloading {len(rows)} visual tiles to {OUT_DIR / args.event}")
    for r in rows:
        name = r["id"].replace("/", "_") + "-visual.tif"
        dest = OUT_DIR / args.event / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {name} already present, skipping")
            continue
        print(f"  {name}  (gsd {r['gsd']}, off-nadir {r['off_nadir']})")
        download(r["visual"], dest)
    print(f"\ndone -> {OUT_DIR / args.event}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("events", help="list open-data events")
    e.add_argument("--grep")
    e.set_defaults(func=cmd_events)

    i = sub.add_parser("items", help="survey the tiles in one event")
    i.add_argument("--event", required=True)
    i.add_argument("--limit", type=int, default=25)
    i.add_argument("--max-acquisitions", type=int, default=0, help="0 = all")
    i.add_argument("--out")
    i.set_defaults(func=cmd_items)

    d = sub.add_parser("download", help="fetch the best visual tiles")
    d.add_argument("--event", required=True)
    d.add_argument("--top", type=int, default=4)
    d.add_argument("--max-off-nadir", type=float, default=None)
    d.add_argument("--id", nargs="+", help="substring match on item id")
    d.add_argument("--min-lat", type=float, help="southern tiles sit lower and carry "
                                                 "settlement; northern ones are glacier")
    d.add_argument("--max-lat", type=float)
    d.add_argument("--items", help="items json; defaults to the cached survey")
    d.set_defaults(func=cmd_download)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
