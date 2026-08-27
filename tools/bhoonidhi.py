"""Bhoonidhi (NRSC) API client: authenticate, list collections, search, download.

Credentials never live in this repo. Put them in D:/sih2026/.bhoonidhi.json, which sits
outside the git tree:

    {"userId": "your_bhoonidhi_username", "password": "your_password"}

or set BHOONIDHI_USER and BHOONIDHI_PASS in the environment.

Tokens are cached to D:/sih2026/.bhoonidhi_token.json and reused until they expire. That is
not an optimisation: the auth endpoint allows 20 requests per hour per IP, and the API docs
say plainly "Do not fetch a new token for each API request you make." Search is capped at
3 requests per second and downloads at 3 concurrent per user.

    python tools/bhoonidhi.py auth
    python tools/bhoonidhi.py collections
    python tools/bhoonidhi.py collections --grep Carto
    python tools/bhoonidhi.py search --collection CartoSat-1_PAN_CartoDEM_30m \\
        --bbox 77.4 12.8 77.8 13.2 --limit 10

Two date rules the API enforces that the spec does not spell out: datetime needs
full ISO timestamps ("2025-09-01T00:00:00Z/2026-08-25T00:00:00Z", not bare dates),
and the range must be 366 days or less. Both come back as HTTP 406.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://bhoonidhi-api.nrsc.gov.in"
CRED_PATH = ROOT / ".bhoonidhi.json"
TOKEN_PATH = ROOT / ".bhoonidhi_token.json"
UA = "DepthWizard/0.1 (SIH2026; research)"


def _post(url: str, payload: dict, token: str | None = None, timeout: int = 60):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, token: str | None = None, timeout: int = 60):
    headers = {"User-Agent": UA}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_credentials() -> dict:
    user, pw = os.environ.get("BHOONIDHI_USER"), os.environ.get("BHOONIDHI_PASS")
    if user and pw:
        return {"userId": user, "password": pw}
    if CRED_PATH.exists():
        c = json.loads(CRED_PATH.read_text())
        if c.get("userId") and c.get("password"):
            return {"userId": c["userId"], "password": c["password"]}
    raise SystemExit(
        f"No credentials. Create {CRED_PATH} containing\n"
        '  {"userId": "your_bhoonidhi_username", "password": "your_password"}\n'
        "or set BHOONIDHI_USER and BHOONIDHI_PASS. That path is outside the git tree, so\n"
        "it cannot be committed by accident."
    )


def get_token(force: bool = False) -> str:
    """Cached bearer token. Refreshed only when expired, to respect 20 auth req/hour."""
    if not force and TOKEN_PATH.exists():
        try:
            t = json.loads(TOKEN_PATH.read_text())
            if t.get("access_token") and time.time() < t.get("expires_at", 0) - 60:
                return t["access_token"]
        except Exception:
            pass
    cred = load_credentials()
    cred["grant_type"] = "password"
    try:
        r = _post(f"{BASE}/auth/token", cred)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise SystemExit(
            f"auth failed: HTTP {e.code}\n{detail}\n\n"
            "401/403 here means the credentials are wrong, or this account has portal\n"
            "access but not API access. The Applications page says to request it from\n"
            "bhoonidhi@nrsc.gov.in, quoting your registered user id."
        )
    tok = r.get("access_token")
    if not tok:
        raise SystemExit(f"auth returned no access_token: {json.dumps(r)[:400]}")
    TOKEN_PATH.write_text(json.dumps({
        "access_token": tok, "refresh_token": r.get("refresh_token"),
        "expires_at": time.time() + float(r.get("expires_in", 3600)),
        "userId": r.get("userId"),
    }, indent=2))
    return tok


def cmd_auth(args):
    tok = get_token(force=args.force)
    meta = json.loads(TOKEN_PATH.read_text())
    left = (meta["expires_at"] - time.time()) / 60.0
    print(f"authenticated as {meta.get('userId')}")
    print(f"  token ...{tok[-12:]}  valid for {left:.0f} more minutes")
    print(f"  cached at {TOKEN_PATH}")


def cmd_collections(args):
    data = _get(f"{BASE}/data/collections", get_token())
    cols = data.get("collections", data if isinstance(data, list) else [])
    rows = []
    for c in cols:
        cid = c.get("id") if isinstance(c, dict) else str(c)
        title = (c.get("title") or "") if isinstance(c, dict) else ""
        if args.grep and args.grep.lower() not in f"{cid} {title}".lower():
            continue
        rows.append((cid, title))
    print(f"{len(rows)} collections" + (f" matching {args.grep!r}" if args.grep else ""))
    for cid, title in rows:
        print(f"  {cid:44s} {title[:60]}")
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"\nwrote {args.out}")


def cmd_search(args):
    body: dict = {"limit": args.limit}
    if args.collection:
        body["collections"] = args.collection
    if args.bbox:
        body["bbox"] = [str(v) for v in args.bbox]
    if args.datetime:
        body["datetime"] = args.datetime
    data = _post(f"{BASE}/data/search", body, get_token())
    feats = data.get("features", [])
    print(f"{len(feats)} results (limit {args.limit})")
    for f in feats:
        p = f.get("properties", {})
        print(f"  {f.get('id','?')}")
        keep = {k: v for k, v in p.items()
                if k in ("datetime", "collection", "gsd", "platform", "instrument",
                         "eo:cloud_cover", "product_level", "price", "isOpenData")}
        if keep:
            print(f"      {keep}")
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"\nwrote {args.out}")


def cmd_download(args):
    """Fetch one item.

    The STAC assets on a Bhoonidhi item only ever expose metadata and a thumbnail, so the
    raster itself comes from /download rather than from an asset href. The endpoint can
    answer three different ways and they are easy to confuse: the bytes, a JSON order
    acknowledgement for anything with Online: N, or an error. Content-Type decides which,
    and a JSON body is NOT a failure -- it means the scene has to be staged first.
    """
    token = get_token()
    url = (f"{BASE}/download?id={urllib.parse.quote(args.id)}"
           f"&collection={urllib.parse.quote(args.collection)}")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Authorization": f"Bearer {token}"})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            disp = r.headers.get("Content-Disposition") or ""
            name = args.name
            if not name and "filename=" in disp:
                name = disp.split("filename=")[-1].strip().strip('";')
            if not name:
                name = args.id

            if "json" in ctype:
                body = json.loads(r.read().decode(errors="replace"))
                print("server returned JSON rather than bytes:")
                print(json.dumps(body, indent=2)[:1500])
                print("\nFor a scene with Online: N this is the expected reply -- it has "
                      "to be ordered and staged before the bytes exist.")
                return

            dest = out_dir / name
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r  {got/1e6:.1f} / {total/1e6:.1f} MB", end="", flush=True)
            print()
            print(f"wrote {dest}  ({got/1e6:.1f} MB, {ctype or 'unknown type'})")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:600]
        raise SystemExit(f"download failed: HTTP {e.code}\n{detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("auth", help="test credentials, cache a token")
    a.add_argument("--force", action="store_true", help="ignore the cached token")
    a.set_defaults(func=cmd_auth)

    c = sub.add_parser("collections", help="list available collections")
    c.add_argument("--grep", help="case-insensitive substring filter")
    c.add_argument("--out")
    c.set_defaults(func=cmd_collections)

    s = sub.add_parser("search", help="search a collection")
    s.add_argument("--collection", nargs="+")
    s.add_argument("--bbox", nargs=4, type=float,
                   metavar=("W", "S", "E", "N"))
    s.add_argument("--datetime", help="e.g. 2020-01-01/2026-08-27")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--out")
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("download", help="fetch one item by id")
    d.add_argument("--id", required=True)
    d.add_argument("--collection", required=True)
    d.add_argument("--out", default=str(ROOT / "data" / "bhoonidhi"))
    d.add_argument("--name", help="output filename; default is the item id")
    d.set_defaults(func=cmd_download)

    args = ap.parse_args()
    try:
        args.func(args)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')[:600]}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as e:
        raise SystemExit(f"network error: {e.reason}")


if __name__ == "__main__":
    main()
