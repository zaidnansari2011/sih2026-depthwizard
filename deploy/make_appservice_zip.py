# -*- coding: utf-8 -*-
"""Build the App Service zip: the CONTENTS of _context/app at the zip root.

deploy/dwz-app.zip is a Docker build context (root = Dockerfile + app/) and zip-deploying
it took the site down for two days on 8 Sep. This writes a differently named file so the
two can never be confused again. Oryx does none of what the Dockerfile does, so the
--extra-index-url that torch==2.11.0+cpu needs has to be inside requirements.txt itself.
"""
import zipfile
from pathlib import Path

CTX = Path("D:/sih2026/depthwizard/deploy/_context")
APP = CTX / "app"
OUT = Path("D:/sih2026/depthwizard/deploy/dwz-appservice.zip")
INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu\n"

req = INDEX + (CTX / "requirements-serve.txt").read_text(encoding="utf-8")

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in sorted(APP.rglob("*")):
        if f.is_file():
            z.writestr(zipfile.ZipInfo(f.relative_to(APP).as_posix()), f.read_bytes())
            n += 1
    z.writestr("requirements.txt", req)
    n += 1

print(f"  {n} files -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
print("  sanity: tools/serve_app.py at root ->", "tools/serve_app.py" in names)
print("          requirements.txt at root  ->", "requirements.txt" in names)
print("          no stray app/ prefix      ->", not any(x.startswith("app/") for x in names))
print("          no Dockerfile             ->", "Dockerfile" not in names)
print("  first line of requirements.txt:", req.splitlines()[0])
