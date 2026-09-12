# -*- coding: utf-8 -*-
"""Build the App Service zip: the CONTENTS of _context/app at the zip root.

deploy/dwz-app.zip is a Docker build context (root = Dockerfile + app/) and zip-deploying
it took the site down for two days on 8 Sep. This writes a differently named file so the
two can never be confused again. Oryx does none of what the Dockerfile does, so the
--extra-index-url that torch==2.11.0+cpu needs has to be inside requirements.txt itself.
"""
import zipfile
from pathlib import Path

ROOT = Path("D:/sih2026/depthwizard")
CTX = ROOT / "deploy/_context"
APP = CTX / "app"
OUT = ROOT / "deploy/dwz-appservice.zip"
INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu\n"

# The staging tree is a copy of files that also live in the repo, and a copy goes stale in
# silence. Measured 12 Sep 2026: the deployed viewer was missing the Maxar CC-BY imagery
# credit for exactly this reason -- the repo had it, the live site did not, and nothing
# anywhere would have said so. A CC-BY credit that does not reach the deployment is a
# licence problem, not a cosmetic one.
#
# So the repo is canonical and this refreshes from it, rather than trusting whoever last
# remembered to copy. Files with no counterpart in the repo (the weights, the HF cache) are
# staging-only by design and are left alone.
stale = []
for f in sorted(APP.rglob("*")):
    if not f.is_file():
        continue
    src = ROOT / f.relative_to(APP)
    if src.is_file() and src.read_bytes() != f.read_bytes():
        f.write_bytes(src.read_bytes())
        stale.append(f.relative_to(APP).as_posix())
if stale:
    print(f"  refreshed {len(stale)} stale file(s) from the repo:")
    for s in stale:
        print(f"    {s}")
else:
    print("  staging tree already matches the repo")

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
