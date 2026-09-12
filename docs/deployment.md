# Deploying DepthWizard publicly

The whole product — viewer, baked scenes, and the upload → DSM → 3D pipeline — runs on an
ordinary **CPU** web host. No GPU VM, no inference service, no GPU bill.

That is a measured claim, not an aspiration. With the GPU hidden from PyTorch, one
1024×1024 tile costs:

| Setting | Inference | Wall (incl. ~8 s model load) | Peak RSS |
|---|---|---|---|
| single pass, 9 windows | **5.0 s** | 13.3 s | 2.05 GB |
| ×8 D4 TTA + zoom-2 fusion | **57.2 s** | 86.2 s | 2.21 GB |

The second row is the configuration every published per-terrain number was measured with.
The first is what a visitor gets by default, because a minute of silence reads as a broken
page. The viewer is told which one produced the surface it is showing.

**Measured on the deployment itself**, 6 Sep, App Service B2 (2 vCPU, 3.5 GB), fast mode,
the same 1024×1024 tile: **60 s** from POST to a served scene — about 4.5× the desktop's
13.3 s, which is what two shared vCPUs against six local cores predicts. That is the
number to quote, not the desktop one.

Sixty seconds is usable for "try it yourself" and too slow to stand in front of. Demo from
a baked scene; offer upload as the thing a judge does afterwards.

## Why App Service and not Container Apps

Both were tried. On an **Azure for Students** subscription:

- **ACR Tasks is barred** (`TasksOperationsNotAllowed`), so `az acr build` cannot build an
  image in the cloud. Building locally instead needs a Docker daemon, and Docker Desktop
  on this machine needs WSL2, which is not installed.
- **Container App environments are capped per region**, and Central India's slot is
  already taken (`MaxNumberOfEnvironmentsInSubExceeded`).

App Service's Python runtime sidesteps both: Oryx installs the dependencies on the
platform from `requirements.txt`, so nothing local has to build or push an image. The
`deploy/Dockerfile` is kept anyway — it is correct, and it is the path back to Container
Apps on any subscription that permits ACR Tasks.

## Layout

    deploy/stage.py              assembles the deployable tree
    deploy/make_appservice_zip.py  packages _context/app for App Service
    deploy/Dockerfile            container path (unused on Students, kept working)
    deploy/requirements-serve.txt
    tools/serve_app.py           the public server

`serve_app.py` is a **separate entry point** from `serve_viewer.py` rather than a flag on
it. The local tool binds loopback, takes 400 MB uploads and trusts whoever started it;
those are the right defaults for a tool you run yourself and the wrong ones for a URL. Two
files means neither can silently become the other.

## Staging

    python deploy/stage.py

Three things dominate the size and each is handled:

- **The checkpoint is stripped, 297 MB → 99.2 MB.** `best.pt` carries optimiser state,
  scheduler state, RNG state and the training args — everything needed to *resume* a run
  and nothing needed to serve one. `epoch` and `val` are kept because `infer.py` prints
  them, and a served result that cannot name its checkpoint is not evidence.
- **The HF cache is baked in.** `model.py` builds the backbone with
  `from_pretrained()`, which reaches Hugging Face on a cache miss. `blobs/` is dropped
  because a HF cache stores every file twice — once under its sha, once as a snapshot link
  — and only the snapshot is resolved at load time. `HF_HUB_OFFLINE=1` turns any surviving
  download into a loud failure instead of a slow first request.
- **`upload_*` scenes are excluded.** They are produced at runtime and sit in the same
  directory as the baked ones, so without the filter the deployment ships whatever was on
  the dev machine. `index.json` is pruned to match, since an entry with no directory makes
  the viewer fetch a missing manifest and presents as the whole picker being broken.

Scenes are gitignored, so they exist only on the machine that generated them. `stage.py`
refuses to build a context with zero baked scenes rather than deploying an empty picker.

## Deploying

    python deploy/stage.py --ckpt ../checkpoints/run07/best.pt
    python deploy/make_appservice_zip.py
    az webapp deploy -g sih2026-depthwizard -n depthwizard-sih2026 \
                     --src-path dwz-appservice.zip --type zip --async true

> **`deploy/dwz-app.zip` is a Docker build context and must never be zip-deployed.**
> Its root is `Dockerfile` + `app/`, which is what the Dockerfile expects
> (`COPY app/ /app/` strips the prefix) and what Oryx cannot use. Deploying it on
> 8 Sep 2026 took the site down for two days. `make_appservice_zip.py` writes a
> **differently named** file, `dwz-appservice.zip`, whose root is the *contents* of
> `_context/app/` — so `tools/` sits at the root, matching the
> `python tools/serve_app.py` startup command — plus a `requirements.txt` that is
> `requirements-serve.txt` with one line prepended:
> `--extra-index-url https://download.pytorch.org/whl/cpu`. Nothing else: adding
> `--only-binary=:all:` (a Dockerfile flag) preceded a build that died inside
> `uv pip install`. The script checks the layout before writing, so a wrong-shaped
> zip fails here rather than on the platform.

**Measured 12 Sep 2026**, restaging to run07: 336 MB, `--async true`, **815 s** end
to end, reporting `status: RuntimeSuccessful` and `numberOfInstancesSuccessful: 1`.
The old container keeps serving throughout the upload, so the site does not go dark.

**Verify a restage by what it serves, not by what the deploy says.** An upload returns
a scene whose `manifest.json` carries no model name, so compare its `height_max_m` and
`sigma_mean_m` against a local single-pass run of each candidate checkpoint. On
`OMA_288_042` that separates them cleanly: run07 gives 38.3 m / σ 1.54, run02 gives
29.6 m / σ 0.87. Note also that the viewer is served at the site **root** —
`/viewer/main.js` is a 404 and the script is `/main.js`, which reads exactly like a
failed deploy if you check the wrong path.

The zip is built at compression level 1 on purpose: the bulk is `.pt`, `.safetensors`,
`.bin` and `.jpg`, none of which compress, so level 9 buys a percent or two for several
minutes.

### Settings that matter

| Setting | Value | Why |
|---|---|---|
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` | run Oryx; without it nothing is installed |
| `HF_HOME` | `/home/site/wwwroot/hf` | the baked cache |
| `HF_HUB_OFFLINE` | `1` | never call HF at request time |
| `DW_WORK` | `/tmp/depthwizard` | uploads and raw output on local disk, not the slow share |
| `DW_CKPT` | `/home/site/wwwroot/weights/best.pt` | |
| `DW_QUALITY` | `fast` | default; clients may ask for `accurate` |
| `OMP_NUM_THREADS` | `2` | matches the plan's cores |
| `WEBSITES_CONTAINER_START_TIME_LIMIT` | `1800` | loading torch is slow on first boot |

Startup command: `python tools/serve_app.py`.

`requirements.txt` declares `--extra-index-url https://download.pytorch.org/whl/cpu` in
the file itself, because Oryx runs a plain `pip install -r requirements.txt` and never
sees command-line flags. Without it pip resolves the default CUDA wheel — ~2.5 GB of
libraries this plan has no GPU to use.

## Limits the public server enforces

Not decoration: one request costs CPU-minutes, so an unbounded queue is a way to take the
site down by being mildly popular.

| Guard | Default | Env |
|---|---|---|
| upload size | 50 MB | `DW_MAX_UPLOAD_MB` |
| **image area** | **~13.8 Mpx (≈3713²) in fast mode** | derived — see below |
| jobs in flight | 3, then 429 | `DW_MAX_QUEUE` |
| per-IP spacing | 20 s | `DW_IP_SPACING` |
| uploaded scenes kept | 12, newest first | `DW_SCENE_BUDGET` |
| job abandoned after | 600 s | `DW_TIMEOUT` |

### Why an area cap, when there is already a byte cap

Because the two are barely related for satellite imagery, and only one of them predicts
cost. A Maxar visual strip is **17408×17408 — 303 Mpx — in 12 MB**, sparse and
JPEG-compressed inside the TIFF. It passes a 50 MB byte cap comfortably and then needs
about **2.8 hours**, so the job used to sit there and time out at 600 s having told the
user nothing in the meantime.

The cap is *derived*, not a magic number: `FIXED_S + mpx × SEC_PER_MPX` against
`BUDGET_FRAC × DW_TIMEOUT`. The constants are the measurements — 25 s fixed (checkpoint
load, export, four array writes) and 33 s per megapixel, from the 1024×1024 upload that
takes 60 s on this plan. So the limit moves correctly if the timeout or the plan changes,
and `TTA_FACTOR` (11.4, from 57.2 s ÷ 5.02 s) drops the ceiling to ~1.2 Mpx in accurate
mode, where it belongs.

The refusal is immediate and specific — it names the dimensions, the projected time, and
the size to crop to. Reading the header also catches a **truncated or still-downloading
GeoTIFF** in the same millisecond, instead of forty seconds into inference with a
`TIFFReadEncodedTile` traceback.

A refusal does **not** consume the caller's per-IP turn; only an accepted job does.
`/api/capabilities` publishes `max_megapixels` and `max_side_px` so a client can warn
before spending the upload.

Baked scenes are never reaped. Per-IP spacing reads `X-Forwarded-For`, which is
client-supplied and therefore spoofable — that is fine, because it only paces polite users
apart and the queue cap is what actually bounds the work.

## Custom domain

`project5.zaidansari.tech`, on Cloudflare DNS:

| Type | Name | Value | Proxy |
|---|---|---|---|
| CNAME | `project5` | `depthwizard-sih2026.azurewebsites.net` | **DNS only** |
| TXT | `asuid.project5` | the app's `customDomainVerificationId` | — |

    az webapp config hostname add    -g <rg> --webapp-name <app> --hostname project5.zaidansari.tech
    az webapp config ssl create      -g <rg> --name <app> --hostname project5.zaidansari.tech

The grey cloud is required *while the certificate is issued*: Azure's free managed
certificate validates by resolving the CNAME to App Service, and Cloudflare's proxy breaks
that check. Once issued, the proxy can be turned on with SSL mode **Full (strict)**.

## Known gaps

- **The model reloads per request**, costing ~8 s of every upload, because `run_job`
  shells out to `infer.py` exactly as the local tool does. Holding it resident means
  reimplementing the auto-zoom, fusion and DEM logic in `infer.py`'s `main()`, and
  duplicating the inference path is how a served result quietly stops matching the
  measured one. It stays until `infer.py` exposes a reusable entry point.
- **DEM anchoring is off by default** (`DW_DEM=1` to enable). `--dem auto` reaches
  Copernicus during a request that is already slow.
- **Uploaded scenes survive `az webapp restart`** — measured twice, 6 Sep, contrary to the
  assumption written here first. A restart reuses the extracted tree under `/tmp/<hash>/`,
  so runtime scenes persist. What *does* clear them is anything that recreates the
  container — an SSL bind or a redeploy — because the platform extracts a fresh tree and
  `stage.py` excludes `upload_*`.

  So `DW_SCENE_BUDGET` is the only routine bound on disk growth, and a visitor's upload
  stays publicly listed until twelve more push it out. There is no delete endpoint; if a
  specific scene must go before then, redeploy.
- **`infer.py --help` crashes** — a `%` in one help string breaks argparse's formatting.
  Cosmetic, but it means `--help` is not a way to discover the flags.
