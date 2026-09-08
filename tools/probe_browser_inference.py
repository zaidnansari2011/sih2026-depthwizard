"""Can the shipped model run in a browser, with no server doing the work?

This decides the hosting architecture. If onnxruntime-web can execute our int8 export at a
usable speed, the product becomes a static site: the upload runs on the visitor's own
machine, hosting is a free CDN, there is nothing to scale and nothing to pay for. If it
cannot, upload needs a real backend, with the cost and operations that implies.

Rather than reason about op coverage, this loads the actual 36.8 MB int8 model in headless
Chrome via onnxruntime-web and runs real 518x518 tiles through it.

Timing note: an earlier version used --virtual-time-budget to wait for the result, and
reported 0 ms per tile. Virtual time makes performance.now() advance on Chrome's schedule
rather than the clock, so every duration collapses. The page now POSTs its result back and
Python waits on real time; the browser's own numbers are then trustworthy too.

Cross-origin isolation headers are served so WASM threads are available, since a deployed
site would set them and single-threaded is not the number to plan around.

    python tools/probe_browser_inference.py
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = Path("D:/sih2026/out/depthwizard_run02.int8.onnx")
STAGE = Path("D:/sih2026/depthwizard/tools/_ortweb")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SIZE = 518
RUNS = 3
NATIVE_MS = 509          # docs/evidence-pack.md, native ONNX Runtime on CPU

RESULT: dict = {}
DONE = threading.Event()

PAGE = """<!doctype html><meta charset=utf-8><title>booting</title>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
<body><script>
(async () => {
  const send = (o) => fetch('/result', { method: 'POST', body: JSON.stringify(o) });
  try {
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = THREADS;
    const t0 = performance.now();
    const s = await ort.InferenceSession.create('./model.onnx',
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const loadMs = performance.now() - t0;
    const N = SIZEPX;
    const data = Float32Array.from({ length: 3 * N * N }, () => Math.random());
    const feeds = { image: new ort.Tensor('float32', data, [1, 3, N, N]) };
    const times = [];
    let out;
    for (let i = 0; i < RUNSN; i++) {
      const t = performance.now();
      out = await s.run(feeds);
      times.push(performance.now() - t);
    }
    const h = out.height_m;
    await send({ ok: true, loadMs: Math.round(loadMs),
                 times: times.map(Math.round), outDims: h.dims,
                 threads: ort.env.wasm.numThreads,
                 crossOriginIsolated: self.crossOriginIsolated === true,
                 sample: [h.data[0].toFixed(3), h.data[h.data.length - 1].toFixed(3)] });
  } catch (e) {
    await send({ ok: false, error: String((e && e.message) || e).slice(0, 400) });
  }
})();
</script></body>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        # Required for SharedArrayBuffer, which onnxruntime-web needs for WASM threads.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        RESULT.update(json.loads(self.rfile.read(n) or b"{}"))
        self.send_response(204)
        self.end_headers()
        DONE.set()


def run(threads: int) -> dict:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(MODEL, STAGE / "model.onnx")
    page = (PAGE.replace("THREADS", str(threads))
                .replace("SIZEPX", str(SIZE)).replace("RUNSN", str(RUNS)))
    (STAGE / "index.html").write_text(page, encoding="utf8")

    RESULT.clear()
    DONE.clear()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.directory = str(STAGE)
    Handler.directory = str(STAGE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--enable-features=SharedArrayBuffer",
         f"--user-data-dir={STAGE / 'prof'}", f"http://127.0.0.1:{port}/index.html"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    got = DONE.wait(timeout=420)
    wall = time.time() - t0
    proc.terminate()
    srv.shutdown()
    if not got:
        return {"ok": False, "error": f"no result within {wall:.0f}s"}
    return {**RESULT, "wallSeconds": round(wall, 1)}


def main() -> int:
    if not MODEL.exists():
        print(f"  no model at {MODEL}")
        return 1
    print(f"  model {MODEL.stat().st_size / 1e6:.1f} MB, tile {SIZE}x{SIZE}, "
          f"{RUNS} runs per configuration\n")
    best = None
    for threads in (1, 4):
        r = run(threads)
        if not r.get("ok"):
            print(f"  threads={threads}: FAILED -- {r.get('error')}")
            continue
        ts = r["times"]
        steady = min(ts[1:]) if len(ts) > 1 else ts[0]
        best = steady if best is None else min(best, steady)
        print(f"  threads={r['threads']:<2} isolated={r['crossOriginIsolated']}  "
              f"load {r['loadMs']:>5} ms   runs {ts} ms   steady {steady} ms   "
              f"(wall {r['wallSeconds']}s)")
        print(f"      out {r['outDims']}, sample {r['sample']}")
    if best is None:
        print("\n  VERDICT: the model does NOT run in a browser.")
        return 1
    print(f"\n  VERDICT: runs in-browser. Best {best} ms per {SIZE}px tile, "
          f"{best / NATIVE_MS:.1f}x native CPU ({NATIVE_MS} ms).")
    px = SIZE * SIZE
    print(f"  A 1024x1024 scene is ~{-(-1024 * 1024 // px)} tiles "
          f"-> roughly {best * -(-1024 * 1024 // px) / 1000:.1f} s in the tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
