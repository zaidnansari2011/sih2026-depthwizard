"""Prove the tripwires fire -- and prove they do not fire on a healthy run.

A guard nobody has seen trigger is not a guard. run05 spent 543.6 minutes producing
nothing because train.py watched for nothing at all; the aborts added afterwards are only
worth the launch decision they support if they have been observed doing their job.

Four cases, each a real train.py invocation on the known-good V2-Small config:

  control      healthy run, tripwires armed        -> must NOT abort  (exit 0)
  rail         log_var_max below the init sigma    -> must abort      (exit 2)
  nan          absurd lr, clipping off             -> must abort      (exit 2)
  gate         epoch-0 gate set unreachably high   -> must abort      (exit 2)

The control matters most. A tripwire that aborts everything is worse than none, because
it would have blocked run02 and run04 as well.

    python tools/tripwire_selftest.py            # ~10 min
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/sih2026/depthwizard")
PY = "D:/sih2026/.venv/Scripts/python.exe"
OUT = Path("D:/sih2026/checkpoints/tripwire_selftest")

# Small, fast, and the backbone we already know trains cleanly, so any abort is
# attributable to the case under test rather than to the model.
BASE = [
    "--batch", "2", "--accum", "1", "--epochs", "1",
    "--workers", "0", "--val-batches", "5", "--log-every", "20",
]

CASES = [
    # name, extra args, expected exit, what a pass means
    ("control", ["--max-steps", "60", "--warmup-mse", "10"], 0,
     "a healthy run is NOT aborted"),
    ("rail", ["--max-steps", "60", "--warmup-mse", "10",
              "--log-var-max", "2.0", "--rail-patience", "10"], 2,
     "sigma pinned at the clamp aborts (run05's fingerprint)"),
    ("nan", ["--max-steps", "80", "--warmup-mse", "0",
             "--lr", "10.0", "--clip", "0"], 2,
     "a non-finite loss aborts instead of running on"),
    ("gate", ["--max-steps", "60", "--warmup-mse", "10",
              "--gate-rmse", "0.001"], 2,
     "an epoch-0 trajectory worse than the gate aborts"),
]


def run(name: str, extra: list[str]) -> tuple[int, str, float]:
    env = dict(os.environ,
               PYTHONPATH=str(ROOT),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "train.py", *BASE, *extra, "--out", str(OUT / name)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=1800)
    return p.returncode, (p.stdout + p.stderr), time.time() - t0


def main() -> int:
    print(f"TRIPWIRE SELF-TEST  {time.strftime('%H:%M:%S')}")
    print(f"  thermal governor left at its default; these are short runs.\n")
    rows = []
    for name, extra, want, meaning in CASES:
        code, log, secs = run(name, extra)
        fired = "TRIPWIRE:" in log
        ok = (code == want) and (fired == (want == 2))
        why = ""
        for line in log.splitlines():
            if "TRIPWIRE:" in line or "    - " in line:
                why = line.strip()
                break
        rows.append((name, ok, code, want, secs, meaning, why))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<8} exit {code} (wanted {want})  "
              f"{secs/60:.1f} min  {meaning}")
        if why:
            print(f"           {why}")
        (OUT / f"{name}.log").parent.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.log").write_text(log, encoding="utf8", errors="replace")

    bad = [r for r in rows if not r[1]]
    print("\n" + "=" * 74)
    if bad:
        print(f"  {len(bad)} of {len(rows)} cases FAILED: "
              f"{', '.join(r[0] for r in bad)}")
        print("  Do not rely on the tripwires until these are understood.")
    else:
        print(f"  All {len(rows)} cases behaved as specified. The aborts fire on the")
        print("  three failure shapes run05 exhibited, and leave a healthy run alone.")
    print("=" * 74)
    print(f"  logs: {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
