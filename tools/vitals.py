"""Machine vitals watchdog. Logs GPU, RAM and disk continuously; shouts only when it matters.

    python tools/vitals.py                     # run forever, log + warn
    python tools/vitals.py --once              # single reading, for a quick check
    python tools/vitals.py --interval 30       # sampling period, seconds
    python tools/vitals.py --summary           # summarise the existing log and exit

Why this exists
---------------
Two failures in this project were invisible until after the damage: a training run that
sat at 91 C for ten minutes, and one that was OOM-killed with no traceback and no
checkpoint. Both were detectable from numbers nobody was reading. So the numbers get read
on a timer, written to a CSV that survives the session, and surfaced only on breach --
a monitor that prints constantly is a monitor that gets ignored.

Thresholds are deliberately below the danger line, because the point is to intervene
before the hardware has to.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOG = Path("D:/sih2026/logs/vitals.csv")

# Warn well before anything is actually at risk. TjMax on a 3060 is 93 C; by 86 the card
# is already down-clocking to protect itself, so that is where a human should hear about
# it. RAM is the one that killed a run outright, so it warns early.
WARN_TEMP = 86.0
CRIT_TEMP = 90.0
WARN_RAM_GB = 1.5
CRIT_RAM_GB = 0.8
WARN_DISK_GB = 8.0


def gpu() -> dict:
    q = ("temperature.gpu,fan.speed,power.draw,clocks.sm,utilization.gpu,"
         "memory.used,memory.total")
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"temp": float(v[0]), "fan": float(v[1]), "power": float(v[2]),
                "clock": float(v[3]), "util": float(v[4]),
                "vram_used": float(v[5]), "vram_total": float(v[6])}
    except Exception:
        return {}


def ram() -> dict:
    class S(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        st = S()
        st.dwLength = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return {"ram_free": st.ullAvailPhys / 2**30, "ram_total": st.ullTotalPhys / 2**30}
    except Exception:
        return {}


def disk(drive: str = "D:\\") -> dict:
    try:
        free = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(drive), None, None, ctypes.byref(free))
        return {"disk_free": free.value / 2**30}
    except Exception:
        return {}


def training_alive() -> int:
    """Count python processes running train.py. Zero during a run means it died."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*train.py*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip() or 0)
    except Exception:
        return -1


def sample() -> dict:
    d = {"ts": datetime.now().isoformat(timespec="seconds")}
    d.update(gpu())
    d.update(ram())
    d.update(disk())
    d["train_procs"] = training_alive()
    return d


def check(s: dict) -> list[str]:
    out = []
    t = s.get("temp")
    if t is not None:
        if t >= CRIT_TEMP:
            out.append(f"CRITICAL GPU {t:.0f}C (fan {s.get('fan',0):.0f}%, "
                       f"{s.get('power',0):.0f}W, {s.get('clock',0):.0f}MHz)")
        elif t >= WARN_TEMP:
            out.append(f"WARN GPU {t:.0f}C (fan {s.get('fan',0):.0f}%, {s.get('power',0):.0f}W)")
    r = s.get("ram_free")
    if r is not None:
        if r <= CRIT_RAM_GB:
            out.append(f"CRITICAL RAM {r:.2f} GB free — a run will be OOM-killed shortly")
        elif r <= WARN_RAM_GB:
            out.append(f"WARN RAM {r:.2f} GB free")
    d = s.get("disk_free")
    if d is not None and d <= WARN_DISK_GB:
        out.append(f"WARN disk D: {d:.1f} GB free")
    return out


def fmt(s: dict) -> str:
    return (f"{s['ts']}  GPU {s.get('temp',0):.0f}C fan {s.get('fan',0):.0f}% "
            f"{s.get('power',0):.0f}W {s.get('clock',0):.0f}MHz util {s.get('util',0):.0f}%  "
            f"VRAM {s.get('vram_used',0)/1024:.1f}/{s.get('vram_total',0)/1024:.1f}GB  "
            f"RAM {s.get('ram_free',0):.1f}GB free  disk {s.get('disk_free',0):.0f}GB  "
            f"train={s.get('train_procs','?')}")


def write_row(s: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        if new:
            w.writeheader()
        w.writerow(s)


def summarise():
    if not LOG.exists():
        print("no log yet")
        return
    rows = list(csv.DictReader(LOG.open(encoding="utf-8")))
    if not rows:
        print("log is empty")
        return
    temps = [float(r["temp"]) for r in rows if r.get("temp")]
    rams = [float(r["ram_free"]) for r in rows if r.get("ram_free")]
    print(f"{len(rows)} samples, {rows[0]['ts']} -> {rows[-1]['ts']}")
    if temps:
        over = sum(1 for t in temps if t >= WARN_TEMP)
        print(f"  GPU temp: mean {sum(temps)/len(temps):.1f}C, peak {max(temps):.0f}C, "
              f"{over} samples >= {WARN_TEMP:.0f}C ({100*over/len(temps):.1f}%)")
    if rams:
        print(f"  RAM free: min {min(rams):.2f} GB, mean {sum(rams)/len(rams):.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only warnings")
    ap.add_argument("--every", type=int, default=10,
                    help="print a routine status line every N samples")
    args = ap.parse_args()

    if args.summary:
        return summarise()

    n = 0
    while True:
        s = sample()
        write_row(s)
        warns = check(s)
        for w in warns:
            print(f"!! {w}", flush=True)
        if not args.quiet and not warns and n % args.every == 0:
            print(fmt(s), flush=True)
        n += 1
        if args.once:
            if args.quiet and not warns:
                print(fmt(s), flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
