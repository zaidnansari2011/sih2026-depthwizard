"""Thermal soak test: answers 'will my 3060 cook itself on an overnight run?'

Puts a sustained ML-shaped load on the GPU and logs temperature, power and clocks
until the temperature plateaus. GPUs reach thermal equilibrium in 10-15 min, so a
20 min result predicts an 8 hour run.

    python tools/thermal_soak.py --minutes 20
    python tools/thermal_soak.py --minutes 20 --power-limit 130
"""
import argparse, csv, subprocess, time, sys

SMI = "nvidia-smi"
FIELDS = ["temperature.gpu", "utilization.gpu", "power.draw", "clocks.sm", "clocks.mem"]


def sample():
    out = subprocess.check_output(
        [SMI, f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return [float(x.strip()) for x in out.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--power-limit", type=int, default=None,
                    help="watts; needs admin. 3060 default is 170")
    ap.add_argument("--out", default="thermal_soak.csv")
    args = ap.parse_args()

    if args.power_limit:
        r = subprocess.run([SMI, "-pl", str(args.power_limit)], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode:
            print("  -> power limit NOT applied (run PowerShell as Administrator)")

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available to PyTorch")
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | soaking {args.minutes:.0f} min\n")

    # ViT-ish sustained matmul load in bf16 -- same tensor-core path training uses.
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)

    t0 = time.time()
    deadline = t0 + args.minutes * 60
    rows, next_log, peak = [], t0, 0.0
    print(f"{'min':>6} {'temp':>6} {'util':>6} {'watts':>7} {'sm_mhz':>8}")
    while time.time() < deadline:
        for _ in range(50):
            a @ b
        torch.cuda.synchronize()
        now = time.time()
        if now >= next_log:
            temp, util, watts, sm, mem = sample()
            peak = max(peak, temp)
            mins = (now - t0) / 60
            rows.append([round(mins, 2), temp, util, watts, sm, mem])
            print(f"{mins:6.1f} {temp:5.0f}C {util:5.0f}% {watts:6.1f}W {sm:8.0f}")
            next_log = now + 15

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minutes", "temp_c", "util_pct", "power_w", "sm_mhz", "mem_mhz"])
        w.writerows(rows)

    tail = [r[1] for r in rows[-8:]]
    plateau = sum(tail) / len(tail)
    print(f"\npeak {peak:.0f}C | plateau {plateau:.0f}C | log -> {args.out}")
    if plateau < 78:
        print("VERDICT: healthy. Overnight runs are fine as-is.")
    elif plateau < 84:
        print("VERDICT: warm but safe. Consider --power-limit 130 for unattended runs.")
    else:
        print("VERDICT: hot. Apply --power-limit 130 and check case airflow / dust.")
    print("Card's own limits: target 83C, max-operating 93C, slowdown 95C, shutdown 98C.")


if __name__ == "__main__":
    main()
