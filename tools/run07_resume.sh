#!/usr/bin/env bash
# run07, resumed. The first launch (06:25 7 Sep) died at step 6,129/14,000 when the
# Claude Code session was torn down -- Start-Process did not detach the process tree.
# This script is registered as a Windows scheduled task instead, which parents to
# svchost and survives session exit.
#
# --resume, not --init-from: last.pt carries opt/sched/gstep/rng/scaler, so the LR
# schedule continues toward its zero at step 14,000 instead of restarting. Resuming a
# run's OWN checkpoint is what --resume is for; the trap the audit found was resuming a
# DIFFERENT run's checkpoint (run02 at epoch 11 into a 3-epoch run), and the guard in
# train.py still covers that case.

set -u
cd /d/sih2026/depthwizard || exit 90

PY=/d/sih2026/.venv/Scripts/python.exe
LOG=/d/sih2026/logs/run07.log
STATUS=/d/sih2026/logs/run07.status
CKPT=D:/sih2026/checkpoints/run07/best.pt

exec >>"$LOG" 2>&1
say() { echo "[$(date '+%H:%M:%S')] $*"; echo "$(date '+%H:%M:%S') $*" >"$STATUS"; }
fail() { say "STOP at $1 -- $2"; echo "$1" >/d/sih2026/logs/run07.failed; exit 1; }

rm -f /d/sih2026/logs/run07.failed
say "=== run07 RESUME from last.pt ==="

( while true; do
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null
    sleep 30
  done ) >>/d/sih2026/logs/run07_gpu_temp.log 2>/dev/null &
TEMPPID=$!

T0=$(date +%s)
"$PY" train.py \
    --shards D:/sih2026/data/shards_union \
    --resume D:/sih2026/checkpoints/run07/last.pt \
    --out D:/sih2026/checkpoints/run07 \
    --height-scale 13.223477220535276 \
    --epochs 4 --max-steps 14000 --batch 8 --workers 2 --max-temp 83
RC=$?
kill $TEMPPID 2>/dev/null
SECS=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && fail train "train.py exited $RC after ${SECS}s"
say "train done in $((SECS/60)) min"
[ -f "$CKPT" ] || fail train "no checkpoint at $CKPT"

# 1 -- matched protocol: directly comparable to out/eval_run02 and the pilot
say "eval 1/4: DFC2019 val, no TTA"
"$PY" tools/evaluate.py --ckpt "$CKPT" --split val --tiles 80 \
    --out D:/sih2026/out/eval_run07 || fail eval1 "no-TTA eval failed"
say "eval 1/4 done"

# 2 -- GAMUS DC holdout, ALL 317 tiles. The 80-tile baseline found only 20 buildings
# above 20 m, too few to resolve the tall-building question; 317 tiles gives ~4x that.
say "eval 2/4: GAMUS DC holdout, 317 tiles"
"$PY" tools/evaluate.py --ckpt "$CKPT" --corpus gamus --split gdc --tiles 317 \
    --out D:/sih2026/out/eval_run07_gamus_full || say "eval 2/4 FAILED (non-fatal)"

# 3 -- the same 317-tile holdout for run02, so the comparison is like-for-like.
say "eval 3/4: run02 on the same 317 holdout tiles"
"$PY" tools/evaluate.py --ckpt D:/sih2026/checkpoints/run02/best.pt \
    --corpus gamus --split gdc --tiles 317 \
    --out D:/sih2026/out/eval_run02_gamus_full || say "eval 3/4 FAILED (non-fatal)"

# 4 -- shipping protocol, comparable to run02's 6.401 m / 3.667 m
say "eval 4/4: DFC2019 val, TTA + zoom-2 fusion"
"$PY" tools/evaluate.py --ckpt "$CKPT" --split val --tiles 80 --tta --fuse-zoom 2 \
    --out D:/sih2026/out/eval_run07_ship || say "eval 4/4 FAILED (non-fatal)"

say "DONE: run07 trained and scored"
exit 0
