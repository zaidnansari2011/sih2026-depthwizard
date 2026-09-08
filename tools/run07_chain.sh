#!/usr/bin/env bash
# run07: the union training run, launched after all seven gates in docs/pilot-gates.md
# passed on the pilot. Trains, then scores in order of decision value so the numbers that
# matter most exist earliest if the run is ever cut short.
#
#   1. DFC2019 val, no TTA   -- directly comparable to out/eval_run02 and the pilot
#   2. GAMUS DC holdout      -- the tall-building instrument; DFC val has only 60
#                               buildings >20 m and cannot resolve this question
#   3. DFC2019 val, TTA+fused -- the shipping protocol, comparable to run02's 6.401 m
#
# Warm start is from run02, not from the pilot: the pilot's LR schedule already annealed
# to zero, and gamus-integration.md Phase 3 pre-registers run02 as the starting point.

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
say "=== run07 start ==="

# Baseline first (~5 min). run07's holdout number is uninterpretable without the
# DFC2019-only model scored on the same holdout -- that is the "DFC2019 only / GAMUS
# held-out city" cell of the §6 matrix. Doing it before the long run means it survives
# even if training or the later evals are cut short.
say "baseline: run02 on the GAMUS DC holdout"
"$PY" tools/evaluate.py --ckpt D:/sih2026/checkpoints/run02/best.pt \
    --corpus gamus --split gdc --tiles 80 \
    --out D:/sih2026/out/eval_run02_gamus_holdout || say "baseline FAILED (non-fatal)"
say "baseline done"

( while true; do
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null
    sleep 30
  done ) >/d/sih2026/logs/run07_gpu_temp.log 2>/dev/null &
TEMPPID=$!

T0=$(date +%s)
"$PY" train.py \
    --shards D:/sih2026/data/shards_union \
    --init-from D:/sih2026/checkpoints/run02/best.pt \
    --out D:/sih2026/checkpoints/run07 \
    --height-scale 13.223477220535276 \
    --epochs 4 --max-steps 14000 --batch 8 --workers 2 --max-temp 83
RC=$?
kill $TEMPPID 2>/dev/null
SECS=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && fail train "train.py exited $RC after ${SECS}s"
say "train done in $((SECS/60)) min"

[ -f "$CKPT" ] || fail train "no checkpoint at $CKPT"

# 1 -- matched protocol, the apples-to-apples comparison
say "eval 1/3: DFC2019 val, no TTA"
"$PY" tools/evaluate.py --ckpt "$CKPT" --split val --tiles 80 \
    --out D:/sih2026/out/eval_run07 || fail eval1 "no-TTA eval failed"
say "eval 1/3 done"

# 2 -- the instrument that can actually see tall buildings
say "eval 2/3: GAMUS DC holdout"
"$PY" tools/evaluate.py --ckpt "$CKPT" --corpus gamus --split gdc --tiles 80 \
    --out D:/sih2026/out/eval_run07_gamus_holdout || say "eval 2/3 FAILED (non-fatal, continuing)"

# 3 -- the shipping protocol, comparable to run02's 6.401 m / 3.667 m
say "eval 3/3: DFC2019 val, TTA + zoom-2 fusion"
"$PY" tools/evaluate.py --ckpt "$CKPT" --split val --tiles 80 --tta --fuse-zoom 2 \
    --out D:/sih2026/out/eval_run07_ship || say "eval 3/3 FAILED (non-fatal)"

say "DONE: run07 trained and scored"
exit 0
