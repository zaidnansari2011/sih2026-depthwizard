#!/usr/bin/env bash
# Overnight chain: wait for the GAMUS ingest, then holdout -> union -> pilot -> score.
#
# Deliberately stops after scoring the pilot. The run07 decision is gated on
# docs/pilot-gates.md and is not made by this script.
#
# Every stage appends to logs/overnight.log and writes a one-line status to
# logs/overnight.status so progress is readable without parsing the full log.

set -u
cd /d/sih2026/depthwizard || exit 90

PY=/d/sih2026/.venv/Scripts/python.exe
LOG=/d/sih2026/logs/overnight.log
STATUS=/d/sih2026/logs/overnight.status
INGEST_PID=32736
MANIFEST=D:/sih2026/data/shards_gamus/manifest.json

exec >>"$LOG" 2>&1

say() { echo "[$(date '+%H:%M:%S')] $*"; echo "$(date '+%H:%M:%S') $*" >"$STATUS"; }
tiles() { "$PY" -c "import json;print(len(json.load(open('$MANIFEST'))['tiles']))" 2>/dev/null || echo 0; }
alive() { tasklist //FI "PID eq $INGEST_PID" 2>/dev/null | grep -q "$INGEST_PID"; }

fail() { say "STOP at $1 -- $2"; echo "$1" >/d/sih2026/logs/overnight.failed; exit 1; }

say "=== chain start, ingest at $(tiles) tiles ==="
rm -f /d/sih2026/logs/overnight.failed

# ---------------------------------------------------------------- A: wait for ingest
last=$(tiles); stall=0
while alive; do
    sleep 120
    now=$(tiles)
    if [ "$now" = "$last" ]; then
        stall=$((stall + 1))
        # 20 min of no new tiles while the process is still up = wedged, not slow.
        [ "$stall" -ge 10 ] && fail A "ingest wedged at $now tiles for 20 min (pid $INGEST_PID alive)"
    else
        stall=0; last=$now
    fi
done
FINAL=$(tiles)
say "A: ingest process exited, $FINAL tiles"
# A crash 10% in looks identical to success from the outside. Demand most of the corpus.
[ "$FINAL" -lt 5500 ] && fail A "ingest exited early with only $FINAL / 6204 tiles"

# ---------------------------------------------------------------- B: holdout GeoTIFFs
say "B: downloading held-out DC block"
"$PY" tools/prepare_gamus.py holdout --holdout-col 50 || fail B "holdout download failed"
say "B: holdout done, $(ls /d/sih2026/data/extracted_gamus 2>/dev/null | wc -l) files"

# ---------------------------------------------------------------- C: union directory
say "C: union dry run"
"$PY" tools/link_union.py --check || fail C "link_union --check failed"
"$PY" tools/link_union.py || fail C "link_union failed"

# G2 evidence: the exact val set train.py will glob out of the union directory.
say "C: union built"
echo "--- G2: val shards in shards_union ---"
ls /d/sih2026/data/shards_union/val_*.npz 2>/dev/null | wc -l
echo "--- G2: any GAMUS shard matching val_* (must be 0) ---"
ls /d/sih2026/data/shards_union/val_*.npz 2>/dev/null | grep -c "_g" || true
echo "--- union composition ---"
echo "train_*: $(ls /d/sih2026/data/shards_union/train_*.npz 2>/dev/null | wc -l)"
echo "train_g*: $(ls /d/sih2026/data/shards_union/train_g*.npz 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- D: pilot
# Sample GPU temperature for G7 for as long as the pilot runs.
( while true; do
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null
    sleep 30
  done ) >/d/sih2026/logs/pilot_gpu_temp.log 2>/dev/null &
TEMPPID=$!

say "D: pilot starting"
PILOT_START=$(date +%s)
"$PY" train.py \
    --shards D:/sih2026/data/shards_union \
    --init-from D:/sih2026/checkpoints/run02/best.pt \
    --out D:/sih2026/checkpoints/pilot_union \
    --height-scale 13.223477220535276 \
    --epochs 1 --max-steps 1500 --batch 8 --workers 2 --max-temp 83
PILOT_RC=$?
PILOT_SECS=$(( $(date +%s) - PILOT_START ))
kill $TEMPPID 2>/dev/null

[ $PILOT_RC -ne 0 ] && fail D "pilot exited $PILOT_RC after ${PILOT_SECS}s"
say "D: pilot finished in ${PILOT_SECS}s"
echo "PILOT_SECS=$PILOT_SECS" >/d/sih2026/logs/pilot_timing.txt

# ---------------------------------------------------------------- E: score
# No --tta: the gates are catastrophe detectors and are stated against run02's
# non-TTA 7.980 m. Adding TTA here would cost 8x for a number the gates cannot use.
say "E: evaluating pilot on DFC2019 val"
"$PY" tools/evaluate.py \
    --ckpt D:/sih2026/checkpoints/pilot_union/best.pt \
    --split val --tiles 80 \
    --out D:/sih2026/out/eval_pilot_union || fail E "evaluate failed"

say "DONE: pilot scored, awaiting gate review"
exit 0
