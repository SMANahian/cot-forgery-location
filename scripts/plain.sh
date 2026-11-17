#!/bin/bash
# The plain pipeline (src/make_plain.py) on one pool of one arm (PLAIN_ARM, default main), on a GPU pod. Each round,
# every auditor steps in order, and a writer also writes; two vote-only steps close the last round, and every auditor
# then casts confirmation votes on every pick. Every stage runs one command, and the first failure stops the script.
# After a crash, rerun with FIRST = the round that was running: a step at the writer's latest round only finishes that
# round's votes. The main arm (gpt-oss and Gemma) is final; Gemma at 49k tokens needed an H200.
#     PLAIN_ARM=qwen bash scripts/plain.sh design 24       # spare passages: rounds, confirmation, calibration, likelihood, go/no-go
#     PLAIN_ARM=qwen bash scripts/plain.sh picks 24        # the frozen rule on authentic and authentic_rep
#     PLAIN_ARM=qwen bash scripts/plain.sh picks 48 24     # a replacement wave, or a resume from round 24
set -uo pipefail
POOL=$1; ROUNDS=$2; FIRST=${3:-0}
case $POOL in design|picks) ;; *) echo "POOL must be design or picks"; exit 1 ;; esac
export PLAIN_ARM=${PLAIN_ARM:-main}
A=$PLAIN_ARM
L=${PLAIN_LOGS:-/workspace/logs}
VENV=${PLAIN_VENV:-/root/venv}
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/root/hf} HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: run scripts/setup_pod.sh first"; exit 1; }
source "$VENV/bin/activate"
mkdir -p "$L"
roles() { python -c "import sys; sys.path.insert(0, 'src'); import make_plain as p; print(*p.$1)"; }
WRITERS=$(roles WRITERS) || exit 1
AUDITORS=$(roles ORDER) || exit 1
echo "RUNNING $(date -u '+%F %H:%M') rounds $FIRST-$((ROUNDS - 1))" > "$L/plain_${A}_$POOL.status"
stage() {  # stage NAME COMMAND...: one command, logged; stop the script if it fails
    local name=$1; shift
    echo "RUNNING $(date -u +%H:%M)" > "$L/$name.status"
    "$@" > "$L/$name.log" 2>&1
    if [ $? -eq 0 ]; then echo "DONE $(date -u +%H:%M)" > "$L/$name.status"
    else echo "FAILED $(date -u +%H:%M)" > "$L/$name.status"; echo "FAILED at $name $(date -u +%H:%M)" > "$L/plain_${A}_$POOL.status"; exit 1; fi
}
P="python src/make_plain.py"
if [ "$POOL" = design ] && [ "$FIRST" = 0 ]; then
    for m in $WRITERS; do stage "plain_${A}_design_passages_$m" $P design "$m"; done
fi
for r in $(seq "$FIRST" $((ROUNDS - 1))); do
    for m in $AUDITORS; do stage "plain_${A}_${POOL}_r${r}_$m" $P step "$m" "$r" "$POOL"; done
done
for m in $AUDITORS; do stage "plain_${A}_${POOL}_close_$m" $P step "$m" -1 "$POOL"; done
for m in $AUDITORS; do stage "plain_${A}_${POOL}_confirm_$m" $P confirm "$m" "$POOL"; done
if [ "$POOL" = design ]; then
    for m in $AUDITORS; do stage "plain_${A}_calibrate_$m" $P calibrate "$m"; done
    stage "plain_${A}_pairs" $P pairs
    for m in $WRITERS; do stage "plain_${A}_score_$m" python src/make_authentic.py "$m" score; done
fi
stage "plain_${A}_${POOL}_status" $P status "$POOL"
echo "ALL $(date -u +%H:%M)" > "$L/plain_${A}_$POOL.status"
