#!/bin/bash
# The plain pipeline (src/make_plain.py) on one pool, on a GPU pod; Gemma at 49k tokens needs an H200. Each round runs
# gpt-oss, then Gemma; two vote-only steps close the last round, and both models then cast confirmation votes on every
# pick. Every stage runs one command, and the first failure stops the script. After a crash, rerun with FIRST = the
# round that was running: a step at the writer's latest round only finishes that round's votes.
#     bash scripts/plain.sh design 24        # spare passages: rounds, confirmation, calibration, likelihood, go/no-go
#     bash scripts/plain.sh picks 24         # the frozen rule on authentic and authentic_rep
#     bash scripts/plain.sh picks 24 12      # resume from round 12
set -uo pipefail
POOL=$1; ROUNDS=$2; FIRST=${3:-0}
case $POOL in design|picks) ;; *) echo "POOL must be design or picks"; exit 1 ;; esac
L=${PLAIN_LOGS:-/workspace/logs}
VENV=${PLAIN_VENV:-/root/venv}
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/root/hf} HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: run scripts/setup_pod.sh first"; exit 1; }
source "$VENV/bin/activate"
mkdir -p "$L"
echo "RUNNING $(date -u '+%F %H:%M') rounds $FIRST-$((ROUNDS - 1))" > "$L/plain_$POOL.status"
stage() {  # stage NAME COMMAND...: one command, logged; stop the script if it fails
    local name=$1; shift
    echo "RUNNING $(date -u +%H:%M)" > "$L/$name.status"
    "$@" > "$L/$name.log" 2>&1
    if [ $? -eq 0 ]; then echo "DONE $(date -u +%H:%M)" > "$L/$name.status"
    else echo "FAILED $(date -u +%H:%M)" > "$L/$name.status"; echo "FAILED at $name $(date -u +%H:%M)" > "$L/plain_$POOL.status"; exit 1; fi
}
P="python src/make_plain.py"
if [ "$POOL" = design ] && [ "$FIRST" = 0 ]; then
    stage plain_design_passages_gptoss $P design gpt-oss-20b
    stage plain_design_passages_gemma $P design gemma-4-31b
fi
for r in $(seq "$FIRST" $((ROUNDS - 1))); do
    stage "plain_${POOL}_r${r}_gptoss" $P step gpt-oss-20b "$r" "$POOL"
    stage "plain_${POOL}_r${r}_gemma" $P step gemma-4-31b "$r" "$POOL"
done
stage "plain_${POOL}_close_gptoss" $P step gpt-oss-20b -1 "$POOL"
stage "plain_${POOL}_close_gemma" $P step gemma-4-31b -1 "$POOL"
stage "plain_${POOL}_confirm_gptoss" $P confirm gpt-oss-20b "$POOL"
stage "plain_${POOL}_confirm_gemma" $P confirm gemma-4-31b "$POOL"
if [ "$POOL" = design ]; then
    stage plain_calibrate_gptoss $P calibrate gpt-oss-20b
    stage plain_calibrate_gemma $P calibrate gemma-4-31b
    stage plain_pairs $P pairs
    stage plain_score_gptoss python src/make_authentic.py gpt-oss-20b score
    stage plain_score_gemma python src/make_authentic.py gemma-4-31b score
fi
stage "plain_${POOL}_status" $P status "$POOL"
echo "ALL $(date -u +%H:%M)" > "$L/plain_$POOL.status"
