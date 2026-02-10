#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE_PATH="$ROOT/Trento.py"

SEEDS=(466 4426 7270 860 5390 5191 5734 6265 5578 8322)
GPUS=(0 1)

OUTPUT_BASE="$ROOT/result/trento_2025_final2"
RUN_BASE="$ROOT/runs/trento_2025_final2"

# LAMDA_START=0.01
# LAMDA_END=0.10
# LAMDA_STEP=0.01

LAMDA_FIXED=0.06

mkdir -p "$OUTPUT_BASE" "$RUN_BASE"

declare -a FREE_GPUS=("${GPUS[@]}")
declare -A PID_TO_GPU PID_TO_DESC PID_TO_LOG

start_job() {
    local gpu="$1"
    local seed="$2"
    local lamda="$3"
    local out_dir="$4"
    local run_dir="$5"
    local lamda_tag="${lamda/./p}"
    local log_file="$out_dir/result_seed${seed}_lamda${lamda_tag}.txt"

    mkdir -p "$run_dir/checkpoint"
    echo "[START] GPU=$gpu Seed=$seed Lamda=$lamda"

    (
        cd "$run_dir"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" \
            python3 "$FILE_PATH" --seed "$seed" --lamda "$lamda" > "$log_file" 2>&1
    ) &

    local pid=$!
    PID_TO_GPU["$pid"]="$gpu"
    PID_TO_DESC["$pid"]="GPU=$gpu Seed=$seed Lamda=$lamda"
    PID_TO_LOG["$pid"]="$log_file"
}

wait_for_any() {
    local finished_pid=""
    local status=0

    while :; do
        for pid in "${!PID_TO_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                finished_pid="$pid"
                if wait "$pid"; then
                    status=0
                else
                    status=$?
                fi
                break 2
            fi
        done
        sleep 1
    done

    local gpu="${PID_TO_GPU[$finished_pid]}"
    local desc="${PID_TO_DESC[$finished_pid]}"
    local log_file="${PID_TO_LOG[$finished_pid]}"

    FREE_GPUS+=("$gpu")

    if [ "$status" -eq 0 ]; then
        echo "[DONE ] $desc"
    else
        echo "[FAIL ] $desc -> Check $log_file"
    fi

    unset PID_TO_GPU["$finished_pid"] PID_TO_DESC["$finished_pid"] PID_TO_LOG["$finished_pid"]
}

# for lamda in $(seq -f "%.2f" "$LAMDA_START" "$LAMDA_STEP" "$LAMDA_END"); do
#     lamda_tag="${lamda/./p}"
#     out_dir="$OUTPUT_BASE/lamda_${lamda_tag}"
#     mkdir -p "$out_dir"
#
#     for seed in "${SEEDS[@]}"; do
#         while [ "${#FREE_GPUS[@]}" -eq 0 ]; do
#             wait_for_any
#         done
#
#         gpu="${FREE_GPUS[0]}"
#         FREE_GPUS=("${FREE_GPUS[@]:1}")
#
#         run_dir="$RUN_BASE/lamda_${lamda_tag}/seed_${seed}"
#         start_job "$gpu" "$seed" "$lamda" "$out_dir" "$run_dir"
#     done
# done
#
# while [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
#     wait_for_any
# done
#
# echo "All runs completed."

lamda_tag="${LAMDA_FIXED/./p}"
out_dir="$OUTPUT_BASE/lamda_${lamda_tag}"
mkdir -p "$out_dir"

for seed in "${SEEDS[@]}"; do
    while [ "${#FREE_GPUS[@]}" -eq 0 ]; do
        wait_for_any
    done

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    run_dir="$RUN_BASE/lamda_${lamda_tag}/seed_${seed}"
    start_job "$gpu" "$seed" "$LAMDA_FIXED" "$out_dir" "$run_dir"
done

while [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
    wait_for_any
done

echo "All fixed-lamda runs completed."
