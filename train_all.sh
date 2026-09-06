#!/usr/bin/env bash
# Runs only the missing training experiments.
# Runs NUM_GPUS trainings in parallel using a shared atomic job queue.
#
# Usage: bash train_missing.sh [--config PATH] [--batch-size N] [--num-gpus N]
# Default config: services/trainer/fine_tuning_config.ini
# Default batch_size: taken from config (no override)
# Default num_gpus: 2

set -euo pipefail

CONFIG="services/trainer/fine_tuning_config.ini"
BATCH_SIZE=""
NUM_GPUS=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";    shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --num-gpus)    NUM_GPUS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

HF_DATASET="victormuryn/wsd-training-dataset"

# ── Only missing experiments ─────────────────────────────────────────────────
#
# Format:
#   augmentation|dataset_seed|train_seed|pool_target
#
# These are the combinations that do NOT yet exist.

MISSING_JOBS=(
    # dropout
    "dropout|42|42|false"
    "dropout|42|123|false"
    "dropout|42|456|false"
    "dropout|123|42|false"
    "dropout|123|123|false"
    "dropout|123|456|false"
    "dropout|456|42|false"
    "dropout|456|123|false"
    "dropout|456|456|false"
    "dropout|42|42|true"
    "dropout|42|123|true"
    "dropout|42|456|true"
    "dropout|123|42|true"
    "dropout|123|123|true"
    "dropout|123|456|true"
    "dropout|456|42|true"
    "dropout|456|123|true"
    "dropout|456|456|true"

    # generated
    "generated|42|42|false"
    "generated|42|123|false"
    "generated|42|456|false"
    "generated|123|42|false"
    "generated|123|123|false"
    "generated|123|456|false"
    "generated|456|42|false"
    "generated|456|123|false"
    "generated|456|456|false"
    "generated|42|42|true"
    "generated|42|123|true"
    "generated|42|456|true"
    "generated|123|42|true"
    "generated|123|123|true"
    "generated|123|456|true"
    "generated|456|42|true"
    "generated|456|123|true"
    "generated|456|456|true"

    # mask
    "mask|42|42|false"
    "mask|42|123|false"
    "mask|42|456|false"
    "mask|123|42|false"
    "mask|123|123|false"
    "mask|123|456|false"
    "mask|456|42|false"
    "mask|456|123|false"
    "mask|456|456|false"
    "mask|42|42|true"
    "mask|42|123|true"
    "mask|42|456|true"
    "mask|123|42|true"
    "mask|123|123|true"
    "mask|123|456|true"
    "mask|456|42|true"
    "mask|456|123|true"
    "mask|456|456|true"

    # token_shuffling
    "token_shuffling|42|42|false"
    "token_shuffling|42|123|false"
    "token_shuffling|42|456|false"
    "token_shuffling|123|42|false"
    "token_shuffling|123|123|false"
    "token_shuffling|123|456|false"
    "token_shuffling|456|42|false"
    "token_shuffling|456|123|false"
    "token_shuffling|456|456|false"
    "token_shuffling|42|42|true"
    "token_shuffling|42|123|true"
    "token_shuffling|42|456|true"
    "token_shuffling|123|42|true"
    "token_shuffling|123|123|true"
    "token_shuffling|123|456|true"
    "token_shuffling|456|42|true"
    "token_shuffling|456|123|true"
    "token_shuffling|456|456|true"

    # translation
    "translation|42|42|false"
    "translation|42|123|false"
    "translation|42|456|false"
    "translation|123|42|false"
    "translation|123|123|false"
    "translation|123|456|false"
    "translation|456|42|false"
    "translation|456|123|false"
    "translation|456|456|false"
    "translation|42|42|true"
    "translation|42|123|true"
    "translation|42|456|true"
    "translation|123|42|true"
    "translation|123|123|true"
    "translation|123|456|true"
    "translation|456|42|true"
    "translation|456|123|true"
    "translation|456|456|true"

    # raw
    "raw|42|42|false"
    "raw|42|123|false"
    "raw|42|456|false"
    "raw|123|42|false"
    "raw|123|123|false"
    "raw|123|456|false"
    "raw|456|42|false"
    "raw|456|123|false"
    "raw|456|456|false"
    "raw|42|42|true"
    "raw|42|123|true"
    "raw|42|456|true"
    "raw|123|42|true"
    "raw|123|123|true"
    "raw|123|456|true"
    "raw|456|42|true"
    "raw|456|123|true"
    "raw|456|456|true"

    # all_augs
    "all_augs|42|42|false"
    "all_augs|42|123|false"
    "all_augs|42|456|false"
    "all_augs|123|42|false"
    "all_augs|123|123|false"
    "all_augs|123|456|false"
    "all_augs|456|42|false"
    "all_augs|456|123|false"
    "all_augs|456|456|false"
    "all_augs|42|42|true"
    "all_augs|42|123|true"
    "all_augs|42|456|true"
    "all_augs|123|42|true"
    "all_augs|123|123|true"
    "all_augs|123|456|true"
    "all_augs|456|42|true"
    "all_augs|456|123|true"
    "all_augs|456|456|true"

    # markov
    "markov|42|42|false"
    "markov|42|123|false"
    "markov|42|456|false"
    "markov|123|42|false"
    "markov|123|123|false"
    "markov|123|456|false"
    "markov|456|42|false"
    "markov|456|123|false"
    "markov|456|456|false"
    "markov|42|42|true"
    "markov|42|123|true"
    "markov|42|456|true"
    "markov|123|42|true"
    "markov|123|123|true"
    "markov|123|456|true"
    "markov|456|42|true"
    "markov|456|123|true"
    "markov|456|456|true"
)

# ── Build job list ────────────────────────────────────────────────────────────
JOBS=()

for job in "${MISSING_JOBS[@]}"; do
    IFS='|' read -r aug dseed tseed pool <<< "$job"

    subset="${aug}_seed${dseed}"
    run_name="${subset}_trainseed${tseed}_pool${pool}"

    JOBS+=("${subset}|${dseed}|${tseed}|${pool}|${run_name}")
done

TOTAL=${#JOBS[@]}

echo "Total missing training jobs: $TOTAL  ($NUM_GPUS GPUs in parallel)"

# ── Shared atomic job counter via temp files + flock ──────────────────────────
COUNTER_FILE=$(mktemp)
LOCK_FILE=$(mktemp)

echo 0 > "$COUNTER_FILE"

cleanup() {
    rm -f "$COUNTER_FILE" "$LOCK_FILE"
}
trap cleanup EXIT

# ── Worker ────────────────────────────────────────────────────────────────────
worker() {
    local gpu=$1

    while true; do
        local idx

        {
            flock -x 9

            idx=$(<"$COUNTER_FILE")
            echo $((idx + 1)) > "$COUNTER_FILE"

        } 9>"$LOCK_FILE"

        [ "$idx" -ge "$TOTAL" ] && break

        local subset dseed tseed pool run_name

        IFS='|' read -r subset dseed tseed pool run_name <<< "${JOBS[$idx]}"

        echo "[GPU $gpu | job $((idx + 1))/$TOTAL] $run_name"

        batch_args=()
        [ -n "$BATCH_SIZE" ] && batch_args=(--batch-size "$BATCH_SIZE")

        python3 -m services.trainer.trainer \
            --config       "$CONFIG" \
            --device       "cuda:${gpu}" \
            --hf-dataset   "$HF_DATASET" \
            --hf-subset    "$subset" \
            --seed         "$tseed" \
            --pool-targets "$pool" \
            --run-name     "$run_name" \
            "${batch_args[@]}" \
            2>&1 | sed "s/^/[gpu${gpu}|${run_name}] /" \
            || echo "[gpu${gpu}|${run_name}] TRAINING FAILED (non-zero exit), continuing to next job" >&2
    done
}

# ── Launch workers ────────────────────────────────────────────────────────────
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    worker "$gpu" &
done

wait

echo "All $TOTAL missing trainings complete."