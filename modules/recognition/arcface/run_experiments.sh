#!/bin/bash

# =============================================================================
# run_experiments.sh
# Runs a series of recognition experiments sequentially.
#
# Usage:
#   bash run_experiments.sh <watermarking_model> \
#       --experiments <exp1> [exp2 ...] \
#       --test_datasets <ds1> [ds2 ...] \
#       --train_datasets <ds1> [ds2 ...]
#
# Example:
#   bash run_experiments.sh stegaformer \
#       --experiments 1_1_255_w16_learn_im 1_3_255_w16_learn_im \
#       --test_datasets CFD LFW \
#       --train_datasets celeba_hq coco
#
# Notes:
#   - Runs experiments both with and without --use_mtcnn
#   - Runs both online and offline evaluations
#   - The last experiment for each (test_dataset x train_dataset) combination
#     in the MTCNN pass includes --debug_img to inspect detected face crops.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Usage function
# --------------------------------------------------------------------------- #
usage() {
    echo "Usage: $0 <watermarking_model> \\"
    echo "    --experiments <exp1> [exp2 ...] \\"
    echo "    --test_datasets <ds1> [ds2 ...] \\"
    echo "    --train_datasets <ds1> [ds2 ...]"
    echo ""
    echo "Example:"
    echo "  $0 stegaformer \\"
    echo "      --experiments 1_1_255_w16_learn_im 1_3_255_w16_learn_im \\"
    echo "      --test_datasets CFD LFW \\"
    echo "      --train_datasets celeba_hq coco"
    exit 1
}

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
if [ "$#" -lt 7 ]; then
    usage
fi

WATERMARKING_MODEL="$1"
shift

EXPERIMENT_NAMES=()
TEST_DATASETS=()
TRAIN_DATASETS=()

current_array=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --experiments)
            current_array="experiments"
            shift
            ;;
        --test_datasets)
            current_array="test"
            shift
            ;;
        --train_datasets)
            current_array="train"
            shift
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            ;;
        *)
            case "$current_array" in
                experiments)  EXPERIMENT_NAMES+=("$1") ;;
                test)         TEST_DATASETS+=("$1") ;;
                train)        TRAIN_DATASETS+=("$1") ;;
                *)
                    echo "Error: Value '$1' found before any flag"
                    usage
                    ;;
            esac
            shift
            ;;
    esac
done

# Validate arrays
if [ "${#EXPERIMENT_NAMES[@]}" -eq 0 ]; then
    echo "Error: At least one experiment name is required"; usage
fi
if [ "${#TEST_DATASETS[@]}" -eq 0 ]; then
    echo "Error: At least one test dataset is required"; usage
fi
if [ "${#TRAIN_DATASETS[@]}" -eq 0 ]; then
    echo "Error: At least one train dataset is required"; usage
fi

# Evaluation formats
EVAL_FORMATS=("online" "offline")

# Last indices (for --debug_img logic)
LAST_EXP_IDX=$(( ${#EXPERIMENT_NAMES[@]} - 1 ))
LAST_TRAIN_IDX=$(( ${#TRAIN_DATASETS[@]} - 1 ))
LAST_FORMAT_IDX=$(( ${#EVAL_FORMATS[@]} - 1 ))

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_cmd() {
    local cmd="$*"
    log "▶  $cmd"
    if eval "$cmd"; then
        log "✅ OK"
    else
        log "❌ FAILED (exit code $?)"
        FAILED_CMDS+=("$cmd")
    fi
    echo "---"
}

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
FAILED_CMDS=()
TOTAL=0

log "=== Starting experiments ==="
log "WM Model       : $WATERMARKING_MODEL"
log "Experiments    : ${EXPERIMENT_NAMES[*]}"
log "Test datasets  : ${TEST_DATASETS[*]}"
log "Train datasets : ${TRAIN_DATASETS[*]}"
log "Eval formats   : ${EVAL_FORMATS[*]}"
echo "==="

for USE_MTCNN in "1"; do #"0"

    if [ "$USE_MTCNN" = "1" ]; then
        log "--- Face detection: WITH MTCNN ---"
    else
        log "--- NO Face detection: WITHOUT MTCNN ---"
    fi

    for exp_idx in "${!EXPERIMENT_NAMES[@]}"; do
        EXP_NAME="${EXPERIMENT_NAMES[$exp_idx]}"

        for TEST_DS in "${TEST_DATASETS[@]}"; do

            for train_idx in "${!TRAIN_DATASETS[@]}"; do
                TRAIN_DS="${TRAIN_DATASETS[$train_idx]}"

                for format_idx in "${!EVAL_FORMATS[@]}"; do
                    FORMAT="${EVAL_FORMATS[$format_idx]}"

                    CMD="python run_recognizer.py"
                    CMD+=" --dataset $TEST_DS"
                    CMD+=" --train_dataset $TRAIN_DS"
                    CMD+=" --watermarking_model $WATERMARKING_MODEL"
                    CMD+=" --experiment_name $EXP_NAME"
                    CMD+=" --format_evaluation $FORMAT"

                    if [ "$USE_MTCNN" = "1" ]; then
                        CMD+=" --use_mtcnn"

                        # --debug_img only on the last command of each experiment
                        if [ "$exp_idx" -eq "$LAST_EXP_IDX" ]; then
                            CMD+=" --debug_img"
                        fi
                    fi

                    run_cmd "$CMD"
                    TOTAL=$((TOTAL + 1))
                done
            done
        done
    done
done

# --------------------------------------------------------------------------- #
# Final summary
# --------------------------------------------------------------------------- #
echo "==="
log "=== Summary ==="
log "Total executed : $TOTAL"
log "Failed         : ${#FAILED_CMDS[@]}"

if [ "${#FAILED_CMDS[@]}" -gt 0 ]; then
    log "Failed commands:"
    for CMD in "${FAILED_CMDS[@]}"; do
        echo "  ✗ $CMD"
    done
    exit 1
else
    log "All experiments completed successfully. 🎉"
fi