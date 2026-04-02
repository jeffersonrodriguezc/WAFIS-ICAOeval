#!/bin/bash

# =============================================================================
# run_experiments.sh
# Ejecuta una serie de experimentos de reconocimiento secuencialmente.
#
# Uso:
#   bash run_experiments.sh <dataset> <watermarking_model> <experiment_name> [experiment_name2 ...]
#
# Ejemplo:
#   bash run_experiments.sh CFD stegaformer 1_1_255_w16_learn_im 1_3_255_w16_learn_im
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #
if [ "$#" -lt 3 ]; then
    echo "Uso: $0 <dataset> <watermarking_model> <experiment_name> [experiment_name2 ...]"
    echo "Ejemplo: $0 CFD stegaformer 1_1_255_w16_learn_im"
    exit 1
fi

DATASET="$1"
WATERMARKING_MODEL="$2"
shift 2
EXPERIMENT_NAMES=("$@")   # el resto de argumentos son experiment_names

# Datasets de entrenamiento a iterar
TRAIN_DATASETS=("celeba_hq" "coco")

# Combinaciones de format_evaluation y use_mtcnn
# Formato: "format_evaluation|use_mtcnn"  (use_mtcnn = 1 → añadir flag, 0 → no)
EVAL_COMBINATIONS=(
    "online|1"
    "offline|1"
    "offline|0"
    "online|0"
)

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
        log "❌ FALLÓ (código $?)"
        FAILED_CMDS+=("$cmd")
    fi
    echo "---"
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
FAILED_CMDS=()
TOTAL=0

log "=== Inicio de experimentos ==="
log "Dataset       : $DATASET"
log "WM Model      : $WATERMARKING_MODEL"
log "Experiments   : ${EXPERIMENT_NAMES[*]}"
echo "==="

for EXP_NAME in "${EXPERIMENT_NAMES[@]}"; do
    for TRAIN_DS in "${TRAIN_DATASETS[@]}"; do
        for COMBO in "${EVAL_COMBINATIONS[@]}"; do
            FORMAT="${COMBO%%|*}"
            USE_MTCNN="${COMBO##*|}"

            CMD="python run_recognizer.py"
            CMD+=" --dataset $DATASET"
            CMD+=" --train_dataset $TRAIN_DS"
            CMD+=" --watermarking_model $WATERMARKING_MODEL"
            CMD+=" --experiment_name $EXP_NAME"
            CMD+=" --format_evaluation $FORMAT"
            [ "$USE_MTCNN" = "1" ] && CMD+=" --use_mtcnn"

            run_cmd "$CMD"
            TOTAL=$((TOTAL + 1))
        done
    done
done

# --------------------------------------------------------------------------- #
# Resumen final
# --------------------------------------------------------------------------- #
echo "==="
log "=== Resumen ==="
log "Total ejecutados : $TOTAL"
log "Fallidos         : ${#FAILED_CMDS[@]}"

if [ "${#FAILED_CMDS[@]}" -gt 0 ]; then
    log "Comandos fallidos:"
    for CMD in "${FAILED_CMDS[@]}"; do
        echo "  ✗ $CMD"
    done
    exit 1
else
    log "Todos los experimentos completados con éxito. 🎉"
fi
