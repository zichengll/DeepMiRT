#!/usr/bin/env bash
# =============================================================================
# Ablation Study Runner
# =============================================================================
# Usage:
#   # Smoke test all ablation experiments (--fast-dev-run):
#   bash scripts/run_ablation.sh --dry-run
#
#   # Run a specific experiment:
#   bash scripts/run_ablation.sh A1_concat
#
#   # Run all P0 experiments:
#   bash scripts/run_ablation.sh --p0
#
#   # Run all experiments sequentially:
#   bash scripts/run_ablation.sh --all
#
#   # Run with multiple seeds (for P0 experiments):
#   bash scripts/run_ablation.sh --p0 --seeds 42,123,456
#
#   # Resume from Phase 1 checkpoint (for experiments needing Phase 2):
#   bash scripts/run_ablation.sh A2_1layer --phase2 --ckpt checkpoints/phase1_best.ckpt
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ABLATION_DIR="${PROJECT_ROOT}/deepmirt/configs/ablation"
TRAIN_SCRIPT="${PROJECT_ROOT}/deepmirt/training/train.py"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/ablation"

# P0 experiments (highest priority)
P0_EXPERIMENTS=(A1_concat A2_1layer A2_4layers B1_random_init B3_frozen_only)
# P1 experiments
P1_EXPERIMENTS=(A4_reverse_qk B2_separate_encoder B4_full_finetune C1_linear_head C2_max_pooling D1_uniform_lr)
# P2 experiments
P2_EXPERIMENTS=(A3_4heads A3_16heads C3_cls_pooling D2_dropout01 D2_dropout05)

# All experiments
ALL_EXPERIMENTS=("${P0_EXPERIMENTS[@]}" "${P1_EXPERIMENTS[@]}" "${P2_EXPERIMENTS[@]}")

# Default parameters
DRY_RUN=false
SEEDS="42"
PHASE2=false
CKPT_PATH=""
EXTRA_OVERRIDES=""

# Parse arguments
EXPERIMENTS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --p0)         EXPERIMENTS=("${P0_EXPERIMENTS[@]}"); shift ;;
        --p1)         EXPERIMENTS=("${P1_EXPERIMENTS[@]}"); shift ;;
        --p2)         EXPERIMENTS=("${P2_EXPERIMENTS[@]}"); shift ;;
        --all)        EXPERIMENTS=("${ALL_EXPERIMENTS[@]}"); shift ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --phase2)     PHASE2=true; shift ;;
        --ckpt)       CKPT_PATH="$2"; shift 2 ;;
        --override)   EXTRA_OVERRIDES="${EXTRA_OVERRIDES} --override $2"; shift 2 ;;
        -*)           echo "Unknown option: $1"; exit 1 ;;
        *)            EXPERIMENTS+=("$1"); shift ;;
    esac
done

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--dry-run|--p0|--p1|--p2|--all|EXPERIMENT_NAME] [--seeds 42,123,456]"
    echo ""
    echo "Available experiments:"
    echo "  P0: ${P0_EXPERIMENTS[*]}"
    echo "  P1: ${P1_EXPERIMENTS[*]}"
    echo "  P2: ${P2_EXPERIMENTS[*]}"
    exit 0
fi

# Convert seeds string to array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

run_experiment() {
    local exp_name="$1"
    local seed="$2"
    local config_file="${ABLATION_DIR}/${exp_name}.yaml"

    if [[ ! -f "$config_file" ]]; then
        echo "[ERROR] Config not found: ${config_file}"
        return 1
    fi

    local run_name="${exp_name}_seed${seed}"
    local ckpt_dir="${CHECKPOINT_DIR}/${run_name}"
    mkdir -p "${ckpt_dir}"

    echo "=============================================="
    echo "[RUN] Experiment: ${run_name}"
    echo "[RUN] Config: ${config_file}"
    echo "[RUN] Checkpoint dir: ${ckpt_dir}"
    echo "=============================================="

    local cmd="python ${TRAIN_SCRIPT} --config ${config_file}"
    cmd="${cmd} --override seed=${seed}"
    cmd="${cmd} --override checkpointing.dirpath=${ckpt_dir}"
    cmd="${cmd} --override logging.log_dir=${CHECKPOINT_DIR}/logs/"

    if [[ "$DRY_RUN" == true ]]; then
        cmd="${cmd} --fast-dev-run"
    fi

    if [[ "$PHASE2" == true ]] && [[ -n "$CKPT_PATH" ]]; then
        cmd="${cmd} --ckpt-path ${CKPT_PATH}"
    fi

    if [[ -n "$EXTRA_OVERRIDES" ]]; then
        cmd="${cmd} ${EXTRA_OVERRIDES}"
    fi

    echo "[CMD] ${cmd}"
    eval "${cmd}"

    echo "[DONE] ${run_name} completed"
    echo ""
}

# Run experiments
TOTAL=${#EXPERIMENTS[@]}
TOTAL_SEEDS=${#SEED_ARRAY[@]}
echo "Running ${TOTAL} experiment(s) x ${TOTAL_SEEDS} seed(s) = $((TOTAL * TOTAL_SEEDS)) run(s)"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        run_experiment "$exp" "$seed"
    done
done

echo "=============================================="
echo "All ablation experiments completed!"
echo "Results saved to: ${CHECKPOINT_DIR}"
echo "=============================================="
