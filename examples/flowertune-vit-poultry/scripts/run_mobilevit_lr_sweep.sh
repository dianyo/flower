#!/bin/bash
# MobileViT Learning Rate Sweep Experiments
# 24 experiments: 3 models × 4 LRs × 2 finetune modes
#
# Usage:
#   ./scripts/run_mobilevit_lr_sweep.sh          # Run all
#   ./scripts/run_mobilevit_lr_sweep.sh v2150    # Run only MobileViT-v2-1.5
#   ./scripts/run_mobilevit_lr_sweep.sh head     # Run only head fine-tuning
#
# Models: mobilevit_s (v1), mobilevitv2_100 (v2-1.0), mobilevitv2_150 (v2-1.5)
# LRs: 1e-2, 1e-3 (default), 1e-4, 1e-5
# Modes: head (head-only), full (full fine-tuning)

set -e  # Exit on error

cd "$(dirname "$0")/.."

FILTER="${1:-all}"

run_exp() {
    echo "========================================"
    echo "Running: $1"
    echo "========================================"
    python scripts/run_experiments.py --experiment "$1" --wandb
}

# MobileViT-v1-S experiments
if [[ "$FILTER" == "all" || "$FILTER" == "v1" || "$FILTER" == "mobilevits" ]]; then
    if [[ "$FILTER" == "all" || "$FILTER" == "v1" || "$FILTER" == "full" ]]; then
        run_exp fl_mobilevit_mobilevits_full_lr1e2
        run_exp fl_mobilevit_mobilevits_full_lr1e3
        run_exp fl_mobilevit_mobilevits_full_lr1e4
        run_exp fl_mobilevit_mobilevits_full_lr1e5
    fi
    if [[ "$FILTER" == "all" || "$FILTER" == "v1" || "$FILTER" == "head" ]]; then
        run_exp fl_mobilevit_mobilevits_head_lr1e2
        run_exp fl_mobilevit_mobilevits_head_lr1e3
        run_exp fl_mobilevit_mobilevits_head_lr1e4
        run_exp fl_mobilevit_mobilevits_head_lr1e5
    fi
fi

# MobileViT-v2-1.0 experiments
if [[ "$FILTER" == "all" || "$FILTER" == "v2100" ]]; then
    if [[ "$FILTER" == "all" || "$FILTER" == "v2100" || "$FILTER" == "full" ]]; then
        run_exp fl_mobilevit_mobilevitv2100_full_lr1e2
        run_exp fl_mobilevit_mobilevitv2100_full_lr1e3
        run_exp fl_mobilevit_mobilevitv2100_full_lr1e4
        run_exp fl_mobilevit_mobilevitv2100_full_lr1e5
    fi
    if [[ "$FILTER" == "all" || "$FILTER" == "v2100" || "$FILTER" == "head" ]]; then
        run_exp fl_mobilevit_mobilevitv2100_head_lr1e2
        run_exp fl_mobilevit_mobilevitv2100_head_lr1e3
        run_exp fl_mobilevit_mobilevitv2100_head_lr1e4
        run_exp fl_mobilevit_mobilevitv2100_head_lr1e5
    fi
fi

# MobileViT-v2-1.5 experiments (best centralized performance)
if [[ "$FILTER" == "all" || "$FILTER" == "v2150" ]]; then
    if [[ "$FILTER" == "all" || "$FILTER" == "v2150" || "$FILTER" == "full" ]]; then
        run_exp fl_mobilevit_mobilevitv2150_full_lr1e2
        run_exp fl_mobilevit_mobilevitv2150_full_lr1e3
        run_exp fl_mobilevit_mobilevitv2150_full_lr1e4
        run_exp fl_mobilevit_mobilevitv2150_full_lr1e5
    fi
    if [[ "$FILTER" == "all" || "$FILTER" == "v2150" || "$FILTER" == "head" ]]; then
        run_exp fl_mobilevit_mobilevitv2150_head_lr1e2
        run_exp fl_mobilevit_mobilevitv2150_head_lr1e3
        run_exp fl_mobilevit_mobilevitv2150_head_lr1e4
        run_exp fl_mobilevit_mobilevitv2150_head_lr1e5
    fi
fi

echo "========================================"
echo "All MobileViT LR sweep experiments completed!"
echo "========================================"
