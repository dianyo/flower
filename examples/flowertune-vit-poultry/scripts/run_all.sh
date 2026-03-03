#!/bin/bash
# Run all experiments with wandb logging
# Usage: ./scripts/run_all.sh [--phases baseline|federated|ablation|all] [--wandb]

set -e
cd "$(dirname "$0")/.."

# Source environment variables (contains WANDB_API_KEY, HF_TOKEN, etc.)
if [ -f dev.env ]; then
    source dev.env
    echo "Loaded environment from dev.env"
fi

source .venv/bin/activate

echo "=============================================="
echo "FlowerTune ViT Poultry - Experiment Runner"
echo "=============================================="
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Date: $(date)"
echo "WANDB_API_KEY: ${WANDB_API_KEY:+set}"
echo "=============================================="

# Pass all arguments to the Python script
python scripts/run_experiments.py "$@"

echo ""
echo "Experiments complete! Check experiment_results/ for outputs."
