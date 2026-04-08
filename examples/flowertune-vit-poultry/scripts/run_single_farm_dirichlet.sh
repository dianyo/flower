#!/bin/bash
# Run single-farm Dirichlet experiments for ViT-S/16, Swin-Tiny, Swin-Small
# (ViT-B/16 already completed)
# Total: 3 models × 10 partitions = 30 experiments

cd /home/ubuntu/flower/flower/examples/flowertune-vit-poultry

echo "Running 30 single-farm Dirichlet experiments..."
echo "Models: ViT-S/16, Swin-Tiny, Swin-Small"
echo "Partitions: 0-9"
echo ""

# ViT-S/16 (10 experiments)
for p in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== ViT-S/16 partition $p ==="
    python scripts/run_experiments.py --experiment single_farm_vits16_dirichlet_p${p} --wandb
done

# Swin-Tiny (10 experiments)
for p in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== Swin-Tiny partition $p ==="
    python scripts/run_experiments.py --experiment single_farm_swintiny_dirichlet_p${p} --wandb
done

# Swin-Small (10 experiments)
for p in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== Swin-Small partition $p ==="
    python scripts/run_experiments.py --experiment single_farm_swinsmall_dirichlet_p${p} --wandb
done

echo ""
echo "All 30 single-farm Dirichlet experiments completed!"
