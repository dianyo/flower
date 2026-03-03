# Experiment Guide: Federated Learning for Poultry Disease Classification

This document outlines the complete experimental pipeline for evaluating federated learning approaches on the poultry fecal disease classification task.

## Quick Start: Automated Experiment Runner

```bash
cd examples/flowertune-vit-poultry
source .venv/bin/activate
source dev.env  # Load WANDB_API_KEY, HF_TOKEN

# List all available experiments
python scripts/run_experiments.py --list

# Run baseline experiments (centralized + single-farm)
python scripts/run_experiments.py --phases baseline --wandb --batch-size 256

# Run federated learning experiments
python scripts/run_experiments.py --phases federated --wandb --batch-size 256

# Run ablation studies
python scripts/run_experiments.py --phases ablation --wandb --batch-size 256

# Run ALL experiments
python scripts/run_experiments.py --phases all --wandb --batch-size 256

# Run a specific experiment
python scripts/run_experiments.py --experiment centralized_vit --wandb --batch-size 256

# Dry run (preview without executing)
python scripts/run_experiments.py --phases all --dry-run
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--phases` | Which phases to run: `baseline`, `federated`, `ablation`, `all` | `all` |
| `--experiment` | Run a specific experiment by name | - |
| `--wandb` | Enable Weights & Biases logging | disabled |
| `--batch-size` | Batch size for training (use 256+ for A100) | 128 |
| `--timeout` | Timeout per experiment in seconds | 3600 |
| `--output-dir` | Directory for results | `experiment_results` |
| `--list` | List all available experiments | - |
| `--dry-run` | Show what would run without executing | - |

### Results Output

Results are saved to `experiment_results/<timestamp>/`:
- `results.json` - Full results with metrics and logs
- `results_summary.csv` - Summary table for quick analysis
- `<experiment_name>.log` - Individual experiment logs

---

## Prerequisites

```bash
cd examples/flowertune-vit-poultry
source .venv/bin/activate
```

Ensure the dataset is uploaded to HuggingFace:
- **4-class dataset**: `Dianyo/poultry-fecal-fl` (8,770 unique images after deduplication)

## Experiment Overview

We evaluate three training paradigms:
1. **Centralized Training** (Upper Bound): All data pooled together
2. **Single-Client Training** (Lower Bound): Training on isolated farm data
3. **Federated Learning**: Distributed training with various strategies

### Research Questions

| RQ | Question | Experiments |
|----|----------|-------------|
| RQ1 | Can FL match centralized performance? | Centralized vs FL-IID |
| RQ2 | How does data heterogeneity affect FL? | FL-IID vs FL-NonIID |
| RQ3 | Does FedProx help with Non-IID data? | FL-NonIID-FedAvg vs FL-NonIID-FedProx |
| RQ4 | How do lightweight models compare? | ViT vs MobileViT vs Swin |

---

## Phase 1: Baseline Experiments

### 1.1 Centralized Baseline (Upper Bound)

Train on the full dataset without partitioning. This establishes the best possible accuracy.

```bash
python -m vitpoultry.centralized_baseline \
    --dataset Dianyo/poultry-fecal-fl \
    --model vit_b_16 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001

# Expected output: Best accuracy, precision, recall, F1 per class
```

**Variants to run:**
```bash
# MobileViT (lightweight)
python -m vitpoultry.centralized_baseline --model mobilevit_s --epochs 20

# Swin Transformer Tiny
python -m vitpoultry.centralized_baseline --model swin_t --epochs 20
```

### 1.2 Single-Farm Baseline (Lower Bound)

Train on only one partition's data. This shows what happens without collaboration.

```bash
# Run for multiple partitions to get variance
for i in 0 1 2 3 4; do
    python -m vitpoultry.single_farm_baseline \
        --partition-id $i \
        --num-partitions 5 \
        --epochs 20 \
        --partitioning iid
done
```

**With Non-IID partitioning:**
```bash
for i in 0 1 2 3 4; do
    python -m vitpoultry.single_farm_baseline \
        --partition-id $i \
        --num-partitions 5 \
        --epochs 20 \
        --partitioning dirichlet \
        --dirichlet-alpha 0.5
done
```

---

## Phase 2: Federated Learning Experiments

All FL experiments use the Flower framework via `flwr run`.

### 2.1 FL with IID Data (Baseline FL)

Uniform random partitioning - each client has balanced class distribution.

```bash
flwr run . --run-config "\
    num-server-rounds=10 \
    strategy='fedavg' \
    partitioning='iid' \
    model-name='vit_b_16' \
    learning-rate=0.001 \
    batch-size=32"
```

### 2.2 FL with Non-IID Data (Dirichlet Partitioning)

Heterogeneous data distribution simulating real-world farm differences.

**FedAvg with moderate heterogeneity (α=0.5):**
```bash
flwr run . --run-config "\
    num-server-rounds=10 \
    strategy='fedavg' \
    partitioning='dirichlet' \
    dirichlet-alpha=0.5 \
    model-name='vit_b_16'"
```

**FedAvg with high heterogeneity (α=0.1):**
```bash
flwr run . --run-config "\
    num-server-rounds=10 \
    strategy='fedavg' \
    partitioning='dirichlet' \
    dirichlet-alpha=0.1 \
    model-name='vit_b_16'"
```

### 2.3 FedProx for Non-IID Data

FedProx adds a proximal term to handle data heterogeneity.

```bash
flwr run . --run-config "\
    num-server-rounds=10 \
    strategy='fedprox' \
    proximal-mu=0.1 \
    partitioning='dirichlet' \
    dirichlet-alpha=0.5 \
    model-name='vit_b_16'"
```

**Ablation on proximal-mu:**
```bash
# Lower regularization
flwr run . --run-config "strategy='fedprox' proximal-mu=0.01 partitioning='dirichlet' dirichlet-alpha=0.5"

# Higher regularization
flwr run . --run-config "strategy='fedprox' proximal-mu=1.0 partitioning='dirichlet' dirichlet-alpha=0.5"
```

---

## Phase 3: Model Architecture Comparison

Compare ViT variants for edge deployment feasibility.

### 3.1 ViT-B-16 (Full-size baseline)
```bash
flwr run . --run-config "model-name='vit_b_16' strategy='fedavg' partitioning='iid'"
```

### 3.2 MobileViT-S (Lightweight)
```bash
flwr run . --run-config "model-name='mobilevit_s' strategy='fedavg' partitioning='iid'"
```

### 3.3 Swin Transformer Tiny
```bash
flwr run . --run-config "model-name='swin_t' strategy='fedavg' partitioning='iid'"
```

---

## Phase 4: Extended Experiments

### 4.1 Number of Clients Ablation

Test with different numbers of simulated farms.

```bash
# 3 clients
flwr run . --run-config "num-server-rounds=10" local-num-partitions=3

# 5 clients (default)
flwr run . --run-config "num-server-rounds=10" local-num-partitions=5

# 10 clients
flwr run . --run-config "num-server-rounds=10" local-num-partitions=10
```

### 4.2 Communication Rounds Ablation

```bash
# Fewer rounds
flwr run . --run-config "num-server-rounds=5 strategy='fedavg' partitioning='iid'"

# More rounds
flwr run . --run-config "num-server-rounds=20 strategy='fedavg' partitioning='iid'"
```

### 4.3 Learning Rate Sensitivity

```bash
flwr run . --run-config "learning-rate=0.0001 strategy='fedavg' partitioning='iid'"
flwr run . --run-config "learning-rate=0.001 strategy='fedavg' partitioning='iid'"
flwr run . --run-config "learning-rate=0.01 strategy='fedavg' partitioning='iid'"
```

---

## Phase 5: Interpretability

### 5.1 Grad-CAM Visualization

Generate attention maps to verify the model focuses on disease indicators.

```bash
# Single image
python -m vitpoultry.gradcam_viz \
    --checkpoint results/best_model.pth \
    --image raw_data/zenodo_farm/cocci/cocci/cocci.1.jpg \
    --output gradcam_cocci.png

# Batch processing (sample from each class)
python -m vitpoultry.gradcam_viz \
    --checkpoint results/best_model.pth \
    --input-dir deduplicated_data/ \
    --output-dir gradcam_results/ \
    --samples-per-class 5
```

---

## Experiment Matrix Summary

| Exp ID | Training | Strategy | Partitioning | α | Model | Purpose |
|--------|----------|----------|--------------|---|-------|---------|
| C1 | Centralized | - | - | - | ViT-B-16 | Upper bound |
| C2 | Centralized | - | - | - | MobileViT | Lightweight upper bound |
| S1-S5 | Single-farm | - | IID | - | ViT-B-16 | Lower bound |
| F1 | FL | FedAvg | IID | - | ViT-B-16 | FL baseline |
| F2 | FL | FedAvg | Dirichlet | 0.5 | ViT-B-16 | Non-IID impact |
| F3 | FL | FedAvg | Dirichlet | 0.1 | ViT-B-16 | High heterogeneity |
| F4 | FL | FedProx | Dirichlet | 0.5 | ViT-B-16 | FedProx benefit |
| F5 | FL | FedProx | Dirichlet | 0.1 | ViT-B-16 | FedProx under stress |
| M1 | FL | FedAvg | IID | - | MobileViT | Lightweight FL |
| M2 | FL | FedAvg | IID | - | Swin-T | Alternative arch |

---

## Expected Results Format

Each experiment should log:

```
Experiment: F2 (FL-FedAvg-NonIID-0.5)
Model: vit_b_16
Rounds: 10
Final Test Accuracy: XX.XX%
Final Test Loss: X.XXX

Per-Class Metrics:
  healthy:  P=0.XX  R=0.XX  F1=0.XX
  cocci:    P=0.XX  R=0.XX  F1=0.XX
  ncd:      P=0.XX  R=0.XX  F1=0.XX
  salmo:    P=0.XX  R=0.XX  F1=0.XX

Macro F1: 0.XX
Weighted F1: 0.XX
```

---

## Results Collection Script

To run all experiments and collect results:

```bash
# Create results directory
mkdir -p results/experiments

# Run all experiments (example)
./scripts/run_all_experiments.sh 2>&1 | tee results/experiment_log.txt
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM recommended
- **RAM**: ≥16GB for data loading
- **Storage**: ~5GB for dataset + results

**Estimated run times (per experiment on single GPU):**
- Centralized baseline: ~30-60 min
- Single-farm baseline: ~10-20 min each
- FL experiment (10 rounds): ~20-40 min

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
flwr run . --run-config "batch-size=16"
```

### Slow Training
```bash
# Use mixed precision (if supported)
# Reduce image size in task.py transforms
```

### Dataset Loading Issues
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/Dianyo___poultry-fecal-fl
```
