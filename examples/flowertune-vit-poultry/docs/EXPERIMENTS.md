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

# Skip already completed experiments (checks experiment_results/)
python scripts/run_experiments.py --phases all --wandb --skip-completed
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
| `--skip-completed` | Skip experiments already in results | - |

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

---

## Model Zoo (<100M Parameters for Edge Deployment)

All models are selected for edge deployment feasibility (farm/mobile devices).

| Model | Params | FLOPs | Image Size | Notes |
|-------|--------|-------|------------|-------|
| `vit_b_16` | 86M | 17.6G | 224×224 | Baseline ViT |
| `vit_s_16` | 22M | 4.6G | 224×224 | Small ViT (efficient) |
| `swin_tiny` | 28M | 4.5G | 224×224 | Swin Transformer Tiny |
| `swin_small` | 50M | 8.7G | 224×224 | Swin Transformer Small |
| `mobilevit_s` | 5.6M | 2.0G | 256×256 | MobileViT v1 Small |
| `mobilevitv2_100` | 4.9M | 1.8G | 256×256 | MobileViT v2 (1.0×) - **Recommended** |
| `mobilevitv2_150` | 10.6M | 4.0G | 256×256 | MobileViT v2 (1.5×) |

**Recommended for Paper**: Focus on `vit_s_16`, `swin_tiny`, `mobilevitv2_100` for edge deployment story.

---

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
| RQ3 | Which FL strategy works best for non-IID? | FedAvg vs FedProx vs FedAdam |
| RQ4 | Which lightweight model is best for edge FL? | ViT-S vs Swin-T vs MobileViT-v2 |
| RQ5 | Does convergence-based training improve results? | Fixed epochs vs early stopping |

---

## Phase 1: Baseline Experiments

### 1.1 Centralized Baseline (Upper Bound)

Train on the full dataset without partitioning. This establishes the best possible accuracy.

**All model variants:**
```bash
# ViT-B-16 (reference)
python -m vitpoultry.centralized_baseline --model vit_b_16 --epochs 20 --batch-size 256 --wandb

# ViT-S-16 (edge-friendly)
python -m vitpoultry.centralized_baseline --model vit_s_16 --epochs 20 --batch-size 256 --wandb

# Swin Tiny
python -m vitpoultry.centralized_baseline --model swin_tiny --epochs 20 --batch-size 256 --wandb

# Swin Small
python -m vitpoultry.centralized_baseline --model swin_small --epochs 20 --batch-size 256 --wandb

# MobileViT v1 Small
python -m vitpoultry.centralized_baseline --model mobilevit_s --epochs 20 --batch-size 256 --wandb

# MobileViT v2 (1.0×) - Recommended
python -m vitpoultry.centralized_baseline --model mobilevitv2_100 --epochs 20 --batch-size 256 --wandb

# MobileViT v2 (1.5×)
python -m vitpoultry.centralized_baseline --model mobilevitv2_150 --epochs 20 --batch-size 256 --wandb
```

**With early stopping (convergence-based):**
```bash
python -m vitpoultry.centralized_baseline --model vit_b_16 --epochs 50 --early-stopping --patience 5 --wandb
```

### 1.2 Single-Farm Baseline (Lower Bound) - All Splits

Train on each partition's data separately to establish lower bound. **Run all partitions and report mean ± std.**

**IID Partitioning (10 partitions):**
```bash
for i in {0..9}; do
    python -m vitpoultry.single_farm_baseline \
        --model vit_b_16 \
        --partition-id $i \
        --num-partitions 10 \
        --epochs 20 \
        --partitioning iid \
        --wandb
done
```

**Non-IID Dirichlet Partitioning (10 partitions, α=0.5):**
```bash
for i in {0..9}; do
    python -m vitpoultry.single_farm_baseline \
        --model vit_b_16 \
        --partition-id $i \
        --num-partitions 10 \
        --epochs 20 \
        --partitioning dirichlet \
        --dirichlet-alpha 0.5 \
        --wandb
done
```

**Expected output format:**
```
Single-Farm Results (IID, 10 partitions):
  Partition 0: 85.2%
  Partition 1: 87.1%
  ...
  Mean ± Std: 86.3% ± 2.1%
```

---

## Phase 2: Federated Learning Experiments

All FL experiments use the Flower framework via `flwr run`.

### 2.1 FL Strategies Comparison

#### FedAvg (Baseline)
```bash
# IID partitioning
flwr run . --run-config 'strategy="fedavg" partitioning="iid" model-name="vit_b_16" num-server-rounds=10 batch-size=256 wandb=true'

# Non-IID (Dirichlet α=0.5)
flwr run . --run-config 'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 batch-size=256 wandb=true'
```

#### FedProx (with μ tuning)

Previous experiments showed poor results with high μ values. **Use lower μ for vision tasks:**

```bash
# μ = 0.001 (recommended starting point)
flwr run . --run-config 'strategy="fedprox" proximal-mu=0.001 partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 wandb=true'

# μ = 0.01
flwr run . --run-config 'strategy="fedprox" proximal-mu=0.01 partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 wandb=true'

# μ = 0.1 (already run - baseline)
flwr run . --run-config 'strategy="fedprox" proximal-mu=0.1 partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 wandb=true'
```

#### FedAdam (Adaptive Optimizer)

Server-side adaptive optimization for better convergence:

```bash
# Default FedAdam (η=0.1, β1=0.9, β2=0.99)
flwr run . --run-config 'strategy="fedadam" partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 wandb=true'

# FedAdam with tuned server LR
flwr run . --run-config 'strategy="fedadam" server-lr=0.01 partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 wandb=true'
```

### 2.2 Convergence-Based FL

Instead of fixed rounds, run until convergence:

```bash
# FedAvg with early stopping (max 30 rounds, patience 5)
flwr run . --run-config 'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=30 early-stopping=true patience=5 wandb=true'
```

---

## Phase 3: Model Architecture Comparison (Edge Deployment Focus)

Compare models for edge deployment feasibility under FL.

### 3.1 Primary Models (<30M params)

```bash
# ViT-S-16 (22M params)
flwr run . --run-config 'model-name="vit_s_16" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'

# Swin Tiny (28M params)
flwr run . --run-config 'model-name="swin_tiny" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'

# MobileViT v2 1.0× (4.9M params) - Most efficient
flwr run . --run-config 'model-name="mobilevitv2_100" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'
```

### 3.2 Extended Models (for ablation)

```bash
# ViT-B-16 (86M params) - reference
flwr run . --run-config 'model-name="vit_b_16" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'

# Swin Small (50M params)
flwr run . --run-config 'model-name="swin_small" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'

# MobileViT v2 1.5× (10.6M params)
flwr run . --run-config 'model-name="mobilevitv2_150" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'

# MobileViT v1 Small (5.6M params) - for comparison with v2
flwr run . --run-config 'model-name="mobilevit_s" strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 num-server-rounds=10 wandb=true'
```

---

## Phase 4: Ablation Studies

### 4.1 Data Heterogeneity (Dirichlet α)

```bash
# High heterogeneity (α=0.1)
flwr run . --run-config 'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.1 model-name="vit_b_16" num-server-rounds=10 wandb=true'

# Moderate heterogeneity (α=0.5) - default
flwr run . --run-config 'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 wandb=true'

# Low heterogeneity (α=1.0)
flwr run . --run-config 'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=1.0 model-name="vit_b_16" num-server-rounds=10 wandb=true'
```

### 4.2 Number of Communication Rounds

```bash
# 5 rounds
flwr run . --run-config 'num-server-rounds=5 strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 wandb=true'

# 10 rounds (default)
flwr run . --run-config 'num-server-rounds=10 strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 wandb=true'

# 20 rounds
flwr run . --run-config 'num-server-rounds=20 strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 wandb=true'
```

### 4.3 Number of Clients

```bash
# 5 clients
flwr run . --run-config 'num-server-rounds=10 strategy="fedavg"' --num-supernodes 5

# 10 clients (default)
flwr run . --run-config 'num-server-rounds=10 strategy="fedavg"' --num-supernodes 10

# 20 clients
flwr run . --run-config 'num-server-rounds=10 strategy="fedavg"' --num-supernodes 20
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

### Baseline Experiments

| Exp ID | Type | Model | Params | Purpose |
|--------|------|-------|--------|---------|
| C1 | Centralized | vit_b_16 | 86M | Upper bound (reference) |
| C2 | Centralized | vit_s_16 | 22M | Upper bound (edge) |
| C3 | Centralized | swin_tiny | 28M | Upper bound (Swin) |
| C4 | Centralized | swin_small | 50M | Upper bound (Swin-S) |
| C5 | Centralized | mobilevit_s | 5.6M | Upper bound (MobileViT v1) |
| C6 | Centralized | mobilevitv2_100 | 4.9M | Upper bound (MobileViT v2) |
| C7 | Centralized | mobilevitv2_150 | 10.6M | Upper bound (MobileViT v2 1.5×) |

### Single-Farm Experiments (All 10 Partitions)

| Exp ID | Model | Partitioning | Purpose |
|--------|-------|--------------|---------|
| SF-IID-0..9 | vit_b_16 | IID | Lower bound (IID, mean±std) |
| SF-DIR-0..9 | vit_b_16 | Dirichlet α=0.5 | Lower bound (Non-IID, mean±std) |

### Federated Learning Experiments

| Exp ID | Strategy | Model | Partitioning | μ/η | Purpose |
|--------|----------|-------|--------------|-----|---------|
| FL-AVG-IID | FedAvg | vit_b_16 | IID | - | FL baseline |
| FL-AVG-DIR | FedAvg | vit_b_16 | Dirichlet 0.5 | - | Non-IID impact |
| FL-PROX-001 | FedProx | vit_b_16 | Dirichlet 0.5 | μ=0.001 | FedProx tuned (low) |
| FL-PROX-01 | FedProx | vit_b_16 | Dirichlet 0.5 | μ=0.01 | FedProx tuned (mid) |
| FL-PROX-1 | FedProx | vit_b_16 | Dirichlet 0.5 | μ=0.1 | FedProx baseline |
| FL-ADAM | FedAdam | vit_b_16 | Dirichlet 0.5 | η=0.1 | Adaptive FL |
| FL-ADAM-LR | FedAdam | vit_b_16 | Dirichlet 0.5 | η=0.01 | FedAdam tuned |

### Model Comparison (FL, FedAvg, Dirichlet α=0.5)

| Exp ID | Model | Params | Purpose |
|--------|-------|--------|---------|
| M-VIT-B | vit_b_16 | 86M | Reference |
| M-VIT-S | vit_s_16 | 22M | Edge ViT |
| M-SWIN-T | swin_tiny | 28M | Swin Tiny |
| M-SWIN-S | swin_small | 50M | Swin Small |
| M-MVIT1 | mobilevit_s | 5.6M | MobileViT v1 |
| M-MVIT2-100 | mobilevitv2_100 | 4.9M | MobileViT v2 (best efficiency) |
| M-MVIT2-150 | mobilevitv2_150 | 10.6M | MobileViT v2 1.5× |

### Ablation Studies

| Exp ID | Variable | Values | Model | Purpose |
|--------|----------|--------|-------|---------|
| A-ALPHA | Dirichlet α | 0.1, 0.5, 1.0 | vit_b_16 | Heterogeneity impact |
| A-ROUNDS | Rounds | 5, 10, 20 | vit_b_16 | Communication efficiency |
| A-CLIENTS | Clients | 5, 10, 20 | vit_b_16 | Scalability |

---

## Completed Experiments (Skip List)

The following experiments have been completed and can be skipped:

### From Run 20260303_073511 (Baselines)
- ✅ centralized_vit (94.87%)
- ✅ centralized_mobilevit (85.18%)
- ✅ centralized_swin (93.39%)
- ✅ single_farm_vit_iid (partition 0 only, 89.05%)
- ✅ single_farm_vit_dirichlet (partition 0 only, 86.32%)

### From Run 20260304_053328 (Federated)
- ✅ fl_fedavg_iid_vit (88.43%)
- ✅ fl_fedavg_dirichlet_vit (88.77%)
- ✅ fl_fedavg_dirichlet_swin (85.40%)
- ✅ fl_fedavg_dirichlet_mobilevit (8.21% - needs re-eval with v2)
- ✅ fl_fedprox_dirichlet_vit_mu01 (31.58% - μ=0.1, too high)
- ✅ fl_fedprox_dirichlet_vit_mu05 (27.77% - μ=0.5, too high)
- ✅ fl_fedprox_dirichlet_vit_mu10 (35.80% - μ=1.0, too high)

---

## Priority Experiments (TODO)

### High Priority (Core Paper Results)
1. [ ] Single-farm all splits (IID): partitions 1-9
2. [ ] Single-farm all splits (Dirichlet): partitions 1-9
3. [ ] FedProx with μ=0.001
4. [ ] FedProx with μ=0.01
5. [ ] FedAdam baseline
6. [ ] MobileViT v2 centralized
7. [ ] MobileViT v2 FL

### Medium Priority (Extended Results)
8. [ ] ViT-S-16 centralized + FL
9. [ ] Swin Small centralized + FL
10. [ ] Convergence-based training (early stopping)

### Low Priority (Ablation)
11. [ ] Dirichlet α=0.1, α=1.0
12. [ ] Round ablation (5, 20 rounds)
13. [ ] Client ablation (5, 20 clients)

---

## Expected Results Format

Each experiment should log:

```
Experiment: FL-AVG-DIR (FL-FedAvg-NonIID-0.5)
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

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM recommended
- **RAM**: ≥16GB for data loading
- **Storage**: ~5GB for dataset + results

**Estimated run times (per experiment on A100):**
- Centralized baseline: ~10-15 min
- Single-farm baseline: ~5-8 min each
- FL experiment (10 rounds): ~8-10 min

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
flwr run . --run-config "batch-size=128"
```

### Slow Training
```bash
# Increase num_workers in DataLoader
# Use mixed precision (if supported)
```

### Dataset Loading Issues
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/Dianyo___poultry-fecal-fl
```
