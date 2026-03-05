#!/usr/bin/env python3
"""Automated experiment runner with wandb logging and skip logic.

Usage:
    python scripts/run_experiments.py --phases all --wandb --batch-size 256
    python scripts/run_experiments.py --phases baseline --wandb --skip-completed
    python scripts/run_experiments.py --experiment centralized_vit --wandb
    python scripts/run_experiments.py --list
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    phase: str
    command: list
    description: str
    is_flwr: bool = False


@dataclass
class ExperimentResult:
    name: str
    phase: str
    status: str
    start_time: str
    end_time: str
    duration_seconds: float
    metrics: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def get_experiments(use_wandb=False, batch_size=128):
    """Generate experiment configs with optional wandb flag."""
    wandb_flag = ["--wandb"] if use_wandb else []
    wandb_fl = "wandb=true" if use_wandb else "wandb=false"
    bs = str(batch_size)
    
    experiments = {}
    
    # ==========================================================================
    # BASELINE: Centralized Training (Upper Bound)
    # ==========================================================================
    for model, desc in [
        ("vit_b_16", "ViT-B/16 (86M params, reference)"),
        ("vit_s_16", "ViT-S/16 (22M params, edge-friendly)"),
        ("swin_tiny", "Swin-Tiny (28M params)"),
        ("swin_small", "Swin-Small (50M params)"),
        ("mobilevit_s", "MobileViT-v1-S (5.6M params)"),
        ("mobilevitv2_100", "MobileViT-v2-1.0 (4.9M params, most efficient)"),
        ("mobilevitv2_150", "MobileViT-v2-1.5 (10.6M params)"),
    ]:
        model_short = model.replace("_", "").replace("v2", "v2_")
        experiments[f"centralized_{model_short}"] = ExperimentConfig(
            name=f"centralized_{model_short}",
            phase="baseline",
            command=["python", "-m", "vitpoultry.centralized_baseline",
                     "--model-name", model, "--epochs", "10", "--batch-size", bs] + wandb_flag,
            description=f"Centralized training with {desc}",
        )
    
    # ==========================================================================
    # BASELINE: Single-Farm Training (Lower Bound) - All Partitions
    # ==========================================================================
    for partitioning in ["iid", "dirichlet"]:
        for partition_id in range(10):
            name = f"single_farm_vit_{partitioning}_p{partition_id}"
            extra_args = ["--dirichlet-alpha", "0.5"] if partitioning == "dirichlet" else []
            experiments[name] = ExperimentConfig(
                name=name,
                phase="baseline",
                command=["python", "-m", "vitpoultry.single_farm_baseline",
                         "--model-name", "vit_b_16", "--epochs", "10", "--batch-size", bs,
                         "--partition-id", str(partition_id), "--num-partitions", "10",
                         "--partitioning", partitioning] + extra_args + wandb_flag,
                description=f"Single farm training ({partitioning}, partition {partition_id})",
            )
    
    # ==========================================================================
    # FEDERATED: FedAvg Experiments
    # ==========================================================================
    # FedAvg IID vs Non-IID
    for partitioning, alpha in [("iid", None), ("dirichlet", 0.5)]:
        alpha_str = f" (α={alpha})" if alpha else ""
        alpha_cfg = f"dirichlet-alpha={alpha} " if alpha else ""
        name = f"fl_fedavg_{partitioning}_vit"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="federated",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedavg" partitioning="{partitioning}" {alpha_cfg}model-name="vit_b_16" num-server-rounds=10 batch-size={batch_size} {wandb_fl}'],
            description=f"FedAvg with {partitioning}{alpha_str} partitioning",
            is_flwr=True,
        )
    
    # FedAvg Model Comparison (all with Dirichlet α=0.5)
    for model, desc in [
        ("vit_s_16", "ViT-S/16"),
        ("swin_tiny", "Swin-Tiny"),
        ("swin_small", "Swin-Small"),
        ("mobilevit_s", "MobileViT-v1"),
        ("mobilevitv2_100", "MobileViT-v2-1.0"),
        ("mobilevitv2_150", "MobileViT-v2-1.5"),
    ]:
        model_short = model.replace("_", "").replace("v2", "v2_")
        name = f"fl_fedavg_dirichlet_{model_short}"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="federated",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 model-name="{model}" num-server-rounds=10 batch-size={batch_size} {wandb_fl}'],
            description=f"FedAvg with Dirichlet and {desc}",
            is_flwr=True,
        )
    
    # ==========================================================================
    # FEDERATED: FedProx Experiments (with μ tuning)
    # ==========================================================================
    for mu in [0.001, 0.01, 0.1, 0.5, 1.0]:
        mu_str = str(mu).replace(".", "")
        name = f"fl_fedprox_dirichlet_vit_mu{mu_str}"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="federated",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedprox" proximal-mu={mu} partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 batch-size={batch_size} {wandb_fl}'],
            description=f"FedProx (μ={mu}) with Dirichlet and ViT",
            is_flwr=True,
        )
    
    # ==========================================================================
    # FEDERATED: FedAdam Experiments
    # ==========================================================================
    # FedAdam with ViT-B/16 (different learning rates)
    for server_lr in [0.1, 0.01]:
        lr_str = str(server_lr).replace(".", "")
        name = f"fl_fedadam_dirichlet_vit_lr{lr_str}"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="federated",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedadam" server-lr={server_lr} partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds=10 batch-size={batch_size} {wandb_fl}'],
            description=f"FedAdam (η={server_lr}) with Dirichlet and ViT-B/16",
            is_flwr=True,
        )
    
    # FedAdam with other models (η=0.1 which worked best)
    for model, desc in [
        ("vit_s_16", "ViT-S/16"),
        ("swin_tiny", "Swin-Tiny"),
        ("swin_small", "Swin-Small"),
    ]:
        model_short = model.replace("_", "")
        name = f"fl_fedadam_dirichlet_{model_short}"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="federated",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedadam" server-lr=0.1 partitioning="dirichlet" dirichlet-alpha=0.5 model-name="{model}" num-server-rounds=10 batch-size={batch_size} {wandb_fl}'],
            description=f"FedAdam (η=0.1) with Dirichlet and {desc}",
            is_flwr=True,
        )
    
    # ==========================================================================
    # ABLATION: Dirichlet Alpha
    # ==========================================================================
    for alpha in [0.1, 0.5, 1.0]:
        alpha_str = str(alpha).replace(".", "")
        name = f"ablation_alpha_{alpha_str}"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="ablation",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha={alpha} model-name="vit_b_16" num-server-rounds=10 batch-size={batch_size} {wandb_fl}'],
            description=f"Ablation: Dirichlet α={alpha}",
            is_flwr=True,
        )
    
    # ==========================================================================
    # ABLATION: Communication Rounds
    # ==========================================================================
    for rounds in [5, 10, 20]:
        name = f"ablation_rounds_{rounds}"
        experiments[name] = ExperimentConfig(
            name=name,
            phase="ablation",
            command=["flwr", "run", ".", "--run-config",
                     f'strategy="fedavg" partitioning="dirichlet" dirichlet-alpha=0.5 model-name="vit_b_16" num-server-rounds={rounds} batch-size={batch_size} {wandb_fl}'],
            description=f"Ablation: {rounds} server rounds",
            is_flwr=True,
        )
    
    return experiments


def get_completed_experiments(output_dir):
    """Scan all result directories and return set of completed experiment names."""
    completed = set()
    output_path = Path(output_dir)
    if not output_path.exists():
        return completed
    
    for run_dir in output_path.iterdir():
        if not run_dir.is_dir():
            continue
        results_file = run_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                for r in results:
                    if r.get("status") == "success":
                        completed.add(r["name"])
            except:
                pass
        # Also check CSV for backwards compatibility
        csv_file = run_dir / "results_summary.csv"
        if csv_file.exists():
            try:
                with open(csv_file) as f:
                    for line in f:
                        if ",success," in line:
                            completed.add(line.split(",")[0])
            except:
                pass
    return completed


def parse_metrics_from_output(output):
    metrics = {}
    for line in output.split("\n"):
        ll = line.lower()
        if "accuracy:" in ll or "accuracy=" in ll:
            try:
                if "=" in line:
                    val = float(line.split("accuracy=")[1].split(",")[0].split()[0])
                else:
                    val = float(line.split(":")[-1].strip().rstrip("%"))
                metrics["accuracy"] = val / 100 if val > 1 else val
            except: pass
        if "precision" in ll and "macro" in ll:
            try: metrics["precision"] = float(line.split(":")[-1].strip())
            except: pass
        if "recall" in ll and "macro" in ll:
            try: metrics["recall"] = float(line.split(":")[-1].strip())
            except: pass
        if "f1" in ll and ("macro" in ll or "score" in ll):
            try: metrics["f1"] = float(line.split(":")[-1].strip())
            except: pass
    return metrics


def run_experiment(config, results_dir, timeout=3600):
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Command: {' '.join(config.command)}")
    print(f"{'='*60}", flush=True)
    
    start = datetime.now()
    try:
        proc = subprocess.run(config.command, capture_output=True, text=True,
                              timeout=timeout, cwd=Path(__file__).parent.parent)
        end = datetime.now()
        dur = (end - start).total_seconds()
        metrics = parse_metrics_from_output(proc.stdout)
        status = "success" if proc.returncode == 0 else "failed"
        
        with open(results_dir / f"{config.name}.log", "w") as f:
            f.write(f"Command: {' '.join(config.command)}\nStatus: {status}\n")
            f.write(f"Duration: {dur:.1f}s\n\n=== STDOUT ===\n{proc.stdout}\n")
            f.write(f"\n=== STDERR ===\n{proc.stderr}")
        
        print(f"Status: {status}, Duration: {dur:.1f}s, Metrics: {metrics}", flush=True)
        return ExperimentResult(config.name, config.phase, status, start.isoformat(),
                                end.isoformat(), dur, metrics, proc.stdout[-5000:], proc.stderr[-2000:])
    except subprocess.TimeoutExpired:
        end = datetime.now()
        dur = (end - start).total_seconds()
        return ExperimentResult(config.name, config.phase, "timeout", start.isoformat(),
                                end.isoformat(), dur, error=f"Timeout after {timeout}s")
    except Exception as e:
        end = datetime.now()
        return ExperimentResult(config.name, config.phase, "failed", start.isoformat(),
                                end.isoformat(), (end-start).total_seconds(), error=str(e))


def save_results(results, results_dir):
    with open(results_dir / "results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    with open(results_dir / "results_summary.csv", "w") as f:
        f.write("name,phase,status,duration,accuracy,precision,recall,f1\n")
        for r in results:
            row = [r.name, r.phase, r.status, f"{r.duration_seconds:.1f}",
                   f"{r.metrics.get('accuracy',''):.4f}" if r.metrics.get('accuracy') else "",
                   f"{r.metrics.get('precision',''):.4f}" if r.metrics.get('precision') else "",
                   f"{r.metrics.get('recall',''):.4f}" if r.metrics.get('recall') else "",
                   f"{r.metrics.get('f1',''):.4f}" if r.metrics.get('f1') else ""]
            f.write(",".join(row) + "\n")


def print_summary(results, skipped=None):
    print("\n" + "="*90, flush=True)
    print("RESULTS SUMMARY")
    print("="*90)
    print(f"{'Name':<40} {'Phase':<12} {'Status':<10} {'Duration':<10} {'Accuracy':<10}")
    print("-"*90)
    for r in results:
        acc = f"{r.metrics.get('accuracy',0):.4f}" if r.metrics.get('accuracy') else "N/A"
        print(f"{r.name:<40} {r.phase:<12} {r.status:<10} {r.duration_seconds:.0f}s{'':<5} {acc:<10}")
    print("="*90)
    success_count = sum(1 for r in results if r.status=='success')
    print(f"Total: {len(results)}, Success: {success_count}", flush=True)
    if skipped:
        print(f"Skipped (already completed): {len(skipped)}")


def main():
    parser = argparse.ArgumentParser(description="Run FL experiments with wandb logging")
    parser.add_argument("--phases", nargs="+", default=["all"],
                        choices=["all", "baseline", "federated", "ablation"])
    parser.add_argument("--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=str, default="experiment_results")
    parser.add_argument("--list", action="store_true", help="List experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--skip-completed", action="store_true", 
                        help="Skip experiments that have already completed successfully")
    args = parser.parse_args()
    
    experiments = get_experiments(use_wandb=args.wandb, batch_size=args.batch_size)
    
    if args.list:
        for phase in ["baseline", "federated", "ablation"]:
            print(f"\n{phase.upper()}:")
            for n, c in experiments.items():
                if c.phase == phase: 
                    print(f"  {n}: {c.description}")
        print(f"\nTotal experiments: {len(experiments)}")
        return
    
    # Filter experiments
    if args.experiment:
        if args.experiment not in experiments:
            print(f"Unknown: {args.experiment}"); sys.exit(1)
        exps = [experiments[args.experiment]]
    elif "all" in args.phases:
        exps = list(experiments.values())
    else:
        exps = [c for c in experiments.values() if c.phase in args.phases]
    
    # Skip completed experiments if requested
    skipped = []
    if args.skip_completed:
        completed = get_completed_experiments(args.output_dir)
        original_count = len(exps)
        skipped = [e for e in exps if e.name in completed]
        exps = [e for e in exps if e.name not in completed]
        if skipped:
            print(f"Skipping {len(skipped)} already completed experiments:")
            for e in skipped[:10]:
                print(f"  - {e.name}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")
    
    print(f"\nExperiments to run: {len(exps)}")
    print(f"WandB: {'enabled' if args.wandb else 'disabled'}")
    print(f"Batch size: {args.batch_size}")
    for e in exps[:20]: 
        print(f"  - {e.name}")
    if len(exps) > 20:
        print(f"  ... and {len(exps) - 20} more")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for e in exps:
            print(f"\n  {e.name}:")
            print(f"    {' '.join(e.command)}")
        return
    
    if not exps:
        print("\nNo experiments to run!")
        return
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / ts
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, cfg in enumerate(exps, 1):
        print(f"\n[{i}/{len(exps)}] {cfg.name}", flush=True)
        results.append(run_experiment(cfg, results_dir, args.timeout))
        save_results(results, results_dir)
    
    print_summary(results, skipped)
    print(f"\nResults saved to: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
