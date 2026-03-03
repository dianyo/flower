"""Single-farm training baseline for lower-bound accuracy.

This script trains the model on data from a single farm/partition to establish
the lower-bound performance that federated learning should improve upon.

Usage:
    python -m vitpoultry.single_farm_baseline --dataset Dianyo/poultry-fecal-fl --partition-id 0
"""

import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from vitpoultry.task import (
    apply_eval_transforms,
    apply_train_transforms,
    get_classification_report,
    get_dataset_partition,
    get_model,
    test,
    trainer,
)

CLASS_NAMES = ["healthy", "cocci", "ncd", "salmo"]


def main():
    parser = argparse.ArgumentParser(description="Single-farm training baseline")
    parser.add_argument("--dataset", type=str, default="Dianyo/poultry-fecal-fl")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="vit_b_16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-partitions", type=int, default=5, help="Total number of partitions")
    parser.add_argument("--partition-id", type=int, default=0, help="Partition to train on")
    parser.add_argument("--partitioning", type=str, default="iid", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="flowertune-vit-poultry")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Skipping wandb logging.")

    if use_wandb:
        run_name = args.wandb_run_name or f"single_farm_{args.model_name}_{args.partitioning}_p{args.partition_id}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "experiment_type": "single_farm",
                "dataset": args.dataset,
                "model_name": args.model_name,
                "num_classes": args.num_classes,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "num_partitions": args.num_partitions,
                "partition_id": args.partition_id,
                "partitioning": args.partitioning,
                "dirichlet_alpha": args.dirichlet_alpha if args.partitioning == "dirichlet" else None,
                "device": args.device,
            },
            tags=["single_farm", "baseline", args.model_name, args.partitioning],
        )

    print("=" * 60)
    print("SINGLE-FARM TRAINING BASELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Partition: {args.partition_id} of {args.num_partitions}")
    print(f"Partitioning: {args.partitioning}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"WandB: {'enabled' if use_wandb else 'disabled'}")
    print()

    print("Loading partition...")
    partition = get_dataset_partition(
        num_partitions=args.num_partitions,
        partition_id=args.partition_id,
        dataset_name=args.dataset,
        partitioning=args.partitioning,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    train_ds = partition.with_transform(apply_train_transforms)
    print(f"Partition {args.partition_id} train samples: {len(train_ds)}")

    print("\nLoading full test set...")
    ds = load_dataset(args.dataset)
    test_ds = ds["test"].with_transform(apply_eval_transforms)
    print(f"Test samples: {len(test_ds)}")

    trainloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=4
    )
    testloader = DataLoader(
        test_ds, batch_size=args.batch_size,
        num_workers=4, pin_memory=True
    )

    print(f"\nInitializing {args.model_name}...")
    model = get_model(args.num_classes, args.model_name)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    print("\nTraining on single partition...")
    for epoch in range(args.epochs):
        train_loss = trainer(model, trainloader, optimizer, epochs=1, device=args.device)
        loss, accuracy = test(model, testloader, args.device)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, test_loss={loss:.4f}, accuracy={accuracy:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": loss,
                "accuracy": accuracy,
            })

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    results = test(model, testloader, args.device, return_detailed=True)
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"Precision (macro): {results['precision']:.4f}")
    print(f"Recall (macro): {results['recall']:.4f}")
    print(f"F1-Score (macro): {results['f1']:.4f}")

    print("\nDetailed Classification Report:")
    print(get_classification_report(results["labels"], results["predictions"], CLASS_NAMES[:args.num_classes]))

    if use_wandb:
        wandb.log({
            "final_accuracy": results["accuracy"],
            "final_precision": results["precision"],
            "final_recall": results["recall"],
            "final_f1": results["f1"],
            "final_loss": results["loss"],
        })
        wandb.summary["best_accuracy"] = results["accuracy"]
        wandb.summary["best_f1"] = results["f1"]
        wandb.finish()

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("This baseline represents the LOWER BOUND for FL performance.")
    print("Training on a single client's data limits generalization.")
    print("FL should achieve accuracy BETWEEN this and the centralized baseline.")


if __name__ == "__main__":
    main()
