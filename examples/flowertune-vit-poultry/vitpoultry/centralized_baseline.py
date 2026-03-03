"""Centralized training baseline for upper-bound accuracy.

This script trains the model on all data combined (no federation) to establish
the upper-bound performance that federated learning should approach.

Usage:
    python -m vitpoultry.centralized_baseline --dataset Dianyo/poultry-fecal-fl --epochs 10
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
    get_model,
    test,
    trainer,
)

CLASS_NAMES = ["healthy", "cocci", "ncd", "salmo"]


def main():
    parser = argparse.ArgumentParser(description="Centralized training baseline")
    parser.add_argument("--dataset", type=str, default="Dianyo/poultry-fecal-fl")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="vit_b_16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="flowertune-vit-poultry")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Skipping wandb logging.")

    if use_wandb:
        run_name = args.wandb_run_name or f"centralized_{args.model_name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "experiment_type": "centralized",
                "dataset": args.dataset,
                "model_name": args.model_name,
                "num_classes": args.num_classes,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "device": args.device,
            },
            tags=["centralized", "baseline", args.model_name],
        )

    print("=" * 60)
    print("CENTRALIZED TRAINING BASELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Classes: {args.num_classes}")
    print(f"Device: {args.device}")
    print(f"WandB: {'enabled' if use_wandb else 'disabled'}")
    print()

    print("Loading dataset...")
    ds = load_dataset(args.dataset)
    train_ds = ds["train"].with_transform(apply_train_transforms)
    test_ds = ds["test"].with_transform(apply_eval_transforms)

    print(f"Train samples: {len(train_ds)}")
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

    print("\nTraining...")
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

    save_path = f"centralized_{args.model_name}_{args.dataset.replace('/', '_')}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")

    if use_wandb:
        wandb.save(save_path)
        wandb.finish()


if __name__ == "__main__":
    main()
