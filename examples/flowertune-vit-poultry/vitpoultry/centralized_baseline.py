"""Centralized training baseline for upper-bound accuracy.

This script trains the model on all data combined (no federation) to establish
the upper-bound performance that federated learning should approach.

Usage:
    python -m vitpoultry.centralized_baseline --dataset Dianyo/poultry-fecal-fl --epochs 10
"""

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

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
    args = parser.parse_args()

    print("=" * 60)
    print("CENTRALIZED TRAINING BASELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Classes: {args.num_classes}")
    print(f"Device: {args.device}")
    print()

    print("Loading dataset...")
    ds = load_dataset(args.dataset)
    train_ds = ds["train"].with_transform(apply_train_transforms)
    test_ds = ds["test"].with_transform(apply_eval_transforms)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=args.batch_size)

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

    save_path = f"centralized_{args.model_name}_{args.dataset.replace('/', '_')}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
