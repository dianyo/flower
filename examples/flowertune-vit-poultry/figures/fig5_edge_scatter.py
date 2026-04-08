"""
Figure 5: Edge efficiency — Federated accuracy vs model parameters (millions).

Data from EXPERIMENT_REPORT.md (FedAdam where better, else FedAvg).
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (params_M, accuracy_%), model_name — ViT and Swin only
VIT_MODELS = [
    (86, 90.02, "ViT-B/16"),
    (22, 89.28, "ViT-S/16"),
]
SWIN_MODELS = [
    (50, 90.31, "Swin-Small"),
    (28, 89.74, "Swin-Tiny"),
]
MODELS = VIT_MODELS + SWIN_MODELS


def main():
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        [m[0] for m in VIT_MODELS], [m[1] for m in VIT_MODELS],
        s=120, c="steelblue", edgecolor="black", linewidth=1, zorder=3, label="ViT"
    )
    ax.scatter(
        [m[0] for m in SWIN_MODELS], [m[1] for m in SWIN_MODELS],
        s=120, c="darkorange", edgecolor="black", linewidth=1, zorder=3, label="Swin"
    )
    for (p, a, n) in MODELS:
        ax.annotate(n, (p, a), xytext=(5, 5), textcoords="offset points", fontsize=9, ha="left")

    ax.set_xlabel("Model parameters (millions)", fontsize=11)
    ax.set_ylabel("Federated test accuracy (%)", fontsize=11)
    ax.set_ylim(88, 92)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_edge_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
