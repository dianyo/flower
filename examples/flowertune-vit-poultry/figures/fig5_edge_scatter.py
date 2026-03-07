"""
Figure 5: Edge efficiency — Federated accuracy vs model parameters (millions).

Data from EXPERIMENT_REPORT.md (FedAdam where better, else FedAvg).
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (params_M, accuracy_%), model_name
MODELS = [
    (86, 90.02, "ViT-B/16"),
    (50, 90.31, "Swin-Small"),
    (28, 89.74, "Swin-Tiny"),
    (22, 89.28, "ViT-S/16"),
]
# Optional: MobileViT (show poor FL)
MOBILE = [
    (10.6, 30.33, "MobileViT-v2-1.5"),
    (5.6, 8.21, "MobileViT-v1-S"),
    (4.9, 33.30, "MobileViT-v2-1.0"),
]


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    params = [m[0] for m in MODELS]
    accs = [m[1] for m in MODELS]
    names = [m[2] for m in MODELS]

    ax.scatter(params, accs, s=120, c="steelblue", edgecolor="black", linewidth=1, zorder=3)
    for (p, a, n) in MODELS:
        ax.annotate(n, (p, a), xytext=(5, 5), textcoords="offset points", fontsize=9, ha="left")

    # Optional: light scatter for MobileViT
    if MOBILE:
        px = [m[0] for m in MOBILE]
        py = [m[1] for m in MOBILE]
        ax.scatter(px, py, s=60, c="lightcoral", edgecolor="gray", alpha=0.8, label="MobileViT (head-only FL)")

    ax.set_xlabel("Model parameters (millions)", fontsize=11)
    ax.set_ylabel("Federated test accuracy (%)", fontsize=11)
    ax.set_title("Figure 5: Edge efficiency — Swin-Tiny sweet spot (28M params, 89.74%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    if MOBILE:
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_edge_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
