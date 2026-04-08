"""
Figure 1: Hero chart only (right-hand side). Left side drawn in TikZ.

- 10 single-farm results (Swin-Small, non-IID Dirichlet α=0.5), dotted line for mean.
- FecalFed (ours) and Centralized bars; FecalFed is visually conspicuous. No FedAdam label.
- No top title (caption used).
Data from EXPERIMENT_REPORT.md §2.2.5 (Swin-Small), §3.4, §1.1.
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Swin-Small single-farm non-IID per-partition accuracy (%), report Table 2.2.5
SINGLE_FARM_ACC = [90.99, 57.98, 26.17, 91.22, 25.43, 83.64, 52.79, 69.38, 89.28, 61.74]
SINGLE_FARM_MEAN = 64.86
FECALFED_ACC = 90.31
CENTRALIZED_ACC = 95.10


def main():
    fig, ax = plt.subplots(figsize=(8, 6.5))

    n_single = len(SINGLE_FARM_ACC)
    x_single = list(range(n_single))
    x_fecalfed = n_single
    x_central = n_single + 1
    width = 0.7

    # 10 single-farm bars (neutral)
    bars_single = ax.bar(
        x_single, SINGLE_FARM_ACC,
        width=width, color="lightgray", edgecolor="gray", linewidth=0.8, label="Single-farm"
    )

    # Dotted horizontal line for single-farm average
    ax.axhline(SINGLE_FARM_MEAN, color="gray", linestyle="--", linewidth=1.5, label=f"Single-farm mean ({SINGLE_FARM_MEAN}%)")

    # FecalFed: conspicuous (bold color, thicker edge, no FedAdam text)
    ax.bar(
        x_fecalfed, FECALFED_ACC,
        width=width * 1.1, color="#e67e22", edgecolor="black", linewidth=2, label="FecalFed (ours)", zorder=3
    )

    # Centralized
    ax.bar(
        x_central, CENTRALIZED_ACC,
        width=width, color="#2ecc71", edgecolor="black", linewidth=0.8, label="Centralized"
    )

    # X labels: 0..9, FecalFed, Centralized — rotate to avoid overlap
    xticks = x_single + [x_fecalfed, x_central]
    xticklabels = [str(i) for i in range(10)] + ["FecalFed", "Centralized"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels just above the FecalFed and Centralized bars (right side)
    ax.text(x_fecalfed, FECALFED_ACC + 2, f"{FECALFED_ACC:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(x_central, CENTRALIZED_ACC + 2, f"{CENTRALIZED_ACC:.2f}%", ha="center", va="bottom", fontsize=9)

    # Legend in upper left so it doesn't cover the right-side value labels
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_hero.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
