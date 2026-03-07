"""
Figure 1: Hero figure — FecalFed architecture (left) + 3-bar comparison chart (right).

Data from EXPERIMENT_REPORT.md:
- Single-Farm Non-IID: 60.97%
- FecalFed (FedAdam): 90.31%
- Centralized: 95.10%
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_architecture(ax):
    """Simple diagram: 10 farm clients → Cloud (Flower) with padlock."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Clients 1–10 (bottom row)
    for i in range(10):
        x = 0.5 + i * 0.9
        box = mpatches.FancyBboxPatch((x, 0.5), 0.7, 0.8, boxstyle="round,pad=0.02", facecolor="lightblue", edgecolor="gray")
        ax.add_patch(box)
        ax.text(x + 0.35, 0.9, str(i + 1), ha="center", va="center", fontsize=8)
        ax.text(x + 0.35, 0.6, "Swin", ha="center", va="center", fontsize=6)

    # Arrows up to server
    for i in range(10):
        x = 0.5 + i * 0.9 + 0.35
        ax.annotate("", xy=(5, 6.5), xytext=(x, 1.35), arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Cloud server
    server = mpatches.FancyBboxPatch((3.5, 6), 3, 1.2, boxstyle="round,pad=0.05", facecolor="wheat", edgecolor="black")
    ax.add_patch(server)
    ax.text(5, 6.6, "Cloud Server (Flower)", ha="center", va="center", fontsize=10)
    ax.text(5, 5.5, "Model weights only", ha="center", va="center", fontsize=8, style="italic")
    # Padlock
    lock = mpatches.FancyBboxPatch((4.6, 5.0), 0.8, 0.45, facecolor="lightgray", edgecolor="black")
    ax.add_patch(lock)
    ax.text(5, 5.22, "Privacy", ha="center", va="center", fontsize=7)
    ax.text(5, 4.6, "No raw data shared", ha="center", va="center", fontsize=7)


def draw_bar_chart(ax):
    """Three bars: Single-Farm Non-IID, FecalFed (FedAdam), Centralized."""
    labels = ["Single-Farm\nNon-IID", "FecalFed\n(FedAdam)", "Centralized"]
    accs = [60.97, 90.31, 95.10]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    x = range(len(labels))
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(0, color="gray", linewidth=0.5)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)


def main():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
    draw_architecture(ax_left)
    ax_left.set_title("FecalFed: Federated learning across farms", fontsize=11)
    draw_bar_chart(ax_right)
    ax_right.set_title("Impact: FL vs single-farm vs centralized", fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_hero.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
