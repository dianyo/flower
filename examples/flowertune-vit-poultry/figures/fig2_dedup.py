"""
Figure 2: Data contamination — flow from 16,513 raw images to 8,770 unique after dual-hash dedup.

Numbers from docs/DATA_CURATION.md. No raw images required for the flow diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_N = 16_513
UNIQUE_N = 8_770
REMOVED_N = 7_743
RATE_PCT = 46.89


def main():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Box 1: Raw images
    b1 = mpatches.FancyBboxPatch((0.5, 1.5), 1.8, 2, boxstyle="round,pad=0.05", facecolor="lightcoral", edgecolor="black")
    ax.add_patch(b1)
    ax.text(1.4, 2.6, "Raw images", ha="center", va="center", fontsize=11)
    ax.text(1.4, 2.2, f"{RAW_N:,}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(1.4, 1.8, "(Zenodo + Roboflow)", ha="center", va="center", fontsize=8)

    # Arrow 1
    ax.annotate("", xy=(3.2, 2.5), xytext=(2.35, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(2.75, 2.85, "dual-hash", ha="center", fontsize=9)
    ax.text(2.75, 2.15, "(aHash + pHash)", ha="center", fontsize=8)

    # Pipeline box
    b2 = mpatches.FancyBboxPatch((3.2, 1.2), 2.2, 2.6, boxstyle="round,pad=0.05", facecolor="lightyellow", edgecolor="black")
    ax.add_patch(b2)
    ax.text(4.3, 3.4, "Deduplication", ha="center", va="center", fontsize=10)
    ax.text(4.3, 2.95, "aHash & pHash", ha="center", va="center", fontsize=9)
    ax.text(4.3, 2.5, "threshold ≤ 5", ha="center", va="center", fontsize=8)
    ax.text(4.3, 1.9, f"Removed: {REMOVED_N:,}", ha="center", va="center", fontsize=9)
    ax.text(4.3, 1.5, f"({RATE_PCT}%)", ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrow 2
    ax.annotate("", xy=(6.5, 2.5), xytext=(5.45, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Unique images
    b3 = mpatches.FancyBboxPatch((6.5, 1.5), 2.2, 2, boxstyle="round,pad=0.05", facecolor="lightgreen", edgecolor="black")
    ax.add_patch(b3)
    ax.text(7.6, 2.6, "Unique images", ha="center", va="center", fontsize=11)
    ax.text(7.6, 2.2, f"{UNIQUE_N:,}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(7.6, 1.8, "4-class dataset", ha="center", va="center", fontsize=8)

    ax.set_title("Figure 2: Data contamination — 46.89% deduplication (16,513 → 8,770 unique)", fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_dedup.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
