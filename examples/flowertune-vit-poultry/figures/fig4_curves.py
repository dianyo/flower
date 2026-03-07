"""
Figure 4: Federated optimization curves — FedAvg vs FedAdam (Swin-Small) over rounds.

Reads round-by-round accuracy from a CSV. Generate the CSV by running FL with
round logging (e.g. server_app saving round, strategy, accuracy to file).

Expected CSV columns: round, strategy, accuracy
  - round: int (1..20)
  - strategy: "FedAvg" or "FedAdam"
  - accuracy: float in [0, 1] or [0, 100]

Usage:
  python fig4_curves.py
  python fig4_curves.py --csv path/to/fl_rounds.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "fl_rounds_swinsmall.csv", help="CSV with round, strategy, accuracy")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "fig4_curves.png", help="Output figure path")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        print("Create it by running FL with round-by-round logging, or use placeholder data.")
        # Write a minimal placeholder CSV so the script can still produce a figure template
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w") as f:
            f.write("round,strategy,accuracy\n")
            for r in range(1, 21):
                # Placeholder curves (replace with real FL round logging)
                f.write(f"{r},FedAvg,{0.75 + 0.007 * (r / 20.):.4f}\n")
                f.write(f"{r},FedAdam,{0.78 + 0.008 * (r / 20.):.4f}\n")
        print(f"Wrote placeholder {args.csv} (replace with real data and re-run).")

    import pandas as pd
    df = pd.read_csv(args.csv)
    if "accuracy" in df.columns and df["accuracy"].max() <= 1.0:
        df["accuracy"] = df["accuracy"] * 100

    strategies = df["strategy"].unique() if "strategy" in df.columns else []
    if len(strategies) == 0:
        print("No 'strategy' column in CSV. Expected columns: round, strategy, accuracy")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("round")
        if len(sub) == 0:
            continue
        ax.plot(sub["round"], sub["accuracy"], label=strat, linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Communication rounds", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("Figure 4: Federated optimization curves (Swin-Small)", fontsize=11)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
