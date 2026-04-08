"""
Figure 4: Federated optimization curves — accuracy and loss over communication rounds.

Can use either:
  A) WandB: fetch data from three ablation runs (rounds 5, 10, 20).
  B) CSV: round-by-round data from a file (legacy).

Usage (WandB, recommended):
  python fig4_curves.py --wandb-runs 91r1o046 02gxoq3x z29oc1lf
  python fig4_curves.py --wandb-runs 91r1o046 02gxoq3x z29oc1lf --wandb-project dianyo/flowertune-vit-poultry

Usage (CSV):
  python fig4_curves.py --csv path/to/fl_rounds.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default WandB project (entity/project) from your run paths
DEFAULT_WANDB_PROJECT = "dianyo/flowertune-vit-poultry"


def fetch_wandb_runs(project: str, run_ids: list[str]):
    """Fetch run history (round, accuracy, loss) and num_rounds from WandB."""
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is required for --wandb-runs. Install with: pip install wandb")

    api = wandb.Api()
    runs_data = []
    for run_id in run_ids:
        run_path = f"{project}/{run_id}"
        run = api.run(run_path)
        num_rounds = run.config.get("num_rounds")
        if num_rounds is None:
            num_rounds = run_id  # fallback label
        history = run.history()
        if history is None or history.empty:
            print(f"Warning: no history for {run_path}")
            continue
        # Prefer logged "round"; fallback to _step
        if "round" not in history.columns and "_step" in history.columns:
            history = history.rename(columns={"_step": "round"})
        if "round" not in history.columns:
            print(f"Warning: no round/_step in {run_path}, columns: {list(history.columns)}")
            continue
        history = history.dropna(subset=["round"]).drop_duplicates(subset=["round"]).sort_values("round")
        if "accuracy" not in history.columns or "loss" not in history.columns:
            print(f"Warning: missing accuracy/loss in {run_path}, columns: {list(history.columns)}")
            continue
        n_rounds = num_rounds if isinstance(num_rounds, int) else int(num_rounds) if num_rounds is not None else None
        label = f"{n_rounds} rounds" if n_rounds is not None else run_id
        runs_data.append({"label": label, "num_rounds": n_rounds, "history": history})
    return runs_data


def plot_from_wandb(runs_data: list, out_path: Path):
    """Two subplots: accuracy vs round, loss vs round."""
    if not runs_data:
        print("No run data to plot.")
        return

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for run in runs_data:
        h = run["history"]
        r = h["round"].astype(int)
        acc = h["accuracy"].values
        loss = h["loss"].values
        if acc.max() <= 1.0:
            acc = acc * 100
        ax_acc.plot(r, acc, label=run["label"], linewidth=2, marker="o", markersize=4)
        ax_loss.plot(r, loss, label=run["label"], linewidth=2, marker="o", markersize=4)

    ax_acc.set_ylabel("Test accuracy (%)", fontsize=11)
    ax_acc.set_ylim(0, 100)
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.grid(True, alpha=0.3)

    ax_loss.set_xlabel("Communication rounds", fontsize=11)
    ax_loss.set_ylabel("Test loss", fontsize=11)
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_from_csv(csv_path: Path, out_path: Path):
    """Single plot: accuracy vs round (CSV with optional strategy column)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "accuracy" in df.columns and df["accuracy"].max() <= 1.0:
        df["accuracy"] = df["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    if "strategy" in df.columns:
        for strat in df["strategy"].unique():
            sub = df[df["strategy"] == strat].sort_values("round")
            if len(sub) > 0:
                ax.plot(sub["round"], sub["accuracy"], label=strat, linewidth=2, marker="o", markersize=4)
    else:
        df = df.sort_values("round")
        ax.plot(df["round"], df["accuracy"], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Communication rounds", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="CSV with round, (strategy,) accuracy, (loss)")
    ap.add_argument("--wandb-runs", nargs=3, metavar="RUN_ID", help="Three WandB run IDs (ablation 5, 10, 20 rounds)")
    ap.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="WandB entity/project")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "fig4_curves.png", help="Output figure path")
    args = ap.parse_args()

    if args.wandb_runs:
        runs_data = fetch_wandb_runs(args.wandb_project, args.wandb_runs)
        plot_from_wandb(runs_data, args.out)
        return

    csv_path = args.csv or (DATA_DIR / "fl_rounds_swinsmall.csv")
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Use --wandb-runs 91r1o046 02gxoq3x z29oc1lf to plot from WandB, or create the CSV.")
        return
    plot_from_csv(csv_path, args.out)


if __name__ == "__main__":
    main()
