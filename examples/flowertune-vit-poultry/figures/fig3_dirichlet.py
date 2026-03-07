"""
Figure 3: Non-IID Dirichlet (alpha=0.5) — stacked bar of samples per client per class.

Uses flwr_datasets with Dianyo/poultry-fecal-fl and DirichletPartitioner to match
the experiment setup. Run from repo root so flwr_datasets is importable:

  cd /path/to/flower
  python examples/flowertune-vit-poultry/figures/fig3_dirichlet.py
"""

import sys
from pathlib import Path

# Allow importing flwr_datasets when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
if DATASETS_DIR.exists():
    sys.path.insert(0, str(DATASETS_DIR))

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Healthy", "Cocci", "NCD", "Salmo"]


def main():
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner
    from flwr_datasets.visualization import plot_label_distributions

    partitioner = DirichletPartitioner(
        num_partitions=10,
        partition_by="label",
        alpha=0.5,
        seed=42,
        min_partition_size=10,
    )
    fds = FederatedDataset(
        dataset="Dianyo/poultry-fecal-fl",
        partitioners={"train": partitioner},
    )
    # Trigger partition assignment (lazy)
    _ = fds.load_partition(0)
    train_partitioner = fds.partitioners["train"]

    figure, axis, dataframe = plot_label_distributions(
        partitioner=train_partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        title="Figure 3: Non-IID farm distribution (Dirichlet α=0.5)",
        legend=True,
        verbose_labels=True,
        legend_title="Class",
    )
    # Optional: use class names if dataframe has numeric column names
    if figure is not None:
        out = OUTPUT_DIR / "fig3_dirichlet.png"
        figure.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        if dataframe is not None and not dataframe.empty:
            print("Per-client counts (sample):")
            print(dataframe.head())


if __name__ == "__main__":
    main()
