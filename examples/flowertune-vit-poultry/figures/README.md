# Paper Figures

Scripts to reproduce the five figures for the FecalFed / poultry fecal disease FL paper.

## Setup

From the repo root (or `examples/flowertune-vit-poultry`):

```bash
pip install matplotlib pandas
# For Figure 3 (Dirichlet distribution) you need flwr_datasets and HuggingFace dataset access:
# pip install flwr_datasets datasets
```

## Scripts

| Script | Figure | Data source | Notes |
|--------|--------|-------------|--------|
| `fig1_hero.py` | Hero: FecalFed architecture + 3-bar comparison | Report (hardcoded) | No dataset or FL run needed |
| `fig2_dedup.py` | Data contamination: 16,513 → 8,770 flow | DATA_CURATION.md numbers | Flow diagram only |
| `fig2_dedup_example.py` | Dedup visual: 1 kept + 3 removed images | logs/deduplication_report_*.json + raw_data | Run after deduplicate; picks random cross-source group |
| `fig3_dirichlet.py` | Non-IID Dirichlet α=0.5 stacked bar | HuggingFace `Dianyo/poultry-fecal-fl` + flwr_datasets | Requires network to load dataset |
| `fig4_curves.py` | Ablation: accuracy & loss vs rounds | WandB runs or CSV | Use `--wandb-runs` with 3 run IDs, or CSV |
| `fig5_edge_scatter.py` | Edge efficiency: accuracy vs parameters | Report (hardcoded) | No run needed |

## Output

Figures are written to `figures/output/` (created automatically):

- `fig1_hero.png`
- `fig2_dedup.png`
- `fig2_dedup_example.png` (only if logs + raw_data exist; run `fig2_dedup_example.py`)
- `fig3_dirichlet.png`
- `fig4_curves.png` (or empty if no CSV)
- `fig5_edge_scatter.png`

## Running

```bash
cd examples/flowertune-vit-poultry/figures
python fig1_hero.py
python fig2_dedup.py
# Figure 2 example panel (needs logs + raw_data from deduplicate run):
python fig2_dedup_example.py
# Optional: --logs-dir /path/to/logs --raw-data /path/to/raw_data --seed 42
python fig5_edge_scatter.py
# Figure 3 (needs dataset):
python fig3_dirichlet.py
# Figure 4 (WandB ablation runs: accuracy + loss):
pip install wandb
python fig4_curves.py --wandb-runs 91r1o046 02gxoq3x z29oc1lf
# Figure 4 (from CSV):
python fig4_curves.py --csv path/to/rounds.csv
```

## Figure 4: WandB or CSV

**From WandB (ablation rounds 5, 10, 20):**

```bash
pip install wandb
python fig4_curves.py --wandb-runs 91r1o046 02gxoq3x z29oc1lf
```

Uses project `dianyo/flowertune-vit-poultry` by default. Override with `--wandb-project entity/project`. The script fetches each run’s history (round, accuracy, loss) and plots two subplots: **Test accuracy (%)** and **Test loss** vs communication rounds. Line labels come from each run’s `num_rounds` config (e.g. "5 rounds", "10 rounds", "20 rounds").

**From CSV (legacy):** columns `round`, optional `strategy`, `accuracy`; optionally `loss`. Single-plot accuracy only.

If you add round-by-round logging to the FL server, save a CSV with columns:

- `round` (int)
- `strategy` ("FedAvg" or "FedAdam")
- `accuracy` (float, 0–1 or 0–100)
- Optionally `model` (e.g. "Swin-Small")

Place at `figures/data/fl_rounds_swinsmall.csv` or pass path via `--csv`.
