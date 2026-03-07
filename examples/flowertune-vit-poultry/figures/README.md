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
| `fig2_dedup.py` | Data contamination: 16,513 → 8,770 flow | DATA_CURATION.md numbers | Flow diagram only; visual example needs raw_data/ |
| `fig3_dirichlet.py` | Non-IID Dirichlet α=0.5 stacked bar | HuggingFace `Dianyo/poultry-fecal-fl` + flwr_datasets | Requires network to load dataset |
| `fig4_curves.py` | FedAvg vs FedAdam optimization curves | CSV (round, strategy, accuracy) | Run FL with round logging first or provide CSV |
| `fig5_edge_scatter.py` | Edge efficiency: accuracy vs parameters | Report (hardcoded) | No run needed |

## Output

Figures are written to `figures/output/` (created automatically):

- `fig1_hero.png`
- `fig2_dedup.png`
- `fig3_dirichlet.png`
- `fig4_curves.png` (or empty if no CSV)
- `fig5_edge_scatter.png`

## Running

```bash
cd examples/flowertune-vit-poultry/figures
python fig1_hero.py
python fig2_dedup.py
python fig5_edge_scatter.py
# Figure 3 (needs dataset):
python fig3_dirichlet.py
# Figure 4 (needs round-by-round CSV from FL run):
python fig4_curves.py
```

## Figure 4 CSV format

If you add round-by-round logging to the FL server, save a CSV with columns:

- `round` (int)
- `strategy` ("FedAvg" or "FedAdam")
- `accuracy` (float, 0–1 or 0–100)
- Optionally `model` (e.g. "Swin-Small")

Place at `figures/data/fl_rounds_swinsmall.csv` or pass path via `--csv`.
