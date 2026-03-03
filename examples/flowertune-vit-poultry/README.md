---
tags: [finetuning, vision, fds, fedprox, grad-cam]
dataset: [Poultry Fecal Health (4-class)]
framework: [torch, torchvision, timm]
---

# Federated Finetuning of Vision Transformers for Poultry Disease Classification

This example demonstrates privacy-preserving federated learning for poultry fecal disease classification targeting the **CVPR Agriculture-Vision 2026 Workshop**. It supports:

- **4-class classification**: Healthy, Coccidiosis, Newcastle Disease (NCD), Salmonella
- **Multiple models**: ViT-B-16, MobileViT-S, Swin Transformer Tiny
- **FL strategies**: FedAvg and FedProx (for non-IID data)
- **Data partitioning**: IID and Dirichlet (heterogeneous)
- **Model interpretability**: Grad-CAM attention visualization

The system finetunes only the classification head on the [Dianyo/poultry-fecal-fl](https://huggingface.co/datasets/Dianyo/poultry-fecal-fl) dataset using [Flower Datasets](https://flower.ai/docs/datasets/). Each client needs minimal VRAM (~1 GB at batch size 32).

## Set up the project

### Fetch the app

```shell
flwr new @dianyo/vitpoultry
```

This will create a new directory called `flowertune-vit-poultry` with the following structure:

```shell
flowertune-vit-poultry
├── vitpoultry
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install using `uv` (recommended) or pip:

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> **Tip:** Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations, how to use more virtual SuperNodes, and how to configure CPU/GPU usage in your ClientApp.

```bash
flwr run .
```

You can override settings defined in `pyproject.toml`:

```bash
# FedAvg with IID partitioning (default)
flwr run . --run-config "num-server-rounds=10 strategy=fedavg partitioning=iid"

# FedProx with non-IID Dirichlet partitioning
flwr run . --run-config "strategy=fedprox partitioning=dirichlet dirichlet-alpha=0.3 proximal-mu=0.1"

# Using MobileViT for edge deployment
flwr run . --run-config "model-name=mobilevit_s"
```

If your system has a GPU you can make use of it:

```bash
flwr run .
```

### Run with the Deployment Engine

To run this app using Flower's Deployment Engine we recommend first creating some demo data using [Flower Datasets](https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html). For example:

```bash
# Install Flower Datasets
pip install "flwr-datasets[vision]"

# Create dataset partitions and save them to disk
flwr-datasets create Dianyo/poultry-health --num-partitions 2 --out-dir demo_data
```

The above command will create two IID partitions of the poultry health dataset and save them in a `demo_data` directory. Next, you can pass one partition to each of your SuperNodes like this:

```bash
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/demo_data/partition_0"
```

Finally, ensure the environment of each SuperNode has all dependencies installed. Then, launch the run via `flwr run` but pointing to a SuperLink connection that specifies the SuperLink your SuperNode is connected to:

```bash
flwr run . <SUPERLINK-CONNECTION> --stream
```

> **Tip:** Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

## Running Experiments

Use the automated experiment runner to systematically run all experiments with wandb logging:

```bash
source dev.env  # Load API keys (WANDB_API_KEY, HF_TOKEN)

# List all available experiments
python scripts/run_experiments.py --list

# Run baseline experiments (centralized + single-farm)
python scripts/run_experiments.py --phases baseline --wandb --batch-size 256

# Run federated learning experiments  
python scripts/run_experiments.py --phases federated --wandb --batch-size 256

# Run ALL experiments
python scripts/run_experiments.py --phases all --wandb --batch-size 256

# Run a specific experiment
python scripts/run_experiments.py --experiment centralized_vit --wandb --batch-size 256
```

Results are saved to `experiment_results/<timestamp>/` with JSON, CSV, and individual logs.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the complete experiment guide.

## Baselines

Establish upper and lower performance bounds before FL experiments:

```bash
# Centralized baseline (upper bound) - with wandb logging
python -m vitpoultry.centralized_baseline --epochs 10 --batch-size 256 --wandb

# Single-farm baseline (lower bound)
python -m vitpoultry.single_farm_baseline --partition-id 0 --batch-size 256 --wandb
```

## Model Interpretability

Generate Grad-CAM visualizations to understand model predictions:

```bash
# Single image
python -m vitpoultry.gradcam_viz --model-path final_model.pt --image-path sample.jpg

# Batch processing
python -m vitpoultry.gradcam_viz --model-path final_model.pt --image-dir test_images/ --num-images 20
```

## Data Curation

To build the consolidated 4-class dataset from multiple sources:

```bash
# 1. Download raw datasets
python -m vitpoultry.data_curation.download_datasets

# 2. Deduplicate images
python -m vitpoultry.data_curation.deduplicate

# 3. Merge and upload to HuggingFace
python -m vitpoultry.data_curation.merge_and_upload
```
