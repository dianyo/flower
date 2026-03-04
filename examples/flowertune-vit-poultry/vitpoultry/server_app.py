"""vitpoultry: A Flower / PyTorch app with Vision Transformers for Poultry Health."""

import os

import torch
from datasets import Dataset, load_dataset
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from vitpoultry.task import apply_eval_transforms, get_finetune_layers, get_model, test

app = ServerApp()

_wandb_initialized = False


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _wandb_initialized

    dataset_name = context.run_config["dataset-name"]
    dataset = load_dataset(dataset_name)
    if "test" in dataset:
        test_set = dataset["test"]
    elif "validation" in dataset:
        test_set = dataset["validation"]
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' has no 'test' or 'validation' split. "
            f"Available splits: {list(dataset.keys())}"
        )
    num_rounds = context.run_config["num-server-rounds"]

    num_classes = context.run_config["num-classes"]
    model_name = context.run_config.get("model-name", "vit_b_16")
    strategy_name = context.run_config.get("strategy", "fedavg")
    proximal_mu = context.run_config.get("proximal-mu", 0.1)
    partitioning = context.run_config.get("partitioning", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    use_wandb = context.run_config.get("wandb", False)

    if use_wandb and WANDB_AVAILABLE and not _wandb_initialized:
        run_name = f"fl_{strategy_name}_{partitioning}_{model_name}"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "flowertune-vit-poultry"),
            name=run_name,
            config={
                "experiment_type": "federated",
                "strategy": strategy_name,
                "model_name": model_name,
                "num_rounds": num_rounds,
                "partitioning": partitioning,
                "dirichlet_alpha": dirichlet_alpha if partitioning == "dirichlet" else None,
                "proximal_mu": proximal_mu if strategy_name == "fedprox" else None,
                "dataset": dataset_name,
                "num_classes": num_classes,
            },
            tags=["federated", strategy_name, model_name, partitioning],
        )
        _wandb_initialized = True

    model = get_model(num_classes, model_name)
    finetune_layers = get_finetune_layers(model, model_name)
    arrays = ArrayRecord(finetune_layers.state_dict())

    if strategy_name == "fedavg":
        strategy = FedAvg(
            fraction_train=0.5,
            fraction_evaluate=0.0,
        )
    elif strategy_name == "fedprox":
        strategy = FedProx(
            fraction_train=0.5,
            fraction_evaluate=0.0,
            proximal_mu=proximal_mu,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose: fedavg, fedprox")

    print(f"Starting FL with strategy={strategy_name}, model={model_name}")
    print(f"WandB: {'enabled' if (use_wandb and WANDB_AVAILABLE) else 'disabled'}")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(test_set, num_classes, model_name, use_wandb and WANDB_AVAILABLE),
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    if use_wandb and WANDB_AVAILABLE and _wandb_initialized:
        wandb.save("final_model.pt")
        wandb.finish()
        _wandb_initialized = False


def get_evaluate_fn(
    centralized_testset: Dataset,
    num_classes: int,
    model_name: str = "vit_b_16",
    use_wandb: bool = False,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Use the entire test set for evaluation."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model(num_classes, model_name)
        finetune_layers = get_finetune_layers(model, model_name)
        finetune_layers.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        model.to(device)

        testset = centralized_testset.with_transform(apply_eval_transforms)
        testloader = DataLoader(testset, batch_size=128, num_workers=4, pin_memory=True)

        loss, accuracy = test(model, testloader, device=device)

        print(f"Round {server_round}: accuracy={accuracy:.4f}, loss={loss:.4f}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "round": server_round,
                "accuracy": accuracy,
                "loss": loss,
            })

        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return evaluate
