"""vitpoultry: A Flower / PyTorch app with Vision Transformers for Poultry Health."""

import torch
from datasets import Dataset, load_dataset
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader

from vitpoultry.task import apply_eval_transforms, get_model, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
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

    model = get_model(num_classes, model_name)
    finetune_layers = model.heads
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

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(test_set, num_classes, model_name),
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def get_evaluate_fn(
    centralized_testset: Dataset,
    num_classes: int,
    model_name: str = "vit_b_16",
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Use the entire test set for evaluation."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model(num_classes, model_name)
        finetune_layers = model.heads
        finetune_layers.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        model.to(device)

        testset = centralized_testset.with_transform(apply_eval_transforms)
        testloader = DataLoader(testset, batch_size=128)

        loss, accuracy = test(model, testloader, device=device)

        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return evaluate
