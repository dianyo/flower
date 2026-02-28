"""vitpoultry: A Flower / PyTorch app with Vision Transformers for Poultry Health."""

import copy
import warnings

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from vitpoultry.task import (
    apply_train_transforms,
    get_dataset_partition,
    get_model,
    load_local_data,
)

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


def trainer_fedprox(
    net,
    trainloader,
    optimizer,
    epochs,
    device,
    proximal_mu: float,
    global_params: dict,
):
    """Train with FedProx proximal term to stay close to global model.

    The proximal term adds (mu/2) * ||w - w_global||^2 to the loss,
    which prevents local models from drifting too far from the global model.
    This is especially useful for non-IID data distributions.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()

            outputs = net(images)
            ce_loss = criterion(outputs, labels)

            proximal_loss = 0.0
            for name, param in net.named_parameters():
                if param.requires_grad and name in global_params:
                    global_param = global_params[name].to(device)
                    proximal_loss += ((param - global_param) ** 2).sum()

            loss = ce_loss + (proximal_mu / 2.0) * proximal_loss

            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]
            loss.backward()
            optimizer.step()

    return total_loss / total_samples


def trainer_standard(net, trainloader, optimizer, epochs, device):
    """Standard training without proximal term (FedAvg)."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]
            loss.backward()
            optimizer.step()

    return total_loss / total_samples


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]
    num_classes = context.run_config["num-classes"]
    model_name = context.run_config.get("model-name", "vit_b_16")
    strategy = context.run_config.get("strategy", "fedavg")
    proximal_mu = context.run_config.get("proximal-mu", 0.1)
    partitioning = context.run_config.get("partitioning", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)

    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        dataset_name = context.run_config["dataset-name"]
        trainpartition = get_dataset_partition(
            num_partitions,
            partition_id,
            dataset_name,
            partitioning=partitioning,
            dirichlet_alpha=dirichlet_alpha,
        )
        trainset = trainpartition.with_transform(apply_train_transforms)
    elif "data-path" in context.node_config:
        data_path = context.node_config["data-path"]
        trainset = load_local_data(data_path, apply_train_transforms)
    else:
        raise ValueError(
            "Could not determine data loading mode. Expected either "
            "'partition-id'/'num-partitions' (Simulation) or "
            "'data-path' (Deployment) in node_config, but got: "
            f"{list(context.node_config.keys())}"
        )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=2, shuffle=True
    )

    model = get_model(num_classes, model_name)
    finetune_layers = model.heads
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    finetune_layers.load_state_dict(global_state_dict, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    if strategy == "fedprox":
        global_params = {name: param.clone() for name, param in model.named_parameters()}
        avg_train_loss = trainer_fedprox(
            model, trainloader, optimizer, epochs=1, device=device,
            proximal_mu=proximal_mu, global_params=global_params
        )
    else:
        avg_train_loss = trainer_standard(
            model, trainloader, optimizer, epochs=1, device=device
        )

    model_record = ArrayRecord(finetune_layers.state_dict())
    metrics = {
        "train_loss": avg_train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
