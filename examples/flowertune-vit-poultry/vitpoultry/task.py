"""vitpoultry: Model, training, and dataset partitioning utilities."""

import torch
from tqdm import tqdm
from datasets import load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


def get_model(num_classes: int, model_name: str = "vit_b_16", finetune_mode: str = "head"):
    """Return a pretrained model with configurable fine-tuning mode.

    Args:
        num_classes: Number of output classes.
        model_name: One of the supported models (see SUPPORTED_MODELS).
        finetune_mode: "head" for head-only fine-tuning, "full" for full model fine-tuning.

    Returns:
        Model with trainable parameters based on finetune_mode.
    """
    full_finetune = finetune_mode == "full"
    
    if model_name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads[-1].in_features
        model.heads[-1] = torch.nn.Linear(in_features, num_classes)
        if full_finetune:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
            model.heads.requires_grad_(True)

    elif model_name == "vit_s_16":
        if not TIMM_AVAILABLE:
            raise ImportError("Install timm for ViT-S: pip install timm")
        model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)
        if not full_finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

    elif model_name == "mobilevit_s":
        if not TIMM_AVAILABLE:
            raise ImportError("Install timm for MobileViT: pip install timm")
        model = timm.create_model("mobilevit_s", pretrained=True, num_classes=num_classes)
        if not full_finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

    elif model_name == "mobilevitv2_100":
        if not TIMM_AVAILABLE:
            raise ImportError("Install timm for MobileViT v2: pip install timm")
        model = timm.create_model("mobilevitv2_100", pretrained=True, num_classes=num_classes)
        if not full_finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.parameters():
                param.requires_grad = True

    elif model_name == "mobilevitv2_150":
        if not TIMM_AVAILABLE:
            raise ImportError("Install timm for MobileViT v2: pip install timm")
        model = timm.create_model("mobilevitv2_150", pretrained=True, num_classes=num_classes)
        if not full_finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.parameters():
                param.requires_grad = True

    elif model_name == "swin_tiny":
        if not TIMM_AVAILABLE:
            raise ImportError("Install timm for Swin: pip install timm")
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
        if not full_finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

    elif model_name == "swin_small":
        if not TIMM_AVAILABLE:
            raise ImportError("Install timm for Swin: pip install timm")
        model = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=num_classes)
        if not full_finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

    else:
        supported = ["vit_b_16", "vit_s_16", "mobilevit_s", "mobilevitv2_100", "mobilevitv2_150", "swin_tiny", "swin_small"]
        raise ValueError(f"Unknown model: {model_name}. Supported: {supported}")

    return model


SUPPORTED_MODELS = {
    "vit_b_16": {"params": "86M", "head_attr": "heads", "source": "torchvision"},
    "vit_s_16": {"params": "22M", "head_attr": "head", "source": "timm"},
    "mobilevit_s": {"params": "5.6M", "head_attr": "head", "source": "timm"},
    "mobilevitv2_100": {"params": "4.9M", "head_attr": "head.fc", "source": "timm"},
    "mobilevitv2_150": {"params": "10.6M", "head_attr": "head.fc", "source": "timm"},
    "swin_tiny": {"params": "28M", "head_attr": "head", "source": "timm"},
    "swin_small": {"params": "50M", "head_attr": "head", "source": "timm"},
}


def calculate_model_bytes(state_dict: dict) -> int:
    """Calculate total bytes for a model state dict (for bandwidth tracking).
    
    Args:
        state_dict: PyTorch state dict (can be from model or specific layers).
    
    Returns:
        Total bytes (each param is float32 = 4 bytes).
    """
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


def get_trainable_params_count(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_finetune_layers(model, model_name: str = "vit_b_16", finetune_mode: str = "head"):
    """Get the finetune layers for a model based on finetune mode.
    
    Args:
        model: The model instance.
        model_name: Name of the model architecture.
        finetune_mode: "head" returns only the classification head, "full" returns entire model.
    
    Different model architectures have different attribute names for the head:
    - vit_b_16 (torchvision): model.heads
    - vit_s_16, mobilevit_s, swin_tiny, swin_small (timm): model.head
    - mobilevitv2_* (timm): model.head.fc
    """
    if finetune_mode == "full":
        return model
    
    if model_name == "vit_b_16":
        return model.heads
    elif model_name in ["vit_s_16", "mobilevit_s", "swin_tiny", "swin_small"]:
        return model.head
    elif model_name in ["mobilevitv2_100", "mobilevitv2_150"]:
        return model.head.fc
    else:
        raise ValueError(f"Unknown model: {model_name}. Check SUPPORTED_MODELS.")


def trainer(net, trainloader, optimizer, epochs, device: torch.device | str):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for batch in tqdm(trainloader, desc="Training"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]
            loss.backward()
            optimizer.step()

    return total_loss / total_samples


def test(net, testloader, device: torch.device | str, return_detailed: bool = False):
    """Validate the network on the entire test set.

    Args:
        net: Model to evaluate.
        testloader: DataLoader for test data.
        device: Device to run on.
        return_detailed: If True, return precision, recall, F1 per class.

    Returns:
        If return_detailed is False: (loss, accuracy)
        If return_detailed is True: dict with loss, accuracy, precision, recall, f1
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_preds = []
    all_labels = []

    net.to(device)
    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(testloader.dataset)

    if not return_detailed:
        return loss, accuracy

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def get_classification_report(labels, predictions, class_names=None):
    """Generate detailed classification report."""
    return classification_report(
        labels, predictions, target_names=class_names, zero_division=0
    )


fds = None


def get_dataset_partition(
    num_partitions: int,
    partition_id: int,
    dataset_name: str,
    partitioning: str = "iid",
    dirichlet_alpha: float = 0.5,
):
    """Get dataset and partition it across clients (Simulation Engine).

    Args:
        num_partitions: Number of client partitions.
        partition_id: ID of the partition to load.
        dataset_name: HuggingFace dataset name.
        partitioning: One of "iid" or "dirichlet" (non-iid).
        dirichlet_alpha: Concentration parameter for Dirichlet (lower = more heterogeneous).

    Returns:
        Dataset partition for the specified client.
    """
    global fds
    if fds is None:
        if partitioning == "iid":
            partitioner = IidPartitioner(num_partitions)
        elif partitioning == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=dirichlet_alpha,
            )
        else:
            raise ValueError(f"Unknown partitioning: {partitioning}. Choose: iid, dirichlet")

        fds = FederatedDataset(
            dataset=dataset_name, partitioners={"train": partitioner}
        )

    return fds.load_partition(partition_id)


def load_local_data(data_path: str, transform_fn):
    """Load a dataset partition from disk (Deployment Engine).

    Expects a HuggingFace dataset saved via `flwr-datasets create`.
    """
    dataset = load_from_disk(data_path)
    return dataset.with_transform(transform_fn)


def apply_eval_transforms(batch):
    """Apply standard evaluation transforms for ViT."""
    transforms = Compose(
        [
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def apply_train_transforms(batch, heavy_augmentation: bool = True):
    """Apply training transforms with optional heavy augmentation.

    Args:
        batch: Batch of images from HuggingFace dataset.
        heavy_augmentation: If True, use aggressive augmentation for better generalization.

    Returns:
        Batch with transformed images.
    """
    if heavy_augmentation:
        transforms = Compose(
            [
                RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotation(degrees=15),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transforms = Compose(
            [
                RandomResizedCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch
