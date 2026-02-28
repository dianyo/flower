"""Grad-CAM visualization for model interpretability.

This script generates Grad-CAM heatmaps to visualize what regions of the image
the model focuses on when making predictions. Useful for:
- Validating that the model focuses on relevant features (fecal matter)
- Identifying potential issues in model reasoning
- Building trust in model predictions

Usage:
    python -m vitpoultry.gradcam_viz --model-path final_model.pt --image-path sample.jpg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from vitpoultry.task import get_model

CLASS_NAMES = ["healthy", "cocci", "ncd", "salmo"]


def get_vit_target_layer(model):
    """Get the target layer for Grad-CAM on ViT models."""
    if hasattr(model, "encoder"):
        return model.encoder.layers[-1].ln_1
    if hasattr(model, "blocks"):
        return model.blocks[-1].norm1
    raise ValueError("Could not find target layer for Grad-CAM")


def reshape_transform(tensor, height=14, width=14):
    """Reshape ViT attention output for Grad-CAM visualization."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def load_and_preprocess_image(image_path: str):
    """Load image and apply preprocessing transforms."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img) / 255.0

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0)

    img_resized = np.array(img.resize((224, 224))) / 255.0

    return input_tensor, img_resized


def generate_gradcam(
    model,
    input_tensor,
    target_class: int = None,
):
    """Generate Grad-CAM heatmap for the input image."""
    target_layer = get_vit_target_layer(model)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    else:
        targets = None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]


def visualize_gradcam(
    image_np: np.ndarray,
    grayscale_cam: np.ndarray,
    prediction: int,
    confidence: float,
    true_label: int = None,
    save_path: str = None,
):
    """Create and save/display Grad-CAM visualization."""
    visualization = show_cam_on_image(image_np.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    title = f"Prediction: {CLASS_NAMES[prediction]} ({confidence:.2%})"
    if true_label is not None:
        title += f"\nTrue: {CLASS_NAMES[true_label]}"
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def batch_visualize(
    model,
    image_dir: str,
    output_dir: str,
    num_images: int = 10,
):
    """Generate Grad-CAM visualizations for multiple images."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(image_dir.rglob(ext))

    images = images[:num_images]
    print(f"Processing {len(images)} images...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    for img_path in images:
        input_tensor, img_np = load_and_preprocess_image(str(img_path))
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = probs.max(dim=1)

        grayscale_cam = generate_gradcam(model, input_tensor, prediction.item())

        save_path = output_dir / f"{img_path.stem}_gradcam.png"
        visualize_gradcam(
            img_np,
            grayscale_cam,
            prediction.item(),
            confidence.item(),
            save_path=str(save_path),
        )


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image-path", type=str, help="Path to single image")
    parser.add_argument("--image-dir", type=str, help="Directory of images for batch processing")
    parser.add_argument("--output-dir", type=str, default="gradcam_outputs")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="vit_b_16")
    parser.add_argument("--num-images", type=int, default=10, help="Max images for batch mode")
    args = parser.parse_args()

    print("=" * 60)
    print("GRAD-CAM VISUALIZATION")
    print("=" * 60)

    print(f"Loading model from {args.model_path}...")
    model = get_model(args.num_classes, args.model_name)

    state_dict = torch.load(args.model_path, map_location="cpu")
    if "heads" in str(list(state_dict.keys())[0]) or len(state_dict) < 10:
        model.heads.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if args.image_dir:
        batch_visualize(model, args.image_dir, args.output_dir, args.num_images)
    elif args.image_path:
        input_tensor, img_np = load_and_preprocess_image(args.image_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = probs.max(dim=1)

        print(f"\nPrediction: {CLASS_NAMES[prediction.item()]} ({confidence.item():.2%})")

        grayscale_cam = generate_gradcam(model, input_tensor, prediction.item())

        Path(args.output_dir).mkdir(exist_ok=True)
        save_path = Path(args.output_dir) / f"{Path(args.image_path).stem}_gradcam.png"
        visualize_gradcam(
            img_np,
            grayscale_cam,
            prediction.item(),
            confidence.item(),
            save_path=str(save_path),
        )
    else:
        print("Error: Provide either --image-path or --image-dir")


if __name__ == "__main__":
    main()
