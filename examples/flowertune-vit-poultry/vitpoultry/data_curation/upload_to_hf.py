"""Upload datasets to HuggingFace Hub.

This script builds and uploads datasets to HuggingFace:
1. Dianyo/poultry-health (binary) - from Mendeley Nigeria
2. Dianyo/poultry-fecal-fl (4-class) - from Zenodo + Roboflow

Prerequisites:
    - Run download_datasets.py first
    - Run deduplicate.py (for 4-class dataset)
    - Login: huggingface-cli login

Usage:
    python -m vitpoultry.data_curation.upload_to_hf --dataset binary
    python -m vitpoultry.data_curation.upload_to_hf --dataset 4class
    python -m vitpoultry.data_curation.upload_to_hf --dataset all
"""

import argparse
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Image

RAW_DATA_DIR = Path("raw_data")
DEDUP_DATA_DIR = Path("deduplicated_data")

LABEL_MAP_BINARY = {
    "healthy": 0,
    "unhealthy": 1,
    "sick": 1,
    "diseased": 1,
}

LABEL_MAP_4CLASS = {
    "healthy": 0,
    "cocci": 1,
    "coccidiosis": 1,
    "ncd": 2,
    "newcastle": 2,
    "salmo": 3,
    "salmonella": 3,
}

CLASS_NAMES_BINARY = ["healthy", "unhealthy"]
CLASS_NAMES_4CLASS = ["healthy", "cocci", "ncd", "salmo"]


def build_binary_dataset(mendeley_dir: Path) -> DatasetDict:
    """Build binary (healthy/unhealthy) dataset from Mendeley download."""
    print(f"\nScanning {mendeley_dir} for images...")
    
    records = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in mendeley_dir.rglob(ext):
            path_str = str(img_path).lower()
            
            label = None
            if "healthy" in path_str and "unhealthy" not in path_str:
                label = 0
            elif any(x in path_str for x in ["unhealthy", "sick", "diseased"]):
                label = 1
            else:
                parent = img_path.parent.name.lower()
                if "healthy" in parent and "un" not in parent:
                    label = 0
                elif any(x in parent for x in ["unhealthy", "sick", "diseased"]):
                    label = 1
            
            if label is not None:
                records.append({
                    "image": str(img_path.absolute()),
                    "label": label,
                })
    
    if not records:
        print("No valid images found!")
        return None
    
    print(f"Found {len(records)} images")
    label_counts = Counter(r["label"] for r in records)
    print(f"  healthy: {label_counts[0]}, unhealthy: {label_counts[1]}")
    
    import random
    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * 0.8)
    
    train_ds = Dataset.from_list(records[:split_idx]).cast_column("image", Image())
    test_ds = Dataset.from_list(records[split_idx:]).cast_column("image", Image())
    
    return DatasetDict({"train": train_ds, "test": test_ds})


def build_4class_dataset(data_dir: Path) -> DatasetDict:
    """Build 4-class dataset from deduplicated sources."""
    import re
    
    print(f"\nScanning {data_dir} for images...")
    
    records = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in data_dir.rglob(ext):
            path_str = str(img_path).lower()
            
            label = None
            for label_name, label_id in LABEL_MAP_4CLASS.items():
                if label_name.lower() in path_str:
                    label = label_id
                    break
            
            if label is None:
                continue
            
            # Extract farm_id
            farm_id = "unknown"
            match = re.search(r"[Ff]arm[_\-]?(\d+)", str(img_path))
            if match:
                farm_id = f"farm_{int(match.group(1)):02d}"
            elif "roboflow" in path_str:
                farm_id = f"roboflow_{hash(img_path.stem) % 10:02d}"
            elif "zenodo" in path_str:
                farm_id = "zenodo_lab"
            
            records.append({
                "image": str(img_path.absolute()),
                "label": label,
                "farm_id": farm_id,
            })
    
    if not records:
        print("No valid images found!")
        return None
    
    print(f"Found {len(records)} images")
    label_counts = Counter(r["label"] for r in records)
    for i, name in enumerate(CLASS_NAMES_4CLASS):
        print(f"  {name}: {label_counts.get(i, 0)}")
    
    farm_ids = sorted(set(r["farm_id"] for r in records))
    print(f"Unique farms: {len(farm_ids)}")
    
    test_farm_count = max(1, len(farm_ids) // 5)
    test_farms = set(farm_ids[:test_farm_count])
    
    train_records = [r for r in records if r["farm_id"] not in test_farms]
    test_records = [r for r in records if r["farm_id"] in test_farms]
    
    train_ds = Dataset.from_list(train_records).cast_column("image", Image())
    test_ds = Dataset.from_list(test_records).cast_column("image", Image())
    
    return DatasetDict({"train": train_ds, "test": test_ds})


def upload_dataset(dataset: DatasetDict, repo_id: str, class_names: list) -> None:
    """Upload dataset to HuggingFace Hub."""
    print(f"\n{'=' * 60}")
    print(f"UPLOADING TO: {repo_id}")
    print(f"{'=' * 60}")
    
    print("\nDataset summary:")
    for split_name, split_ds in dataset.items():
        print(f"  {split_name}: {len(split_ds)} images")
        label_counts = Counter(split_ds["label"])
        for i, name in enumerate(class_names):
            print(f"    {name}: {label_counts.get(i, 0)}")
    
    confirm = input(f"\nUpload to {repo_id}? [y/N]: ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return
    
    print("\nUploading...")
    dataset.push_to_hub(repo_id, private=False)
    print(f"\n✓ Uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to HuggingFace")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["binary", "4class", "all"],
        help="Which dataset to upload: binary, 4class, or all",
    )
    parser.add_argument(
        "--repo-binary",
        type=str,
        default="Dianyo/poultry-health",
        help="HuggingFace repo for binary dataset",
    )
    parser.add_argument(
        "--repo-4class",
        type=str,
        default="Dianyo/poultry-fecal-fl",
        help="HuggingFace repo for 4-class dataset",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("HUGGINGFACE DATASET UPLOAD")
    print("=" * 60)
    print("\nMake sure you're logged in: huggingface-cli login")
    
    if args.dataset in ["binary", "all"]:
        mendeley_dir = RAW_DATA_DIR / "mendeley_nigeria"
        if not mendeley_dir.exists():
            print(f"\nError: {mendeley_dir} not found")
            print("Run download_datasets.py first")
        else:
            ds_binary = build_binary_dataset(mendeley_dir)
            if ds_binary:
                upload_dataset(ds_binary, args.repo_binary, CLASS_NAMES_BINARY)
    
    if args.dataset in ["4class", "all"]:
        if not DEDUP_DATA_DIR.exists():
            print(f"\nError: {DEDUP_DATA_DIR} not found")
            print("Run deduplicate.py first")
        else:
            ds_4class = build_4class_dataset(DEDUP_DATA_DIR)
            if ds_4class:
                upload_dataset(ds_4class, args.repo_4class, CLASS_NAMES_4CLASS)


if __name__ == "__main__":
    main()
