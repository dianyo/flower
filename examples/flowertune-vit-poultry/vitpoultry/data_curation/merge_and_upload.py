"""Merge deduplicated datasets and upload to HuggingFace.

This script:
1. Merges deduplicated images from multiple sources
2. Standardizes labels to 4-class (healthy/cocci/ncd/salmo)
3. Extracts farm_id metadata for natural federated splits
4. Creates train/test splits by farm_id
5. Uploads to HuggingFace as Dianyo/poultry-fecal-fl
6. Verifies existing Dianyo/fecal-health and Dianyo/poultry-health datasets

Usage:
    python -m vitpoultry.data_curation.merge_and_upload
"""

import re
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Image, load_dataset

DATA_DIR = Path("deduplicated_data")

LABEL_MAP_4CLASS = {
    "healthy": 0,
    "Healthy": 0,
    "HEALTHY": 0,
    "cocci": 1,
    "Cocci": 1,
    "coccidiosis": 1,
    "Coccidiosis": 1,
    "ncd": 2,
    "NCD": 2,
    "newcastle": 2,
    "Newcastle": 2,
    "salmo": 3,
    "Salmo": 3,
    "salmonella": 3,
    "Salmonella": 3,
}

CLASS_NAMES_4 = ["healthy", "cocci", "ncd", "salmo"]


def extract_farm_id(filepath: Path, source: str) -> str:
    """Extract farm_id from filepath or metadata.

    Farm IDs are used for natural federated learning splits, where each farm
    represents a different client with potentially different data distributions.
    """
    path_str = str(filepath)

    if "zenodo" in source.lower() or "tanzania" in source.lower():
        match = re.search(r"[Ff]arm[_\-]?(\d+)", path_str)
        if match:
            return f"tanzania_farm_{int(match.group(1)):02d}"

        if "lab" in path_str.lower():
            return "tanzania_lab"

    if "roboflow" in source.lower():
        hash_val = hash(filepath.stem) % 10
        return f"roboflow_group_{hash_val:02d}"

    if "fecal-health" in source.lower() or "fecal_health" in source.lower():
        return "lab_confirmed"

    return f"{source}_unknown"


def parse_label_from_path(filepath: Path) -> int:
    """Extract label from filepath (folder name or filename)."""
    path_str = str(filepath).lower()

    for label_name, label_id in LABEL_MAP_4CLASS.items():
        if label_name.lower() in path_str:
            return label_id

    return -1


def build_4class_dataset(data_dir: Path) -> DatasetDict:
    """Build 4-class dataset from deduplicated sources."""
    records = []

    source_dirs = ["zenodo_tanzania", "roboflow", "huggingface"]

    for source_name in source_dirs:
        source_dir = data_dir / source_name
        if not source_dir.exists():
            print(f"  Skipping {source_name} (not found)")
            continue

        print(f"  Processing {source_name}...")
        count = 0

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for img_path in source_dir.rglob(ext):
                label = parse_label_from_path(img_path)
                if label == -1:
                    continue

                farm_id = extract_farm_id(img_path, source_name)
                records.append(
                    {
                        "image": str(img_path.absolute()),
                        "label": label,
                        "farm_id": farm_id,
                        "source": source_name,
                    }
                )
                count += 1

        print(f"    Found {count} valid images")

    if not records:
        print("Error: No images found. Run deduplicate.py first.")
        return None

    print(f"\n4-Class Dataset: {len(records)} total images")
    print(f"Label distribution: {Counter(r['label'] for r in records)}")
    print(f"Unique farms: {len(set(r['farm_id'] for r in records))}")

    farm_ids = sorted(set(r["farm_id"] for r in records))
    test_farm_count = max(1, len(farm_ids) // 5)
    test_farms = set(farm_ids[:test_farm_count])

    print(f"\nSplitting by farm_id:")
    print(f"  Test farms ({len(test_farms)}): {test_farms}")
    print(f"  Train farms ({len(farm_ids) - len(test_farms)})")

    train_records = [r for r in records if r["farm_id"] not in test_farms]
    test_records = [r for r in records if r["farm_id"] in test_farms]

    train_ds = Dataset.from_list(train_records).cast_column("image", Image())
    test_ds = Dataset.from_list(test_records).cast_column("image", Image())

    return DatasetDict({"train": train_ds, "test": test_ds})


def print_stats(dataset: DatasetDict) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 40)
    print("Dataset Statistics")
    print("=" * 40)

    for split_name, split_ds in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  Total images: {len(split_ds)}")

        if "label" in split_ds.features:
            label_counts = Counter(split_ds["label"])
            for i, name in enumerate(CLASS_NAMES_4):
                print(f"    {name}: {label_counts.get(i, 0)}")

        if "farm_id" in split_ds.features:
            farm_counts = Counter(split_ds["farm_id"])
            print(f"  Unique farms: {len(farm_counts)}")


def verify_existing_datasets() -> None:
    """Verify existing HuggingFace datasets match expected sources."""
    print("\n" + "=" * 60)
    print("VERIFYING EXISTING HUGGINGFACE DATASETS")
    print("=" * 60)

    print("\n--- Dianyo/fecal-health (Zenodo LAB data, 4-class) ---")
    try:
        ds = load_dataset("Dianyo/fecal-health")
        for split_name, split_ds in ds.items():
            print(f"  {split_name}: {len(split_ds)} images")
            if "label" in split_ds.features:
                label_counts = Counter(split_ds["label"])
                print(f"  Labels: {dict(label_counts)}")
    except Exception as e:
        print(f"  Error loading: {e}")

    print("\n--- Dianyo/poultry-health (Mendeley Nigeria, binary) ---")
    try:
        ds = load_dataset("Dianyo/poultry-health")
        for split_name, split_ds in ds.items():
            print(f"  {split_name}: {len(split_ds)} images")
            if "label" in split_ds.features:
                label_counts = Counter(split_ds["label"])
                print(f"  Labels: {dict(label_counts)}")
    except Exception as e:
        print(f"  Error loading: {e}")


def main():
    """Main merge and upload routine."""
    print("=" * 60)
    print("MERGE AND UPLOAD DATASETS")
    print("=" * 60)

    verify_existing_datasets()

    print("\n" + "=" * 60)
    print("BUILDING NEW 4-CLASS DATASET")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\nError: {DATA_DIR} not found.")
        print("Run deduplicate.py first to create deduplicated images.")
        return

    print(f"\nScanning {DATA_DIR}...")
    ds_4class = build_4class_dataset(DATA_DIR)

    if ds_4class is None:
        return

    print_stats(ds_4class)

    print("\n" + "=" * 60)
    print("UPLOADING TO HUGGINGFACE")
    print("=" * 60)

    print("\nTo upload, uncomment the following line and ensure you're logged in:")
    print("  huggingface-cli login")
    print("\nUpload command:")
    print('  ds_4class.push_to_hub("Dianyo/poultry-fecal-fl", private=False)')

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Verify the dataset looks correct")
    print("2. Run: huggingface-cli login")
    print("3. Uncomment push_to_hub call and re-run")
    print("4. Dataset will be available at: huggingface.co/datasets/Dianyo/poultry-fecal-fl")


if __name__ == "__main__":
    main()
