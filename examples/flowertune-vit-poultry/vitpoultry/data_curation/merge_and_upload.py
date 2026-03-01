"""DEPRECATED: Use upload_to_hf.py instead.

This script has been superseded by:
- upload_to_hf.py: Build and upload datasets to HuggingFace
- verify_datasets.py: Verify downloaded data

Original description:
Merge deduplicated datasets and upload to HuggingFace.
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


LABEL_MAP_BINARY = {
    "healthy": 0,
    "Healthy": 0,
    "HEALTHY": 0,
    "unhealthy": 1,
    "Unhealthy": 1,
    "UNHEALTHY": 1,
    "sick": 1,
    "Sick": 1,
    "diseased": 1,
    "Diseased": 1,
}

CLASS_NAMES_BINARY = ["healthy", "unhealthy"]


def build_binary_dataset_from_mendeley(mendeley_dir: Path) -> DatasetDict:
    """Build binary dataset from full Mendeley Nigeria download.
    
    This creates a complete binary (healthy/unhealthy) dataset from
    the raw Mendeley download, which may contain more data than
    the existing Dianyo/poultry-health.
    """
    records = []
    
    if not mendeley_dir.exists():
        print(f"  Mendeley directory not found: {mendeley_dir}")
        return None
    
    print(f"  Scanning {mendeley_dir}...")
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in mendeley_dir.rglob(ext):
            path_str = str(img_path).lower()
            
            # Determine label from path
            if "healthy" in path_str and "unhealthy" not in path_str:
                label = 0
            elif any(x in path_str for x in ["unhealthy", "sick", "diseased"]):
                label = 1
            else:
                # Try to infer from parent folder
                parent = img_path.parent.name.lower()
                if "healthy" in parent and "un" not in parent:
                    label = 0
                elif any(x in parent for x in ["unhealthy", "sick", "diseased"]):
                    label = 1
                else:
                    continue
            
            records.append({
                "image": str(img_path.absolute()),
                "label": label,
            })
    
    if not records:
        print("  No valid images found in Mendeley directory")
        return None
    
    print(f"  Found {len(records)} images")
    print(f"  Label distribution: {Counter(r['label'] for r in records)}")
    
    # 80/20 train/test split
    import random
    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * 0.8)
    
    train_records = records[:split_idx]
    test_records = records[split_idx:]
    
    train_ds = Dataset.from_list(train_records).cast_column("image", Image())
    test_ds = Dataset.from_list(test_records).cast_column("image", Image())
    
    return DatasetDict({"train": train_ds, "test": test_ds})


def compare_binary_datasets(mendeley_dir: Path) -> None:
    """Compare Mendeley download with existing HuggingFace binary dataset."""
    print("\n" + "=" * 60)
    print("COMPARING BINARY DATASETS")
    print("=" * 60)
    
    # Load existing HF dataset
    print("\n1. Existing Dianyo/poultry-health on HuggingFace:")
    try:
        hf_ds = load_dataset("Dianyo/poultry-health")
        hf_total = sum(len(split) for split in hf_ds.values())
        print(f"   Total images: {hf_total}")
        for split_name, split_ds in hf_ds.items():
            print(f"   {split_name}: {len(split_ds)}")
    except Exception as e:
        print(f"   Error loading: {e}")
        hf_total = 0
    
    # Count Mendeley images
    print(f"\n2. Full Mendeley download at {mendeley_dir}:")
    if mendeley_dir.exists():
        mendeley_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            mendeley_images.extend(mendeley_dir.rglob(ext))
        mendeley_total = len(mendeley_images)
        print(f"   Total images: {mendeley_total}")
        
        if mendeley_total > hf_total:
            print(f"\n   ⚠️  Mendeley has {mendeley_total - hf_total} MORE images than HuggingFace!")
            print("   Consider rebuilding Dianyo/poultry-health with full data.")
        elif mendeley_total < hf_total:
            print(f"\n   HuggingFace has {hf_total - mendeley_total} more images (may include augmented data)")
        else:
            print("\n   ✓ Image counts match")
    else:
        print("   Directory not found - run download_datasets.py first")


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
    
    # Compare binary datasets (Mendeley vs HuggingFace)
    raw_data_dir = Path("raw_data")
    mendeley_dir = raw_data_dir / "mendeley_nigeria"
    compare_binary_datasets(mendeley_dir)

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
    
    # Also build binary dataset from full Mendeley if available
    print("\n" + "=" * 60)
    print("BUILDING BINARY DATASET FROM MENDELEY")
    print("=" * 60)
    
    if mendeley_dir.exists():
        ds_binary = build_binary_dataset_from_mendeley(mendeley_dir)
        if ds_binary:
            print("\nBinary dataset statistics:")
            for split_name, split_ds in ds_binary.items():
                label_counts = Counter(split_ds["label"])
                print(f"  {split_name}: {len(split_ds)} images")
                print(f"    healthy: {label_counts.get(0, 0)}, unhealthy: {label_counts.get(1, 0)}")
    else:
        print(f"Mendeley directory not found: {mendeley_dir}")
        print("Run download_datasets.py first if you want to build the full binary dataset.")

    print("\n" + "=" * 60)
    print("UPLOADING TO HUGGINGFACE")
    print("=" * 60)

    print("\nTo upload, ensure you're logged in first:")
    print("  huggingface-cli login")
    print("\n4-class dataset upload:")
    print('  ds_4class.push_to_hub("Dianyo/poultry-fecal-fl", private=False)')
    print("\nBinary dataset upload (if Mendeley has more data):")
    print('  ds_binary.push_to_hub("Dianyo/poultry-health", private=False)')

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Verify the datasets look correct")
    print("2. Run: huggingface-cli login")
    print("3. Uncomment push_to_hub calls in this script and re-run")
    print("4. Datasets will be available at:")
    print("   - huggingface.co/datasets/Dianyo/poultry-fecal-fl (4-class)")
    print("   - huggingface.co/datasets/Dianyo/poultry-health (binary, if updated)")


if __name__ == "__main__":
    main()
