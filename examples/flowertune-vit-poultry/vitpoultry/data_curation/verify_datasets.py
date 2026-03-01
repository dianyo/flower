"""Quick verification of downloaded datasets.

Compares local downloads with HuggingFace datasets to check for missing data.

Usage:
    python -m vitpoultry.data_curation.verify_datasets
"""

from collections import Counter
from pathlib import Path

from datasets import load_dataset

RAW_DATA_DIR = Path("raw_data")


def count_images(directory: Path) -> int:
    """Count image files in a directory recursively."""
    if not directory.exists():
        return 0
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        count += len(list(directory.rglob(ext)))
    return count


def show_folder_structure(directory: Path, max_depth: int = 2) -> None:
    """Show folder structure with image counts."""
    if not directory.exists():
        print(f"    Directory not found: {directory}")
        return
    
    for item in sorted(directory.iterdir()):
        if item.is_dir():
            img_count = count_images(item)
            print(f"    {item.name}/: {img_count} images")


def verify_huggingface_datasets() -> dict:
    """Verify and count HuggingFace datasets."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE DATASETS")
    print("=" * 60)
    
    hf_counts = {}
    
    datasets_to_check = [
        ("Dianyo/fecal-health", "4-class (Zenodo LAB)"),
        ("Dianyo/poultry-health", "binary (Mendeley Nigeria)"),
    ]
    
    for dataset_name, description in datasets_to_check:
        print(f"\n{dataset_name} - {description}:")
        try:
            ds = load_dataset(dataset_name)
            total = 0
            for split_name, split_ds in ds.items():
                print(f"  {split_name}: {len(split_ds)} images")
                total += len(split_ds)
                if "label" in split_ds.features:
                    label_counts = Counter(split_ds["label"])
                    print(f"    Labels: {dict(label_counts)}")
            hf_counts[dataset_name] = total
        except Exception as e:
            print(f"  Error: {e}")
            hf_counts[dataset_name] = 0
    
    return hf_counts


def verify_local_downloads() -> dict:
    """Verify and count local downloaded datasets."""
    print("\n" + "=" * 60)
    print("LOCAL DOWNLOADS")
    print("=" * 60)
    
    local_counts = {}
    
    sources = [
        ("huggingface", "HuggingFace cache"),
        ("zenodo_tanzania", "Zenodo Tanzania (farm data)"),
        ("roboflow", "Roboflow datasets"),
        ("mendeley_nigeria", "Mendeley Nigeria (binary)"),
    ]
    
    for folder_name, description in sources:
        folder_path = RAW_DATA_DIR / folder_name
        count = count_images(folder_path)
        local_counts[folder_name] = count
        
        print(f"\n{folder_name}/ - {description}:")
        if folder_path.exists():
            print(f"  Total images: {count}")
            show_folder_structure(folder_path)
        else:
            print(f"  Not downloaded yet")
    
    return local_counts


def compare_datasets(hf_counts: dict, local_counts: dict) -> None:
    """Compare HuggingFace with local downloads."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    # Compare Mendeley
    hf_binary = hf_counts.get("Dianyo/poultry-health", 0)
    local_mendeley = local_counts.get("mendeley_nigeria", 0)
    
    print(f"\nBinary dataset (Mendeley):")
    print(f"  HuggingFace (poultry-health): {hf_binary} images")
    print(f"  Local Mendeley download:      {local_mendeley} images")
    
    if local_mendeley > hf_binary:
        diff = local_mendeley - hf_binary
        print(f"  ⚠️  Local has {diff} MORE images - consider updating HuggingFace!")
    elif local_mendeley < hf_binary and local_mendeley > 0:
        print(f"  ✓ HuggingFace has more (may include augmented data)")
    elif local_mendeley == 0:
        print(f"  ℹ️  Mendeley not downloaded locally yet")
    else:
        print(f"  ✓ Counts match")
    
    # Show Roboflow + Zenodo for 4-class
    local_zenodo = local_counts.get("zenodo_tanzania", 0)
    local_roboflow = local_counts.get("roboflow", 0)
    
    print(f"\n4-class dataset sources:")
    print(f"  Zenodo Tanzania (farm): {local_zenodo} images")
    print(f"  Roboflow:               {local_roboflow} images")
    print(f"  Total for new dataset:  {local_zenodo + local_roboflow} images")


def main():
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DATA_DIR.absolute()}")
    
    hf_counts = verify_huggingface_datasets()
    local_counts = verify_local_downloads()
    compare_datasets(hf_counts, local_counts)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. If local Mendeley has more images → update Dianyo/poultry-health")
    print("2. Run: python -m vitpoultry.data_curation.deduplicate")
    print("3. Run: python -m vitpoultry.data_curation.merge_and_upload")


if __name__ == "__main__":
    main()
