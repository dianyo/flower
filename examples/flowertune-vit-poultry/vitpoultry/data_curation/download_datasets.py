"""Download and organize datasets from multiple sources.

This script downloads poultry fecal health data from:
1. HuggingFace: Dianyo/fecal-health (Zenodo LAB data, 4-class PCR-confirmed)
2. Zenodo: AI4D Tanzania dataset (record 5801834) - contains FARM data with farm_id
3. Manual instructions for Roboflow and Mendeley datasets

Usage:
    python -m vitpoultry.data_curation.download_datasets
"""

import os
import requests
import zipfile
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path("raw_data")


def download_file(url: str, filepath: Path, desc: str = None) -> None:
    """Download a file with progress bar."""
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=desc or filepath.name
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_zenodo(record_id: str, output_dir: Path) -> None:
    """Download dataset from Zenodo API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(url).json()

    print(f"\nZenodo record {record_id}: {resp.get('metadata', {}).get('title', 'Unknown')}")

    for file_info in resp.get("files", []):
        file_url = file_info["links"]["self"]
        filename = file_info["key"]
        filepath = output_dir / filename

        if filepath.exists():
            print(f"  Skipping {filename} (already exists)")
            continue

        print(f"  Downloading {filename}...")
        download_file(file_url, filepath, filename)

        if filename.endswith(".zip"):
            print(f"  Extracting {filename}...")
            extract_dir = output_dir / filename.replace(".zip", "")
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(extract_dir)


def download_huggingface(dataset_name: str, output_dir: Path):
    """Download existing HuggingFace dataset."""
    print(f"\nDownloading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name)

    save_path = output_dir / dataset_name.replace("/", "_")
    ds.save_to_disk(str(save_path))
    print(f"  Saved to {save_path}")

    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} samples")

    return ds


def print_manual_instructions() -> None:
    """Print instructions for manual downloads."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)

    print("\n1. ROBOFLOW DATASET (Fecal Disease Images)")
    print("-" * 40)
    print("   URL: https://universe.roboflow.com/fecal/disease-images-fecal")
    print("   Steps:")
    print("   a) Click 'Download Dataset'")
    print("   b) Choose 'Folder' format (not YOLO/COCO)")
    print(f"   c) Extract to: {DATA_DIR.absolute() / 'roboflow'}")

    print("\n2. MENDELEY NIGERIA DATASET (Binary Classification)")
    print("-" * 40)
    print("   URL: https://data.mendeley.com/datasets/8pnbzpt2k9/1")
    print("   Note: This dataset is already uploaded as Dianyo/poultry-health")
    print("   Download only if you need to verify the source data.")
    print(f"   Extract to: {DATA_DIR.absolute() / 'mendeley_nigeria'}")


def main():
    """Main download routine."""
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Data directory: {DATA_DIR.absolute()}")

    print("\n" + "=" * 60)
    print("STEP 1: HuggingFace Datasets (Existing)")
    print("=" * 60)
    hf_dir = DATA_DIR / "huggingface"
    hf_dir.mkdir(exist_ok=True)

    download_huggingface("Dianyo/fecal-health", hf_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Zenodo AI4D Tanzania Dataset (FARM data)")
    print("=" * 60)
    print("Record 5801834 contains both lab and farm data with farm_id metadata")

    zenodo_dir = DATA_DIR / "zenodo_tanzania"
    download_zenodo("5801834", zenodo_dir)

    print_manual_instructions()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print("\nContents:")
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            file_count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/: {file_count} files")
        else:
            print(f"  {item.name}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Complete manual downloads (Roboflow, optionally Mendeley)")
    print("2. Run: python -m vitpoultry.data_curation.deduplicate")
    print("3. Run: python -m vitpoultry.data_curation.merge_and_upload")


if __name__ == "__main__":
    main()
