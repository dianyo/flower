"""Download and organize datasets from multiple sources.

This script downloads poultry fecal health data from:
1. HuggingFace: Dianyo/fecal-health (Zenodo LAB data, 4-class PCR-confirmed)
2. Zenodo: AI4D Tanzania dataset (record 5801834) - contains FARM data with farm_id
3. Roboflow: Two fecal disease datasets via API
4. Mendeley: Nigeria binary dataset

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

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_KEY")

ROBOFLOW_DATASETS = [
    {
        "workspace": "fecal",
        "project": "disease-images-fecal",
        "version": 2,
        "description": "Fecal Disease Images (main dataset)",
        "format": "folder",  # Classification project
        "project_type": "classification",
    },
    {
        "workspace": "thesis-pr4oh",
        "project": "fecal-lbh0j",
        "version": 1,
        "description": "Fecal LBH0J (thesis dataset)",
        "format": "coco",  # Object detection - download as COCO, extract images
        "project_type": "object-detection",
    },
]


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
    
    # Check if already downloaded (look for image files)
    existing_images = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.jpeg")) + list(output_dir.rglob("*.png"))
    if existing_images:
        print(f"\nSkipping Zenodo {record_id} (found {len(existing_images)} images in {output_dir})")
        return
    
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
    save_path = output_dir / dataset_name.replace("/", "_")
    
    # Check if already downloaded locally
    if save_path.exists() and (save_path / "dataset_info.json").exists():
        print(f"\nSkipping {dataset_name} (already exists at {save_path})")
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(save_path))
            for split_name, split_ds in ds.items():
                print(f"  {split_name}: {len(split_ds)} samples")
            return ds
        except Exception:
            pass  # Fall through to re-download
    
    print(f"\nDownloading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name)
    ds.save_to_disk(str(save_path))
    print(f"  Saved to {save_path}")

    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} samples")

    return ds


def download_mendeley_nigeria(output_dir: Path) -> None:
    """Download Mendeley Nigeria dataset (binary classification).
    
    DOI: 10.17632/8pnbzpt2k9.1
    This is the source for Dianyo/poultry-health (which may be partial).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded (manually or previously)
    # Look for image files in the directory
    existing_images = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.jpeg")) + list(output_dir.rglob("*.png"))
    if existing_images:
        print(f"\nSkipping Mendeley download (found {len(existing_images)} images in {output_dir})")
        
        # Show folder structure
        subdirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"  Folders: {', '.join(subdirs[:5])}{'...' if len(subdirs) > 5 else ''}")
        return
    
    # Mendeley Data API endpoint for this dataset
    # The dataset has multiple versions, we want v1
    api_url = "https://data.mendeley.com/public-api/datasets/8pnbzpt2k9"
    
    print("\nFetching Mendeley dataset metadata...")
    try:
        resp = requests.get(api_url)
        if resp.status_code != 200:
            print(f"  API returned {resp.status_code}, falling back to manual instructions")
            print_mendeley_manual_instructions(output_dir)
            return
            
        data = resp.json()
        
        # Get the files from the dataset
        files = data.get("files", [])
        if not files:
            print("  No files found via API, trying direct download...")
            # Try direct download URL pattern
            direct_url = "https://data.mendeley.com/public-files/datasets/8pnbzpt2k9/files/poultry_data.zip/file_downloaded"
            filepath = output_dir / "poultry_data.zip"
            if not filepath.exists():
                try:
                    download_file(direct_url, filepath, "poultry_data.zip")
                    print(f"  Extracting to {output_dir}...")
                    with zipfile.ZipFile(filepath, "r") as z:
                        z.extractall(output_dir)
                except Exception as e:
                    print(f"  Direct download failed: {e}")
                    print_mendeley_manual_instructions(output_dir)
            return
            
        for file_info in files:
            filename = file_info.get("filename", "unknown")
            file_id = file_info.get("id")
            
            if not file_id:
                continue
                
            filepath = output_dir / filename
            if filepath.exists():
                print(f"  Skipping {filename} (already exists)")
                continue
            
            # Construct download URL
            download_url = f"https://data.mendeley.com/public-files/datasets/8pnbzpt2k9/files/{file_id}/file_downloaded"
            print(f"  Downloading {filename}...")
            download_file(download_url, filepath, filename)
            
            if filename.endswith(".zip"):
                print(f"  Extracting {filename}...")
                with zipfile.ZipFile(filepath, "r") as z:
                    z.extractall(output_dir / filename.replace(".zip", ""))
                    
    except Exception as e:
        print(f"  Error accessing Mendeley API: {e}")
        print_mendeley_manual_instructions(output_dir)


def download_roboflow(output_dir: Path, api_key: str = None) -> None:
    """Download datasets from Roboflow using their API.
    
    Downloads both:
    1. fecal/disease-images-fecal (main dataset)
    2. thesis-pr4oh/fecal-lbh0j (thesis dataset)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    api_key = api_key or ROBOFLOW_API_KEY
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  Roboflow not installed. Run: pip install roboflow")
        print_roboflow_manual_instructions(output_dir)
        return
    
    rf = Roboflow(api_key=api_key)
    
    for dataset_info in ROBOFLOW_DATASETS:
        workspace = dataset_info["workspace"]
        project_name = dataset_info["project"]
        version_num = dataset_info["version"]
        description = dataset_info["description"]
        dl_format = dataset_info.get("format", "folder")
        project_type = dataset_info.get("project_type", "classification")
        
        dataset_dir = output_dir / f"{workspace}_{project_name}"
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  Skipping {description} (already exists at {dataset_dir})")
            continue
        
        print(f"\n  Downloading: {description}")
        print(f"    Workspace: {workspace}")
        print(f"    Project: {project_name}")
        print(f"    Version: {version_num}")
        print(f"    Format: {dl_format} ({project_type})")
        
        try:
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_num)
            
            dataset = version.download(dl_format, location=str(dataset_dir))
            print(f"    ✓ Downloaded to: {dataset_dir}")
            
            # For object detection datasets, reorganize images by class for classification
            if project_type == "object-detection":
                print(f"    Note: Object detection dataset - images can be used for classification")
                print(f"    Images are in: {dataset_dir}/train/, {dataset_dir}/valid/, {dataset_dir}/test/")
            
        except Exception as e:
            print(f"    ✗ Error downloading {project_name}: {e}")
            continue


def print_roboflow_manual_instructions(output_dir: Path) -> None:
    """Print manual download instructions for Roboflow datasets."""
    print("\n  MANUAL DOWNLOAD REQUIRED for Roboflow:")
    print("\n  Dataset 1: disease-images-fecal")
    print("    URL: https://universe.roboflow.com/fecal/disease-images-fecal")
    print("    → Download as 'Folder' format")
    print(f"    → Extract to: {output_dir / 'fecal_disease-images-fecal'}")
    
    print("\n  Dataset 2: fecal-lbh0j")
    print("    URL: https://universe.roboflow.com/thesis-pr4oh/fecal-lbh0j")
    print("    → Download as 'Folder' format")
    print(f"    → Extract to: {output_dir / 'thesis-pr4oh_fecal-lbh0j'}")


def print_mendeley_manual_instructions(output_dir: Path) -> None:
    """Print manual download instructions for Mendeley dataset."""
    print("\n  MANUAL DOWNLOAD REQUIRED:")
    print("  URL: https://data.mendeley.com/datasets/8pnbzpt2k9/1")
    print("  1. Click 'Download' button")
    print("  2. Accept terms if prompted")
    print(f"  3. Extract to: {output_dir.absolute()}")


def main():
    """Main download routine."""
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Data directory: {DATA_DIR.absolute()}")

    print("\n" + "=" * 60)
    print("STEP 1: HuggingFace Datasets (Existing)")
    print("=" * 60)
    hf_dir = DATA_DIR / "huggingface"
    hf_dir.mkdir(exist_ok=True)

    # Download 4-class dataset (Zenodo LAB data)
    download_huggingface("Dianyo/fecal-health", hf_dir)
    
    # Download binary dataset for comparison with full Mendeley
    print("\nDownloading Dianyo/poultry-health (binary) for verification...")
    download_huggingface("Dianyo/poultry-health", hf_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Zenodo AI4D Tanzania Dataset (FARM data)")
    print("=" * 60)
    print("Record 5801834 contains both lab and farm data with farm_id metadata")

    zenodo_dir = DATA_DIR / "zenodo_tanzania"
    download_zenodo("5801834", zenodo_dir)

    print("\n" + "=" * 60)
    print("STEP 3: Roboflow Datasets (2 datasets)")
    print("=" * 60)
    print("Downloading fecal disease images from Roboflow...")
    
    roboflow_dir = DATA_DIR / "roboflow"
    download_roboflow(roboflow_dir)

    print("\n" + "=" * 60)
    print("STEP 4: Mendeley Nigeria Dataset (Full Binary)")
    print("=" * 60)
    print("DOI: 10.17632/8pnbzpt2k9.1")
    print("This is the FULL source for Dianyo/poultry-health")
    
    mendeley_dir = DATA_DIR / "mendeley_nigeria"
    download_mendeley_nigeria(mendeley_dir)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print("\nContents:")
    for item in sorted(DATA_DIR.iterdir()):
        if item.is_dir():
            file_count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/: {file_count} files")
        else:
            print(f"  {item.name}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print("Compare Mendeley download with Dianyo/poultry-health:")
    print(f"  - Full Mendeley: {DATA_DIR / 'mendeley_nigeria'}")
    print(f"  - HF poultry-health: {DATA_DIR / 'huggingface' / 'Dianyo_poultry-health'}")
    print("If Mendeley has more images, consider updating poultry-health on HuggingFace.")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Complete manual download for Roboflow (if needed)")
    print("2. Run: python -m vitpoultry.data_curation.deduplicate")
    print("3. Run: python -m vitpoultry.data_curation.merge_and_upload")


if __name__ == "__main__":
    main()
