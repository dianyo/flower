"""Convert COCO object detection format to classification folder structure.

This script reorganizes COCO-format datasets into class-based folders
so they work with the standard classification pipeline.

Run this AFTER download_datasets.py and BEFORE deduplicate.py

Usage:
    python -m vitpoultry.data_curation.convert_coco_to_classification
"""

import json
import shutil
from collections import Counter
from pathlib import Path

RAW_DATA_DIR = Path("raw_data")

COCO_CATEGORY_TO_CLASS = {
    "healthy": "healthy",
    "coccidiosis-like": "cocci",
    "coccidiosis": "cocci",
    "ncd-like": "ncd",
    "ncd": "ncd",
    "newcastle": "ncd",
    "salmonella-like": "salmo",
    "salmonella": "salmo",
}


def convert_coco_dataset(coco_dir: Path, output_dir: Path) -> int:
    """Convert a COCO dataset to classification folder structure.
    
    Args:
        coco_dir: Directory containing train/, valid/, test/ with COCO annotations
        output_dir: Output directory for class-organized images
        
    Returns:
        Number of images converted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_converted = 0
    label_counts = Counter()
    
    for split in ["train", "valid", "test"]:
        split_dir = coco_dir / split
        ann_file = split_dir / "_annotations.coco.json"
        
        if not ann_file.exists():
            continue
        
        print(f"  Processing {split}/...")
        
        with open(ann_file) as f:
            data = json.load(f)
        
        # Build COCO category_id -> our class name mapping
        coco_id_to_class = {}
        for cat in data.get("categories", []):
            cat_name = cat["name"].lower()
            for coco_name, our_class in COCO_CATEGORY_TO_CLASS.items():
                if coco_name in cat_name:
                    coco_id_to_class[cat["id"]] = our_class
                    break
        
        print(f"    Categories found: {coco_id_to_class}")
        
        # Build image_id -> filename mapping
        image_id_to_info = {}
        for img in data.get("images", []):
            image_id_to_info[img["id"]] = {
                "file_name": img["file_name"],
                "path": split_dir / img["file_name"],
            }
        
        # Get the primary label for each image (first annotation)
        image_labels = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            
            if img_id not in image_labels and cat_id in coco_id_to_class:
                image_labels[img_id] = coco_id_to_class[cat_id]
        
        # Copy images to class folders
        for img_id, class_name in image_labels.items():
            if img_id not in image_id_to_info:
                continue
            
            src_path = image_id_to_info[img_id]["path"]
            if not src_path.exists():
                continue
            
            # Create output path: output_dir/class_name/filename
            dst_dir = output_dir / class_name
            dst_dir.mkdir(exist_ok=True)
            dst_path = dst_dir / src_path.name
            
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                total_converted += 1
                label_counts[class_name] += 1
    
    print(f"    Converted: {dict(label_counts)}")
    return total_converted


def main():
    print("=" * 60)
    print("CONVERT COCO TO CLASSIFICATION FORMAT")
    print("=" * 60)
    
    roboflow_dir = RAW_DATA_DIR / "roboflow"
    
    if not roboflow_dir.exists():
        print(f"\nError: {roboflow_dir} not found")
        print("Run download_datasets.py first")
        return
    
    total = 0
    
    # Find all COCO datasets in roboflow folder
    for dataset_dir in roboflow_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        # Check if it's a COCO format (has _annotations.coco.json)
        has_coco = any(
            (dataset_dir / split / "_annotations.coco.json").exists()
            for split in ["train", "valid", "test"]
        )
        
        if not has_coco:
            print(f"\n{dataset_dir.name}: Already in folder format (skipping)")
            continue
        
        print(f"\n{dataset_dir.name}: COCO format detected")
        
        # Convert to classification format
        output_dir = roboflow_dir / f"{dataset_dir.name}_classification"
        converted = convert_coco_dataset(dataset_dir, output_dir)
        total += converted
        print(f"  Output: {output_dir}")
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total images converted: {total}")
    print("\nNext step: python -m vitpoultry.data_curation.deduplicate")


if __name__ == "__main__":
    main()
