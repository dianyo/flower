"""Deduplicate images using perceptual hashing.

This script scans downloaded images and removes duplicates using average hash (aHash)
and perceptual hash (pHash) from the imagehash library.

Usage:
    python -m vitpoultry.data_curation.deduplicate
"""

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("deduplicated_data")
LOGS_DIR = Path("logs")
HASH_SIZE = 16
HASH_THRESHOLD = 5

# Only deduplicate 4-class sources (binary Mendeley data is used raw)
SOURCES_TO_DEDUPLICATE = ["zenodo_farm", "zenodo_lab", "roboflow"]


def get_image_metadata(image_path: Path) -> dict:
    """Get metadata for an image file."""
    try:
        file_size = image_path.stat().st_size
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
        return {
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "mode": mode,
        }
    except Exception:
        return {"file_size_bytes": 0, "width": 0, "height": 0, "mode": "unknown"}


def get_source_from_path(image_path: Path) -> str:
    """Extract source dataset from image path."""
    try:
        rel_path = image_path.relative_to(RAW_DATA_DIR)
        return rel_path.parts[0] if rel_path.parts else "unknown"
    except ValueError:
        return "unknown"


def get_label_from_path(image_path: Path) -> str:
    """Extract class label from image path."""
    path_str = str(image_path).lower()
    for label in ["healthy", "cocci", "coccidiosis", "ncd", "newcastle", "salmo", "salmonella", "unhealthy", "sick"]:
        if label in path_str:
            return label
    return "unknown"


def compute_hashes(image_path: Path) -> tuple:
    """Compute perceptual hashes for an image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            ahash = imagehash.average_hash(img, hash_size=HASH_SIZE)
            phash = imagehash.phash(img, hash_size=HASH_SIZE)
            return ahash, phash
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def find_duplicates(image_paths: list[Path]) -> tuple[dict[Path, list[Path]], dict[Path, tuple]]:
    """Find duplicate images using perceptual hashing.

    Returns:
        Tuple of (duplicates dict, hashes dict)
        - duplicates: Dictionary mapping canonical image to list of duplicates.
        - hashes: Dictionary mapping image path to (ahash, phash) tuple.
    """
    hash_to_path: dict[str, Path] = {}
    duplicates: dict[Path, list[Path]] = defaultdict(list)
    hashes_computed = {}

    for img_path in tqdm(image_paths, desc="Computing hashes"):
        ahash, phash = compute_hashes(img_path)
        if ahash is None:
            continue

        hashes_computed[img_path] = (ahash, phash)
        combined_key = f"{ahash}_{phash}"

        if combined_key in hash_to_path:
            canonical = hash_to_path[combined_key]
            duplicates[canonical].append(img_path)
        else:
            hash_to_path[combined_key] = img_path

    print(f"\nExact hash matches: {sum(len(v) for v in duplicates.values())} duplicates")

    near_dupes = find_near_duplicates(hashes_computed, duplicates, hash_to_path)
    print(f"Near-duplicate matches (threshold={HASH_THRESHOLD}): {near_dupes}")

    return duplicates, hashes_computed


def find_near_duplicates(
    hashes: dict[Path, tuple],
    duplicates: dict[Path, list[Path]],
    hash_to_path: dict[str, Path],
) -> int:
    """Find near-duplicate images based on hash distance threshold."""
    near_dupe_count = 0
    processed = set()

    canonical_paths = list(hash_to_path.values())
    for i, path1 in enumerate(tqdm(canonical_paths, desc="Finding near-duplicates")):
        if path1 in processed:
            continue

        ahash1, phash1 = hashes[path1]

        for path2 in canonical_paths[i + 1 :]:
            if path2 in processed:
                continue

            ahash2, phash2 = hashes[path2]
            a_dist = ahash1 - ahash2
            p_dist = phash1 - phash2

            if a_dist <= HASH_THRESHOLD and p_dist <= HASH_THRESHOLD:
                duplicates[path1].append(path2)
                processed.add(path2)
                near_dupe_count += 1

    return near_dupe_count


def collect_images(data_dir: Path, sources: list[str] | None = None) -> list[Path]:
    """Collect image files from subdirectories.
    
    Args:
        data_dir: Root directory containing source folders
        sources: List of source folder names to include. If None, include all.
    """
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    
    if sources:
        # Only collect from specified sources
        for source in sources:
            source_dir = data_dir / source
            if source_dir.exists():
                for ext in extensions:
                    images.extend(source_dir.rglob(ext))
    else:
        # Collect from all subdirectories
        for ext in extensions:
            images.extend(data_dir.rglob(ext))
    
    return images


def copy_unique_images(
    image_paths: list[Path],
    duplicates: dict[Path, list[Path]],
    output_dir: Path,
) -> int:
    """Copy unique (non-duplicate) images to output directory."""
    all_duplicates = set()
    for dupe_list in duplicates.values():
        all_duplicates.update(dupe_list)

    unique_count = 0
    for img_path in tqdm(image_paths, desc="Copying unique images"):
        if img_path in all_duplicates:
            continue

        rel_path = img_path.relative_to(RAW_DATA_DIR)
        dest_path = output_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dest_path)
        unique_count += 1

    return unique_count


def generate_deduplication_report(
    duplicates: dict[Path, list[Path]],
    hashes: dict[Path, tuple],
    total_images: int,
    unique_count: int,
) -> None:
    """Generate comprehensive deduplication report in JSON and text formats."""
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build detailed duplicate groups
    duplicate_groups = []
    cross_source_dupes = 0
    cross_label_dupes = 0
    
    for canonical, dupes in duplicates.items():
        if not dupes:
            continue
        
        canonical_source = get_source_from_path(canonical)
        canonical_label = get_label_from_path(canonical)
        canonical_meta = get_image_metadata(canonical)
        canonical_ahash, canonical_phash = hashes.get(canonical, (None, None))
        
        group = {
            "kept": {
                "path": str(canonical.relative_to(RAW_DATA_DIR)),
                "source": canonical_source,
                "label": canonical_label,
                "ahash": str(canonical_ahash) if canonical_ahash else None,
                "phash": str(canonical_phash) if canonical_phash else None,
                **canonical_meta,
            },
            "removed": [],
        }
        
        sources_in_group = {canonical_source}
        labels_in_group = {canonical_label}
        
        for dupe in dupes:
            dupe_source = get_source_from_path(dupe)
            dupe_label = get_label_from_path(dupe)
            dupe_meta = get_image_metadata(dupe)
            dupe_ahash, dupe_phash = hashes.get(dupe, (None, None))
            
            # Calculate hash distances
            a_dist = canonical_ahash - dupe_ahash if canonical_ahash and dupe_ahash else -1
            p_dist = canonical_phash - dupe_phash if canonical_phash and dupe_phash else -1
            
            group["removed"].append({
                "path": str(dupe.relative_to(RAW_DATA_DIR)),
                "source": dupe_source,
                "label": dupe_label,
                "ahash": str(dupe_ahash) if dupe_ahash else None,
                "phash": str(dupe_phash) if dupe_phash else None,
                "ahash_distance": a_dist,
                "phash_distance": p_dist,
                **dupe_meta,
            })
            
            sources_in_group.add(dupe_source)
            labels_in_group.add(dupe_label)
        
        group["num_duplicates"] = len(dupes)
        group["cross_source"] = len(sources_in_group) > 1
        group["cross_label"] = len(labels_in_group) > 1
        group["sources_involved"] = sorted(sources_in_group)
        group["labels_involved"] = sorted(labels_in_group)
        
        if group["cross_source"]:
            cross_source_dupes += len(dupes)
        if group["cross_label"]:
            cross_label_dupes += len(dupes)
        
        duplicate_groups.append(group)
    
    # Sort by number of duplicates (descending)
    duplicate_groups.sort(key=lambda x: x["num_duplicates"], reverse=True)
    
    # Calculate statistics by source
    source_stats = defaultdict(lambda: {"total": 0, "kept": 0, "removed": 0})
    for group in duplicate_groups:
        source_stats[group["kept"]["source"]]["kept"] += 1
        for dupe in group["removed"]:
            source_stats[dupe["source"]]["removed"] += 1
    
    # Build summary
    total_removed = sum(len(g["removed"]) for g in duplicate_groups)
    summary = {
        "timestamp": timestamp,
        "settings": {
            "hash_size": HASH_SIZE,
            "hash_threshold": HASH_THRESHOLD,
            "raw_data_dir": str(RAW_DATA_DIR.absolute()),
            "output_dir": str(OUTPUT_DIR.absolute()),
        },
        "totals": {
            "images_scanned": total_images,
            "unique_images": unique_count,
            "duplicates_removed": total_removed,
            "duplicate_groups": len(duplicate_groups),
            "deduplication_rate": f"{(total_removed / total_images * 100):.2f}%",
        },
        "cross_dataset_analysis": {
            "cross_source_duplicates": cross_source_dupes,
            "cross_label_duplicates": cross_label_dupes,
        },
        "by_source": dict(source_stats),
    }
    
    # Write JSON report (machine-readable)
    json_report = {
        "summary": summary,
        "duplicate_groups": duplicate_groups,
    }
    json_path = LOGS_DIR / f"deduplication_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"\nJSON report saved to: {json_path}")
    
    # Write text report (human-readable)
    txt_path = LOGS_DIR / f"deduplication_report_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DEDUPLICATION REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SETTINGS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Hash size: {HASH_SIZE}\n")
        f.write(f"Hash threshold: {HASH_THRESHOLD}\n")
        f.write(f"Raw data dir: {RAW_DATA_DIR.absolute()}\n")
        f.write(f"Output dir: {OUTPUT_DIR.absolute()}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Images scanned: {total_images}\n")
        f.write(f"Unique images: {unique_count}\n")
        f.write(f"Duplicates removed: {total_removed}\n")
        f.write(f"Duplicate groups: {len(duplicate_groups)}\n")
        f.write(f"Deduplication rate: {(total_removed / total_images * 100):.2f}%\n\n")
        
        f.write("CROSS-DATASET ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cross-source duplicates: {cross_source_dupes}\n")
        f.write(f"Cross-label duplicates (potential labeling issues): {cross_label_dupes}\n\n")
        
        f.write("BY SOURCE\n")
        f.write("-" * 40 + "\n")
        for source, stats in sorted(source_stats.items()):
            f.write(f"  {source}: kept={stats['kept']}, removed={stats['removed']}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("DUPLICATE GROUPS (sorted by size)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, group in enumerate(duplicate_groups[:100], 1):  # Top 100 groups
            f.write(f"--- Group {i} ({group['num_duplicates']} duplicates) ---\n")
            if group["cross_source"]:
                f.write(f"  ⚠️  CROSS-SOURCE: {group['sources_involved']}\n")
            if group["cross_label"]:
                f.write(f"  ⚠️  CROSS-LABEL: {group['labels_involved']}\n")
            
            kept = group["kept"]
            f.write(f"  KEPT: {kept['path']}\n")
            f.write(f"    source={kept['source']}, label={kept['label']}\n")
            f.write(f"    size={kept['width']}x{kept['height']}, {kept['file_size_bytes']/1024:.1f}KB\n")
            f.write(f"    ahash={kept['ahash'][:16]}...\n")
            
            for j, removed in enumerate(group["removed"][:5], 1):  # First 5 duplicates
                f.write(f"  REMOVED #{j}: {removed['path']}\n")
                f.write(f"    source={removed['source']}, label={removed['label']}\n")
                f.write(f"    size={removed['width']}x{removed['height']}, {removed['file_size_bytes']/1024:.1f}KB\n")
                f.write(f"    hash_distance: ahash={removed['ahash_distance']}, phash={removed['phash_distance']}\n")
            
            if len(group["removed"]) > 5:
                f.write(f"  ... and {len(group['removed']) - 5} more duplicates\n")
            f.write("\n")
        
        if len(duplicate_groups) > 100:
            f.write(f"\n... and {len(duplicate_groups) - 100} more groups (see JSON for full list)\n")
    
    print(f"Text report saved to: {txt_path}")
    
    # Print cross-label warnings (potential labeling issues)
    cross_label_groups = [g for g in duplicate_groups if g["cross_label"]]
    if cross_label_groups:
        print(f"\n⚠️  WARNING: {len(cross_label_groups)} duplicate groups have DIFFERENT labels!")
        print("This may indicate labeling inconsistencies. Check the report for details.")


def main():
    """Main deduplication routine."""
    print("=" * 60)
    print("IMAGE DEDUPLICATION (4-class sources only)")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"Error: {RAW_DATA_DIR} not found. Run download_datasets.py first.")
        return

    print(f"\nSources to deduplicate: {SOURCES_TO_DEDUPLICATE}")
    print("(Mendeley binary data is excluded - used raw for binary dataset)")
    
    print(f"\nScanning {RAW_DATA_DIR} for images...")
    images = collect_images(RAW_DATA_DIR, sources=SOURCES_TO_DEDUPLICATE)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found. Make sure datasets are downloaded.")
        return

    by_source = defaultdict(int)
    for img in images:
        parts = img.relative_to(RAW_DATA_DIR).parts
        source = parts[0] if parts else "unknown"
        by_source[source] += 1

    print("\nImages by source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")

    print("\n" + "-" * 40)
    duplicates, hashes = find_duplicates(images)

    OUTPUT_DIR.mkdir(exist_ok=True)
    unique_count = copy_unique_images(images, duplicates, OUTPUT_DIR)

    total_dupes = sum(len(v) for v in duplicates.values())
    print("\n" + "=" * 60)
    print("DEDUPLICATION SUMMARY")
    print("=" * 60)
    print(f"Total images scanned: {len(images)}")
    print(f"Duplicates found: {total_dupes}")
    print(f"Unique images: {unique_count}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # Generate detailed report
    print("\n" + "-" * 40)
    print("Generating deduplication report...")
    generate_deduplication_report(duplicates, hashes, len(images), unique_count)

    print("\nNext step: python -m vitpoultry.data_curation.upload_to_hf")


if __name__ == "__main__":
    main()
