"""Deduplicate images using perceptual hashing.

This script scans downloaded images and removes duplicates using average hash (aHash)
and perceptual hash (pHash) from the imagehash library.

Usage:
    python -m vitpoultry.data_curation.deduplicate
"""

import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("deduplicated_data")
HASH_SIZE = 16
HASH_THRESHOLD = 5


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


def find_duplicates(image_paths: list[Path]) -> dict[Path, list[Path]]:
    """Find duplicate images using perceptual hashing.

    Returns:
        Dictionary mapping canonical image to list of duplicates.
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

    return duplicates


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


def collect_images(data_dir: Path) -> list[Path]:
    """Collect all image files from subdirectories."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
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


def main():
    """Main deduplication routine."""
    print("=" * 60)
    print("IMAGE DEDUPLICATION")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"Error: {RAW_DATA_DIR} not found. Run download_datasets.py first.")
        return

    print(f"\nScanning {RAW_DATA_DIR} for images...")
    images = collect_images(RAW_DATA_DIR)
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
    duplicates = find_duplicates(images)

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

    print("\nNext step: python -m vitpoultry.data_curation.merge_and_upload")


if __name__ == "__main__":
    main()
