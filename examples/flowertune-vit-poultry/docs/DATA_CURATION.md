# Data Curation Process for Poultry Fecal Disease Classification

This document describes the data curation pipeline for building the federated learning datasets used in our poultry disease classification research.

## Overview

We curate two datasets for different experimental settings:

| Dataset | Classes | Source | HuggingFace Repo |
|---------|---------|--------|------------------|
| **Binary** | healthy, unhealthy | Mendeley Nigeria | `Dianyo/poultry-health` |
| **4-class** | healthy, cocci, ncd, salmo | Zenodo + Roboflow | `Dianyo/poultry-fecal-fl` |

## Data Sources

### 1. Zenodo AI4D Tanzania Datasets

We download two Zenodo records from the same research project:

#### a) Record 4628934 - FARM Data (Main Dataset)
- **DOI**: 10.5281/zenodo.4628934
- **URL**: https://zenodo.org/records/4628934
- **Content**: Poultry fecal images from farms in Arusha and Kilimanjaro, Tanzania
- **Collection Period**: September 2020 - February 2021
- **Labels**: 4-class
- **Image Counts**:
  - cocci.zip: 2,103 images
  - healthy.zip: 2,057 images
  - salmo.zip: 2,276 images
  - ncd.zip: 376 images
  - **Total: 6,812 images**
- **Download Location**: `raw_data/zenodo_farm/`

#### b) Record 5801834 - LAB Data (PCR-Confirmed)
- **DOI**: 10.5281/zenodo.5801834
- **Content**: Laboratory PCR-confirmed samples (higher quality, smaller)
- **Labels**: 4-class
- **Image Count**: ~1,255 images
- **Use Case**: High-confidence validation subset
- **Download Location**: `raw_data/zenodo_lab/`

### 2. Roboflow Fecal Disease Datasets
Two datasets from Roboflow Universe:

#### a) disease-images-fecal (Classification)
- **URL**: https://universe.roboflow.com/fecal/disease-images-fecal
- **Format**: Folder (classification)
- **Labels**: 4-class

#### b) fecal-lbh0j (Object Detection)
- **URL**: https://universe.roboflow.com/thesis-pr4oh/fecal-lbh0j
- **Format**: COCO (object detection)
- **Labels**: Coccidiosis-like, Healthy, NCD-like, Salmonella-like
- **Preprocessing**: Converted to classification format (see below)

### 3. Mendeley Nigeria Dataset
- **DOI**: 10.17632/8pnbzpt2k9.1
- **Content**: Poultry fecal images from Nigerian farms
- **Labels**: Binary (healthy, unhealthy)
- **Download**: Manual or via Mendeley API

### 4. HuggingFace Existing Datasets
- `Dianyo/fecal-health`: Zenodo LAB data (4-class, PCR-confirmed)
- `Dianyo/poultry-health`: Existing binary dataset (subset of Mendeley)

## Curation Pipeline

### Step 1: Download Raw Data

```bash
python -m vitpoultry.data_curation.download_datasets
```

**Process**:
1. Downloads from HuggingFace (existing datasets for verification)
2. Downloads from Zenodo API (Tanzania farm data)
3. Downloads from Roboflow API (both classification and detection datasets)
4. Attempts Mendeley download (falls back to manual instructions)

**Skip Logic**: Each source checks if data already exists to avoid re-downloading.

### Step 2: Convert COCO to Classification Format

```bash
python -m vitpoultry.data_curation.convert_coco_to_classification
```

**Process**:
1. Parses COCO annotation JSON files
2. Maps COCO categories to our 4-class scheme:
   - `Healthy` → `healthy`
   - `Coccidiosis-like` → `cocci`
   - `NCD-like` → `ncd`
   - `Salmonella-like` → `salmo`
3. **Discards multi-class images**: Images with annotations for multiple different classes are excluded to ensure single-label classification
4. Reorganizes images into class-based folder structure

**Category Mapping**:
```
COCO Category          → Our Class
─────────────────────────────────
Healthy                → healthy
Coccidiosis-like       → cocci
NCD-like               → ncd
Salmonella-like        → salmo
```

### Step 3: Image Deduplication

```bash
python -m vitpoultry.data_curation.deduplicate
```

**Process**:
1. Scans all images from: `zenodo_tanzania/`, `roboflow/`, `huggingface/`
2. Computes perceptual hashes using `imagehash` library:
   - Average Hash (aHash): hash_size=16
   - Perceptual Hash (pHash): hash_size=16
3. Identifies duplicates:
   - **Exact duplicates**: Identical hash values
   - **Near-duplicates**: Hash distance ≤ 5 for both aHash and pHash
4. Copies unique images to `deduplicated_data/`

**Deduplication Algorithm**:
```python
HASH_SIZE = 16
HASH_THRESHOLD = 5

for each image:
    ahash = average_hash(image, hash_size=16)
    phash = perceptual_hash(image, hash_size=16)
    
    if (ahash_distance <= 5) AND (phash_distance <= 5):
        mark_as_duplicate()
```

### Step 4: Verification

```bash
python -m vitpoultry.data_curation.verify_datasets
```

**Process**:
1. Counts images in local downloads
2. Loads HuggingFace datasets and compares counts
3. Reports discrepancies (e.g., if Mendeley has more images than HuggingFace)

### Step 5: Upload to HuggingFace

```bash
# Binary dataset
python -m vitpoultry.data_curation.upload_to_hf --dataset binary

# 4-class dataset
python -m vitpoultry.data_curation.upload_to_hf --dataset 4class
```

**Process**:
1. Builds dataset from local files
2. Extracts labels from folder structure
3. Creates train/test splits (80/20 random split)
4. Uploads to HuggingFace Hub

## Label Standardization

### 4-Class Labels
| ID | Name | Variants in Sources |
|----|------|---------------------|
| 0 | healthy | Healthy, HEALTHY |
| 1 | cocci | Cocci, Coccidiosis, Coccidiosis-like |
| 2 | ncd | NCD, Newcastle, NCD-like |
| 3 | salmo | Salmo, Salmonella, Salmonella-like |

### Binary Labels
| ID | Name | Variants in Sources |
|----|------|---------------------|
| 0 | healthy | Healthy, HEALTHY |
| 1 | unhealthy | Unhealthy, Sick, Diseased |

## Data Splits

Both datasets use stratified random 80/20 train/test splits:
- **Train**: 80% of images (stratified by class)
- **Test**: 20% of images (stratified by class)

**Note**: The original Zenodo dataset does not include farm metadata. For federated learning experiments simulating multiple farms, we use synthetic partitioning strategies:
- **IID**: Uniform random partition across clients
- **Non-IID**: Dirichlet partitioning (α controls heterogeneity)

## Quality Control

### Multi-Class Image Filtering
Images with multiple object annotations of **different classes** are discarded during COCO conversion. This ensures each image has a single, unambiguous label.

### Duplicate Removal
Perceptual hashing removes:
- Exact duplicates (same image from multiple sources)
- Near-duplicates (slightly modified versions, resized copies)

## Output Directory Structure

```
raw_data/
├── huggingface/
│   ├── Dianyo_fecal-health/
│   └── Dianyo_poultry-health/
├── zenodo_farm/                    # Record 4628934 (6,812 images)
│   ├── cocci/
│   ├── healthy/
│   ├── salmo/
│   └── ncd/
├── zenodo_lab/                     # Record 5801834 (PCR-confirmed)
│   └── [extracted lab data]
├── roboflow/
│   ├── fecal_disease-images-fecal/
│   │   ├── healthy/
│   │   ├── cocci/
│   │   ├── ncd/
│   │   └── salmo/
│   ├── thesis-pr4oh_fecal-lbh0j/
│   │   └── [COCO format]
│   └── thesis-pr4oh_fecal-lbh0j_classification/
│       ├── healthy/
│       ├── cocci/
│       ├── ncd/
│       └── salmo/
└── mendeley_nigeria/
    ├── healthy/
    └── unhealthy/

deduplicated_data/
├── zenodo_tanzania/
├── roboflow/
└── huggingface/
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `download_datasets.py` | Download from all sources |
| `convert_coco_to_classification.py` | Convert COCO detection → classification |
| `deduplicate.py` | Remove duplicate images |
| `verify_datasets.py` | Compare local vs HuggingFace counts |
| `upload_to_hf.py` | Build and upload datasets |
| `merge_and_upload.py` | Alternative merge script (legacy) |

## Reproducibility

To reproduce the exact dataset:

```bash
cd examples/flowertune-vit-poultry
source .venv/bin/activate

# Full pipeline
python -m vitpoultry.data_curation.download_datasets
python -m vitpoultry.data_curation.convert_coco_to_classification
python -m vitpoultry.data_curation.deduplicate
python -m vitpoultry.data_curation.verify_datasets
python -m vitpoultry.data_curation.upload_to_hf --dataset 4class
```

## Version History

- **v1**: Initial dataset from Zenodo LAB data only
- **v2**: Added Roboflow datasets, COCO conversion, deduplication pipeline
- **v3**: Added farm-based splits for realistic FL evaluation
