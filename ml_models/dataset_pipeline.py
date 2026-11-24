"""
DermAI Dataset Pipeline & Leakage-Free Stratified Group Splitting

Milestone 2 - Stage 1 Implementation:
- Ingestion and deep physical validation of HAM10000 images and metadata
- Canonical 7-class indexing and metadata enrichment
- Stratified Group K-Fold splitting by lesion_id (70% train / 15% val / 15% test)
- Strict mathematical proof of zero lesion overlap across all splits
- Export of clean manifest CSVs for model training and evaluation
"""

import os
import sys
from typing import Dict, Tuple, Set, Any
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# Import configuration
try:
    from .config import (
        METADATA_CSV,
        IMAGES_DIR,
        DATA_SPLITS_DIR,
        TRAIN_SPLIT_CSV,
        VAL_SPLIT_CSV,
        TEST_SPLIT_CSV,
        CLASS_LABELS,
        LABEL_TO_IDX,
        CLASS_NAMES,
        RISK_LEVELS,
        RANDOM_SEED
    )
except ImportError:
    from config import (
        METADATA_CSV,
        IMAGES_DIR,
        DATA_SPLITS_DIR,
        TRAIN_SPLIT_CSV,
        VAL_SPLIT_CSV,
        TEST_SPLIT_CSV,
        CLASS_LABELS,
        LABEL_TO_IDX,
        CLASS_NAMES,
        RISK_LEVELS,
        RANDOM_SEED
    )


def load_and_validate_dataset(
    metadata_csv_path: str = METADATA_CSV,
    images_dir_path: str = IMAGES_DIR,
    verify_images: bool = True
) -> pd.DataFrame:
    """
    Loads HAM10000 metadata, matches every record to its physical image file,
    validates readability, and enriches records with canonical class indices.
    """
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv_path}")
    if not os.path.exists(images_dir_path):
        raise FileNotFoundError(f"Images directory not found: {images_dir_path}")

    print(f"[Dataset Pipeline] Reading metadata from: {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)
    total_records = len(df)
    print(f"[Dataset Pipeline] Loaded {total_records} metadata rows.")

    # Validate required columns
    required_cols = {"lesion_id", "image_id", "dx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")

    # Verify diagnostic classes
    unknown_dx = set(df["dx"].unique()) - set(CLASS_LABELS)
    if unknown_dx:
        raise ValueError(f"Unrecognized diagnostic classes found: {unknown_dx}")

    # Build and verify physical image file paths
    print(f"[Dataset Pipeline] Validating image presence in: {images_dir_path}")
    image_paths = []
    missing_files = []

    for _, row in df.iterrows():
        img_id = row["image_id"]
        img_path = os.path.join(images_dir_path, f"{img_id}.jpg")
        if not os.path.isfile(img_path):
            missing_files.append(img_path)
        image_paths.append(img_path)

    if missing_files:
        raise FileNotFoundError(
            f"Integrity check failed: {len(missing_files)} metadata records lack matching image files on disk. "
            f"First missing: {missing_files[0]}"
        )

    df["image_path"] = image_paths

    # Verify binary readability of images (sample or full check)
    if verify_images:
        print("[Dataset Pipeline] Verifying image readability via Pillow...")
        corrupted = 0
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as err:
                print(f"[ERROR] Corrupted image file: {path} ({err})")
                corrupted += 1
        if corrupted > 0:
            raise IOError(f"Found {corrupted} corrupted image files on disk.")
        print(f"[Dataset Pipeline] All {len(image_paths)} images verified readable without corruption.")

    # Map class labels to integer indices and metadata
    df["label_idx"] = df["dx"].map(LABEL_TO_IDX)
    df["disease_name"] = df["dx"].map(CLASS_NAMES)
    df["risk_level"] = df["dx"].map(RISK_LEVELS)

    return df


def create_stratified_group_splits(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates a 70% train, 15% validation, and 15% testing split using StratifiedGroupKFold.
    
    Guarantees:
    1. Grouping by `lesion_id`: Images of the same lesion are NEVER split across folds.
    2. Zero data leakage: Intersection between any two split lesion sets is empty.
    3. Stratification: Preserves the empirical 7-class distribution in all splits.
    """
    print(f"[Dataset Pipeline] Generating stratified group splits (Seed={random_seed})...")
    
    # 20 folds = 5% per fold: 14 folds (70%) for train, 3 folds (15%) for val, 3 folds (15%) for test
    n_splits = 20
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    folds = list(sgkf.split(df, y=df["dx"], groups=df["lesion_id"]))

    df = df.copy()
    df["fold"] = -1
    for fold_idx, (_, test_indices) in enumerate(folds):
        df.iloc[test_indices, df.columns.get_loc("fold")] = fold_idx

    # Partition folds into splits
    train_mask = df["fold"].isin(range(0, 14))  # 14/20 = 70%
    val_mask = df["fold"].isin(range(14, 17))    # 3/20 = 15%
    test_mask = df["fold"].isin(range(17, 20))   # 3/20 = 15%

    train_df = df[train_mask].drop(columns=["fold"]).copy().reset_index(drop=True)
    val_df = df[val_mask].drop(columns=["fold"]).copy().reset_index(drop=True)
    test_df = df[test_mask].drop(columns=["fold"]).copy().reset_index(drop=True)

    # Re-verify zero leakage
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    overlap_train_val = train_lesions.intersection(val_lesions)
    overlap_train_test = train_lesions.intersection(test_lesions)
    overlap_val_test = val_lesions.intersection(test_lesions)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"CRITICAL DATA LEAKAGE: Overlapping lesion IDs detected! "
            f"Train/Val: {len(overlap_train_val)}, "
            f"Train/Test: {len(overlap_train_test)}, "
            f"Val/Test: {len(overlap_val_test)}"
        )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = DATA_SPLITS_DIR
) -> Dict[str, str]:
    """Saves split manifests to CSV files with standardized column ordering."""
    os.makedirs(output_dir, exist_ok=True)

    target_columns = [
        "image_id",
        "lesion_id",
        "dx",
        "label_idx",
        "disease_name",
        "risk_level",
        "image_path",
        "dx_type",
        "age",
        "sex",
        "localization",
        "dataset"
    ]
    # Reorder columns that exist
    cols_to_use = [col for col in target_columns if col in train_df.columns]

    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")

    train_df[cols_to_use].to_csv(train_path, index=False)
    val_df[cols_to_use].to_csv(val_path, index=False)
    test_df[cols_to_use].to_csv(test_path, index=False)

    print(f"[Dataset Pipeline] Saved train split ({len(train_df)} rows) -> {train_path}")
    print(f"[Dataset Pipeline] Saved val split   ({len(val_df)} rows) -> {val_path}")
    print(f"[Dataset Pipeline] Saved test split  ({len(test_df)} rows) -> {test_path}")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Executes comprehensive clinical and machine learning validation on splits:
    - Image and lesion counts
    - Absolute zero lesion overlap verification
    - Class representation and balance
    - File existence on disk
    """
    total_images = len(train_df) + len(val_df) + len(test_df)
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    all_unique_lesions = train_lesions.union(val_lesions).union(test_lesions)

    overlap_train_val = train_lesions.intersection(val_lesions)
    overlap_train_test = train_lesions.intersection(test_lesions)
    overlap_val_test = val_lesions.intersection(test_lesions)
    total_leakage = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)

    # File existence check
    all_paths = list(train_df["image_path"]) + list(val_df["image_path"]) + list(test_df["image_path"])
    missing_paths = [p for p in all_paths if not os.path.isfile(p)]

    # Compute class distribution tables
    splits_summary = {}
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df["dx"].value_counts().to_dict()
        percentages = (split_df["dx"].value_counts(normalize=True) * 100).round(2).to_dict()
        splits_summary[name] = {
            "image_count": len(split_df),
            "image_percentage": round((len(split_df) / total_images) * 100, 2),
            "unique_lesions": len(set(split_df["lesion_id"])),
            "class_counts": counts,
            "class_percentages": percentages,
            "all_classes_present": len(counts) == len(CLASS_LABELS)
        }

    validation_result = {
        "total_images": total_images,
        "total_unique_lesions": len(all_unique_lesions),
        "zero_leakage_verified": total_leakage == 0,
        "overlapping_lesion_counts": {
            "train_val": len(overlap_train_val),
            "train_test": len(overlap_train_test),
            "val_test": len(overlap_val_test)
        },
        "all_image_paths_exist": len(missing_paths) == 0,
        "missing_paths_count": len(missing_paths),
        "splits": splits_summary
    }

    return validation_result


def print_validation_report(val_res: Dict[str, Any]):
    """Prints formatted validation summary to stdout."""
    print("\n" + "=" * 78)
    print("DERMAI MILESTONE 2 – STAGE 1 DATASET SPLIT VALIDATION")
    print("=" * 78)
    print(f"Total Images Processed:       {val_res['total_images']:,}")
    print(f"Total Unique Lesions:         {val_res['total_unique_lesions']:,}")
    print(f"Zero Data Leakage:            {'PASSED (0 overlapping lesions)' if val_res['zero_leakage_verified'] else 'FAILED'}")
    print(f"All Image Paths Exist:        {'PASSED (100% on disk)' if val_res['all_image_paths_exist'] else 'FAILED'}")

    print("\n" + "-" * 78)
    print(f"{'Split':<8} | {'Images':<8} | {'Image %':<8} | {'Lesions':<8} | {'Lesion %':<9} | {'All 7 Classes':<13}")
    print("-" * 78)
    total_lesions = val_res["total_unique_lesions"]
    for split_name in ["Train", "Val", "Test"]:
        s = val_res["splits"][split_name]
        lesion_pct = round((s["unique_lesions"] / total_lesions) * 100, 2)
        classes_ok = "YES (7/7)" if s["all_classes_present"] else "NO"
        print(f"{split_name:<8} | {s['image_count']:<8} | {s['image_percentage']:<7}% | {s['unique_lesions']:<8} | {lesion_pct:<8}% | {classes_ok:<13}")
    print("-" * 78)

    print("\nClass Distribution Across Splits (Count & % of split):")
    print("-" * 78)
    header = f"{'Class':<8} | {'Train Count':<12} | {'Train %':<8} | {'Val Count':<10} | {'Val %':<8} | {'Test Count':<11} | {'Test %':<8}"
    print(header)
    print("-" * 78)

    for dx in CLASS_LABELS:
        tr_c = val_res["splits"]["Train"]["class_counts"].get(dx, 0)
        tr_p = val_res["splits"]["Train"]["class_percentages"].get(dx, 0.0)
        va_c = val_res["splits"]["Val"]["class_counts"].get(dx, 0)
        va_p = val_res["splits"]["Val"]["class_percentages"].get(dx, 0.0)
        te_c = val_res["splits"]["Test"]["class_counts"].get(dx, 0)
        te_p = val_res["splits"]["Test"]["class_percentages"].get(dx, 0.0)
        print(f"{dx:<8} | {tr_c:<12} | {tr_p:>6.2f}% | {va_c:<10} | {va_p:>6.2f}% | {te_c:<11} | {te_p:>6.2f}%")
    print("-" * 78)
    print("=" * 78 + "\n")


if __name__ == "__main__":
    # Execute end-to-end dataset preparation and splitting
    dataset_df = load_and_validate_dataset(METADATA_CSV, IMAGES_DIR, verify_images=True)
    train_split, val_split, test_split = create_stratified_group_splits(dataset_df, RANDOM_SEED)
    save_splits(train_split, val_split, test_split, DATA_SPLITS_DIR)
    results = validate_splits(train_split, val_split, test_split)
    print_validation_report(results)

