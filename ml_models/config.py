"""
DermAI Machine Learning Configuration & Hyperparameters

Milestone 2 Configuration:
- Paths for HAM10000 dataset, split manifests, and model checkpoints
- Canonical 7-class taxonomy mapping with integer indices and clinical descriptors
- MobileNetV2 input dimensions and batch configurations
- Split ratios and random seed for reproducible stratified group splitting
"""

import os
from typing import Dict, List, Tuple

# Base Project Paths
ML_MODELS_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(ML_MODELS_DIR)

# Dataset Paths
DATASET_ROOT: str = os.path.join(PROJECT_ROOT, "ml_dataset", "HAM10000")
METADATA_CSV: str = os.path.join(DATASET_ROOT, "HAM10000_metadata.csv")
IMAGES_DIR: str = os.path.join(DATASET_ROOT, "images")

# Generated Split Manifest Paths
DATA_SPLITS_DIR: str = os.path.join(ML_MODELS_DIR, "data_splits")
TRAIN_SPLIT_CSV: str = os.path.join(DATA_SPLITS_DIR, "train_split.csv")
VAL_SPLIT_CSV: str = os.path.join(DATA_SPLITS_DIR, "val_split.csv")
TEST_SPLIT_CSV: str = os.path.join(DATA_SPLITS_DIR, "test_split.csv")

# Model Persistence & Evaluation Artifact Directories
SAVED_MODELS_DIR: str = os.path.join(ML_MODELS_DIR, "saved_models")
EVALUATION_DIR: str = os.path.join(ML_MODELS_DIR, "evaluation_results")
BEST_MODEL_PATH: str = os.path.join(SAVED_MODELS_DIR, "mobilenetv2_skin_lesion_best.keras")
MODEL_METADATA_PATH: str = os.path.join(SAVED_MODELS_DIR, "model_metadata.json")

# Ensure required directories exist
for directory in [DATA_SPLITS_DIR, SAVED_MODELS_DIR, EVALUATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Canonical 7-Class Taxonomy (Alphabetical order per HAM10000 / ISIC convention)
CLASS_LABELS: List[str] = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions (solar lentigines / seborrheic keratoses)
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc"    # Vascular lesions (angiomas, pyogenic granulomas)
]

NUM_CLASSES: int = len(CLASS_LABELS)

LABEL_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(CLASS_LABELS)}

CLASS_NAMES: Dict[str, str] = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

RISK_LEVELS: Dict[str, str] = {
    "mel": "High Risk",
    "bcc": "High Risk",
    "akiec": "Moderate Risk",
    "nv": "Low Risk",
    "bkl": "Low Risk",
    "df": "Low Risk",
    "vasc": "Low Risk"
}

# Image Dimension & Normalization Specifications for MobileNetV2
IMAGE_WIDTH: int = 224
IMAGE_HEIGHT: int = 224
IMAGE_CHANNELS: int = 3
TARGET_IMAGE_SIZE: Tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# Splitting Hyperparameters
SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}
RANDOM_SEED: int = 42

# Training Hyperparameters (Stage 2 Reference)
BATCH_SIZE: int = 64
STAGE1_EPOCHS: int = 15
STAGE2_EPOCHS: int = 15
STAGE1_LR: float = 1e-3
STAGE2_LR: float = 1e-5

