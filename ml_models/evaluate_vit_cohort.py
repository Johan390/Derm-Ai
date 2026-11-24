"""
DermAI Vision Transformer (ViT) Cohort Evaluation Script

Evaluates `vit_skin_lesion_exp2_best.keras` across:
1. The 265 class-balanced test subset (identical to `model_comparison.json`).
2. The full held-out test split (1,501 images from `test_split.csv`).

Computes:
- Top-1 Accuracy, Balanced Accuracy
- Macro-averaged and Weighted-averaged Precision, Recall, F1-Score
- Per-class metrics (Precision, Recall, F1, Support)
- Confusion Matrix (7x7)
- ROC-AUC (One-vs-Rest macro and weighted)
- Inference latency per sample (ms)
- Direct comparison with MobileNetV2, EfficientNetB0, and Ensemble.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, roc_auc_score
)

# Setup workspace paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import preprocess_input as vit_preprocess_input
from ml_models import vision_transformer

# Configuration
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "ml_models", "saved_models")
EVAL_RESULTS_DIR = os.path.join(BASE_DIR, "ml_models", "evaluation_results")
DATA_SPLITS_DIR = os.path.join(BASE_DIR, "ml_models", "data_splits")
IMAGES_DIR = os.path.join(BASE_DIR, "ml_dataset", "HAM10000", "images")

VIT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "vit_skin_lesion_exp2_best.keras")
TEST_SPLIT_CSV = os.path.join(DATA_SPLITS_DIR, "test_split.csv")
MODEL_COMP_JSON = os.path.join(EVAL_RESULTS_DIR, "model_comparison.json")
OUTPUT_JSON = os.path.join(EVAL_RESULTS_DIR, "vit_cohort_evaluation.json")

CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}
RISK_LEVELS = {
    "mel": "High Risk",
    "bcc": "High Risk",
    "akiec": "Moderate Risk",
    "nv": "Low Risk",
    "bkl": "Low Risk",
    "df": "Low Risk",
    "vasc": "Low Risk"
}


def load_and_preprocess_images(df):
    """Loads and preprocesses images matching the ViT input specification."""
    images = []
    labels = []
    valid_indices = []
    missing_count = 0

    for idx, row in df.iterrows():
        img_id = str(row["image_id"]).strip()
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.isfile(img_path):
            missing_count += 1
            continue

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            missing_count += 1
            continue

        h, w = bgr.shape[:2]
        side = min(h, w)
        cy, cx = (h - side) // 2, (w - side) // 2
        crop = bgr[cy:cy + side, cx:cx + side]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        scaled = vit_preprocess_input(rgb)

        images.append(scaled)
        labels.append(int(row["label_idx"]))
        valid_indices.append(idx)

    if missing_count > 0:
        print(f"[Warning] {missing_count} images were missing or unreadable on disk.")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def compute_metrics(y_true, y_pred, y_probs):
    """Computes full suite of classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS))))

    # ROC-AUC (One-vs-Rest)
    try:
        y_true_one_hot = np.eye(len(CLASS_LABELS))[y_true]
        roc_macro = roc_auc_score(y_true_one_hot, y_probs, multi_class="ovr", average="macro")
        roc_weighted = roc_auc_score(y_true_one_hot, y_probs, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"[Warning] ROC-AUC computation notice: {e}")
        roc_macro = None
        roc_weighted = None

    per_class = {}
    for i, label in enumerate(CLASS_LABELS):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true == i, y_pred == i, average="binary", zero_division=0
        )
        support = int(np.sum(y_true == i))
        per_class[label] = {
            "name": CLASS_NAMES[label],
            "risk_level": RISK_LEVELS[label],
            "support": support,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4)
        }

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "roc_auc_ovr_macro": round(float(roc_macro), 4) if roc_macro is not None else None,
        "roc_auc_ovr_weighted": round(float(roc_weighted), 4) if roc_weighted is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


def main():
    print("=" * 80)
    print("DERMAI: VISION TRANSFORMER (ViT) COHORT BENCHMARK & EVALUATION")
    print("=" * 80)

    # 1. Load ViT Model
    print(f"\n[1/4] Loading trained ViT model from: {VIT_MODEL_PATH}")
    if not os.path.isfile(VIT_MODEL_PATH):
        raise FileNotFoundError(f"ViT model file missing: {VIT_MODEL_PATH}")

    model = keras.models.load_model(VIT_MODEL_PATH, compile=False)
    print(f"Model successfully loaded: {model.name}")
    print(f"Input shape: {model.input_shape}, Output shape: {model.output_shape}")

    # 2. Load Test Splits
    print(f"\n[2/4] Reading held-out test split: {TEST_SPLIT_CSV}")
    test_df = pd.read_csv(TEST_SPLIT_CSV)
    print(f"Total samples in full test split: {len(test_df)}")

    # Sample class-balanced 265 subset matching model_comparison.json
    subsets = []
    for _, group in test_df.groupby("label_idx"):
        subsets.append(group.sample(min(len(group), 45), random_state=42))
    balanced_df = pd.concat(subsets, ignore_index=True)
    print(f"Class-balanced benchmark cohort: {len(balanced_df)} samples")

    # 3. Evaluate on Class-Balanced Subset (265 images)
    print("\n[3/4] Running inference on 265 class-balanced test cohort...")
    X_bal, y_bal = load_and_preprocess_images(balanced_df)
    print(f"Loaded {len(X_bal)} preprocessed images.")

    # Measure latency
    t0 = time.time()
    probs_bal = model(X_bal, training=False).numpy()
    total_time_bal = time.time() - t0
    latency_ms_bal = (total_time_bal / len(X_bal)) * 1000.0
    preds_bal = np.argmax(probs_bal, axis=1)

    metrics_bal = compute_metrics(y_bal, preds_bal, probs_bal)
    metrics_bal["sample_count"] = len(y_bal)
    metrics_bal["inference_latency_ms"] = round(float(latency_ms_bal), 2)
    metrics_bal["total_inference_time_sec"] = round(float(total_time_bal), 2)

    print(f"\n--- 265 Class-Balanced Cohort Results ---")
    print(f"Accuracy:          {metrics_bal['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy: {metrics_bal['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics_bal['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics_bal['weighted_f1']:.4f}")
    if metrics_bal['roc_auc_ovr_macro'] is not None:
        print(f"ROC-AUC (Macro):   {metrics_bal['roc_auc_ovr_macro']:.4f}")
    print(f"Avg Latency/image: {latency_ms_bal:.2f} ms")

    # 4. Evaluate on Full Test Split (1,501 images)
    print("\n[4/4] Running inference on full held-out test split (1,501 images)...")
    X_full, y_full = load_and_preprocess_images(test_df)
    print(f"Loaded {len(X_full)} full preprocessed test images.")

    # Batch inference for memory efficiency
    batch_size = 64
    probs_full_list = []
    t0_full = time.time()
    for i in range(0, len(X_full), batch_size):
        b_x = X_full[i:i + batch_size]
        b_p = model(b_x, training=False).numpy()
        probs_full_list.append(b_p)
    probs_full = np.concatenate(probs_full_list, axis=0)
    total_time_full = time.time() - t0_full
    latency_ms_full = (total_time_full / len(X_full)) * 1000.0
    preds_full = np.argmax(probs_full, axis=1)

    metrics_full = compute_metrics(y_full, preds_full, probs_full)
    metrics_full["sample_count"] = len(y_full)
    metrics_full["inference_latency_ms"] = round(float(latency_ms_full), 2)
    metrics_full["total_inference_time_sec"] = round(float(total_time_full), 2)

    print(f"\n--- Full Test Split (1,501 Images) Results ---")
    print(f"Accuracy:          {metrics_full['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy: {metrics_full['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics_full['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics_full['weighted_f1']:.4f}")
    if metrics_full['roc_auc_ovr_macro'] is not None:
        print(f"ROC-AUC (Macro):   {metrics_full['roc_auc_ovr_macro']:.4f}")
    print(f"Avg Latency/image: {latency_ms_full:.2f} ms")

    # 5. Load and align existing model comparison data
    existing_comp = {}
    if os.path.isfile(MODEL_COMP_JSON):
        with open(MODEL_COMP_JSON, "r", encoding="utf-8") as f:
            existing_comp = json.load(f)

    # 6. Save consolidated evaluation JSON
    eval_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "v2.0",
        "model_name": "DermAI_VisionTransformer",
        "architecture": "Vision Transformer (ViT-16)",
        "checkpoint_file": "vit_skin_lesion_exp2_best.keras",
        "classes": CLASS_LABELS,
        "class_names": CLASS_NAMES,
        "risk_levels": RISK_LEVELS,
        "balanced_test_cohort_265": metrics_bal,
        "full_test_split_1501": metrics_full,
        "cross_architecture_comparison_265": {
            "mobilenetv2_baseline": existing_comp.get("overall_metrics", {}).get("mobilenetv2_baseline", {}),
            "efficientnetb0_advanced": existing_comp.get("overall_metrics", {}).get("efficientnetb0_advanced", {}),
            "soft_voting_ensemble": existing_comp.get("overall_metrics", {}).get("soft_voting_ensemble", {}),
            "vision_transformer": {
                "accuracy": metrics_bal["accuracy"],
                "balanced_accuracy": metrics_bal["balanced_accuracy"],
                "macro_precision": metrics_bal["macro_precision"],
                "macro_recall": metrics_bal["macro_recall"],
                "macro_f1": metrics_bal["macro_f1"],
                "roc_auc_ovr_macro": metrics_bal.get("roc_auc_ovr_macro"),
                "confusion_matrix": metrics_bal["confusion_matrix"]
            }
        }
    }

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n[OK] ViT cohort evaluation saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
