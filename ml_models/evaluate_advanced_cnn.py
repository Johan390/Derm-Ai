"""
DermAI Milestone 3: Advanced CNN Evaluation & Hyperparameter Optimization

Performs:
1. Lightweight hyperparameter optimization of EfficientNetB0 classification head using validation data.
   - Evaluates learning rates and dropout configurations on validation split only.
2. Trains the best head weights on a stratified representative training subset (frozen backbone).
3. Evaluates both MobileNetV2 and EfficientNetB0 on held-out test split images.
4. Generates genuine quantitative comparison metrics:
   - Accuracy, balanced accuracy, macro precision, recall, F1-score
   - Per-class performance
   - Confusion matrix
5. Persists evaluation artifacts to ml_models/evaluation_results/model_comparison.json
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
import keras
from keras import layers
from keras.applications.efficientnet import preprocess_input as effnet_preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

from config import (
    CLASS_LABELS, LABEL_TO_IDX, IDX_TO_LABEL, CLASS_NAMES, RISK_LEVELS,
    DATASET_ROOT, IMAGES_DIR, TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV,
    BEST_MODEL_PATH, SAVED_MODELS_DIR, EVALUATION_DIR
)
from advanced_cnn import build_efficientnetb0_classifier


def load_image_batch(df_subset, preprocess_fn):
    """Loads and preprocesses a batch of images from disk."""
    import cv2
    images = []
    labels = []
    valid_indices = []

    for idx, row in df_subset.iterrows():
        img_id = row["image_id"]
        # Look in images directory
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
            
        h, w = bgr.shape[:2]
        min_dim = min(h, w)
        cy, cx = (h - min_dim) // 2, (w - min_dim) // 2
        crop = bgr[cy:cy + min_dim, cx:cx + min_dim]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        scaled = preprocess_fn(rgb)
        
        images.append(scaled)
        labels.append(int(row["label_idx"]))
        valid_indices.append(idx)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def extract_features(base_model, images, batch_size=32):
    """Batched feature extraction through frozen backbone."""
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        feats = base_model(batch, training=False)
        # Apply global average pooling
        pooled = np.mean(feats.numpy(), axis=(1, 2))
        features.append(pooled)
    return np.concatenate(features, axis=0)


def run_milestone3_model_evaluation():
    print("=" * 70)
    print("DERMAI MILESTONE 3: ADVANCED CNN EVALUATION & COMPARISON")
    print("=" * 70)
    
    # 1. Load data splits
    train_df = pd.read_csv(TRAIN_SPLIT_CSV)
    val_df = pd.read_csv(VAL_SPLIT_CSV)
    test_df = pd.read_csv(TEST_SPLIT_CSV)

    print(f"Dataset splits loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Sample representative subsets for CPU feasibility (stratified by label_idx)
    def strat_sample(df, n_per_class):
        subsets = []
        for _, group in df.groupby("label_idx"):
            subsets.append(group.sample(min(len(group), n_per_class), random_state=42))
        return pd.concat(subsets, ignore_index=True)

    train_sample = strat_sample(train_df, 80)
    val_sample = strat_sample(val_df, 30)
    test_sample = strat_sample(test_df, 45)

    print(f"Sampled subsets for CPU execution: Train={len(train_sample)}, Val={len(val_sample)}, Test={len(test_sample)}")

    # 2. Load images for EfficientNetB0
    print("\n[1/5] Loading EfficientNetB0 images...")
    X_train_eff, y_train = load_image_batch(train_sample, effnet_preprocess_input)
    X_val_eff, y_val = load_image_batch(val_sample, effnet_preprocess_input)
    X_test_eff, y_test = load_image_batch(test_sample, effnet_preprocess_input)
    print(f"Loaded: X_train={X_train_eff.shape}, X_val={X_val_eff.shape}, X_test={X_test_eff.shape}")

    # 3. Feature extraction with EfficientNetB0 backbone
    print("\n[2/5] Precomputing EfficientNetB0 backbone features...")
    effnet_full, effnet_base = build_efficientnetb0_classifier()
    
    t0 = time.time()
    feats_train = extract_features(effnet_base, X_train_eff)
    feats_val = extract_features(effnet_base, X_val_eff)
    feats_test = extract_features(effnet_base, X_test_eff)
    print(f"Feature extraction completed in {time.time() - t0:.2f}s (Features shape: {feats_train.shape})")

    # 4. Hyperparameter optimization on validation set
    print("\n[3/5] Hyperparameter optimization on validation set...")
    param_grid = [
        {"lr": 1e-3, "dropout": 0.3, "dense": 128},
        {"lr": 5e-4, "dropout": 0.2, "dense": 128},
        {"lr": 1e-3, "dropout": 0.2, "dense": 256},
    ]

    best_val_acc = 0.0
    best_config = None
    best_head = None
    hpo_results = []

    y_train_cat = keras.utils.to_categorical(y_train, num_classes=7)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=7)

    for cfg in param_grid:
        head_in = keras.Input(shape=(1280,))
        x = layers.BatchNormalization()(head_in)
        x = layers.Dropout(cfg["dropout"])(x)
        x = layers.Dense(cfg["dense"], activation="relu")(x)
        x = layers.Dropout(cfg["dropout"] * 0.7)(x)
        head_out = layers.Dense(7, activation="softmax")(x)
        head_model = keras.Model(inputs=head_in, outputs=head_out)
        
        head_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        hist = head_model.fit(
            feats_train, y_train_cat,
            validation_data=(feats_val, y_val_cat),
            epochs=12,
            batch_size=32,
            verbose=0
        )
        
        final_val_acc = float(hist.history["val_accuracy"][-1])
        hpo_results.append({
            "config": cfg,
            "val_accuracy": round(final_val_acc, 4),
            "val_loss": round(float(hist.history["val_loss"][-1]), 4)
        })
        print(f"  Config {cfg} -> Val Acc: {final_val_acc:.4f}, Val Loss: {hist.history['val_loss'][-1]:.4f}")

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_config = cfg
            best_head = head_model

    print(f"\nBest Hyperparameter Configuration: {best_config} with Val Acc: {best_val_acc:.4f}")

    # Build and assemble full EfficientNetB0 model with trained head
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")
    x = effnet_base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = best_head(x)
    final_effnet = keras.Model(inputs=inputs, outputs=x, name="DermAI_EfficientNetB0_Trained")
    
    # Save the model
    effnet_path = os.path.join(SAVED_MODELS_DIR, "efficientnetb0_skin_lesion.keras")
    final_effnet.save(effnet_path)
    print(f"Saved optimized EfficientNetB0 to {effnet_path}")

    # 5. Evaluate EfficientNetB0 on Test Set
    print("\n[4/5] Evaluating EfficientNetB0 on held-out test split...")
    effnet_preds_prob = best_head(feats_test).numpy()
    effnet_preds = np.argmax(effnet_preds_prob, axis=1)

    effnet_acc = accuracy_score(y_test, effnet_preds)
    effnet_bal_acc = balanced_accuracy_score(y_test, effnet_preds)
    effnet_prec, effnet_rec, effnet_f1, _ = precision_recall_fscore_support(
        y_test, effnet_preds, average="macro", zero_division=0
    )
    effnet_cm = confusion_matrix(y_test, effnet_preds, labels=list(range(7)))

    # 6. Evaluate Baseline MobileNetV2 on exact same Test Set
    print("\n[5/5] Evaluating MobileNetV2 on exact same test split...")
    X_test_mob, _ = load_image_batch(test_sample, mobilenet_preprocess_input)
    mobilenet_model = keras.models.load_model(BEST_MODEL_PATH, compile=False)
    mobilenet_preds_prob = mobilenet_model(X_test_mob, training=False).numpy()
    mobilenet_preds = np.argmax(mobilenet_preds_prob, axis=1)

    mob_acc = accuracy_score(y_test, mobilenet_preds)
    mob_bal_acc = balanced_accuracy_score(y_test, mobilenet_preds)
    mob_prec, mob_rec, mob_f1, _ = precision_recall_fscore_support(
        y_test, mobilenet_preds, average="macro", zero_division=0
    )
    mob_cm = confusion_matrix(y_test, mobilenet_preds, labels=list(range(7)))

    # Per-class metrics
    per_class_comparison = {}
    for i, label in enumerate(CLASS_LABELS):
        mob_p, mob_r, mob_f, _ = precision_recall_fscore_support(
            y_test == i, mobilenet_preds == i, average="binary", zero_division=0
        )
        eff_p, eff_r, eff_f, _ = precision_recall_fscore_support(
            y_test == i, effnet_preds == i, average="binary", zero_division=0
        )
        per_class_comparison[label] = {
            "name": CLASS_NAMES[label],
            "risk_level": RISK_LEVELS[label],
            "support": int(np.sum(y_test == i)),
            "mobilenetv2": {
                "precision": round(float(mob_p), 4),
                "recall": round(float(mob_r), 4),
                "f1_score": round(float(mob_f), 4)
            },
            "efficientnetb0": {
                "precision": round(float(eff_p), 4),
                "recall": round(float(eff_r), 4),
                "f1_score": round(float(eff_f), 4)
            }
        }

    # Soft voting ensemble evaluation
    ensemble_probs = (mobilenet_preds_prob + effnet_preds_prob) / 2.0
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ens_acc = accuracy_score(y_test, ensemble_preds)
    ens_bal_acc = balanced_accuracy_score(y_test, ensemble_preds)
    ens_prec, ens_rec, ens_f1, _ = precision_recall_fscore_support(
        y_test, ensemble_preds, average="macro", zero_division=0
    )

    comparison_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_subset_size": len(y_test),
        "hyperparameter_optimization": {
            "search_space": param_grid,
            "results": hpo_results,
            "selected_config": best_config,
            "best_val_accuracy": round(float(best_val_acc), 4)
        },
        "overall_metrics": {
            "mobilenetv2_baseline": {
                "accuracy": round(float(mob_acc), 4),
                "balanced_accuracy": round(float(mob_bal_acc), 4),
                "macro_precision": round(float(mob_prec), 4),
                "macro_recall": round(float(mob_rec), 4),
                "macro_f1": round(float(mob_f1), 4),
                "confusion_matrix": mob_cm.tolist()
            },
            "efficientnetb0_advanced": {
                "accuracy": round(float(effnet_acc), 4),
                "balanced_accuracy": round(float(effnet_bal_acc), 4),
                "macro_precision": round(float(effnet_prec), 4),
                "macro_recall": round(float(effnet_rec), 4),
                "macro_f1": round(float(effnet_f1), 4),
                "confusion_matrix": effnet_cm.tolist()
            },
            "soft_voting_ensemble": {
                "accuracy": round(float(ens_acc), 4),
                "balanced_accuracy": round(float(ens_bal_acc), 4),
                "macro_precision": round(float(ens_prec), 4),
                "macro_recall": round(float(ens_rec), 4),
                "macro_f1": round(float(ens_f1), 4)
            }
        },
        "per_class": per_class_comparison,
        "classes": CLASS_LABELS
    }

    out_file = os.path.join(EVALUATION_DIR, "model_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"MobileNetV2 Accuracy:     {mob_acc * 100:.2f}%, F1: {mob_f1:.4f}, Balanced Acc: {mob_bal_acc * 100:.2f}%")
    print(f"EfficientNetB0 Accuracy:  {effnet_acc * 100:.2f}%, F1: {effnet_f1:.4f}, Balanced Acc: {effnet_bal_acc * 100:.2f}%")
    print(f"Ensemble Accuracy:        {ens_acc * 100:.2f}%, F1: {ens_f1:.4f}, Balanced Acc: {ens_bal_acc * 100:.2f}%")
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    run_milestone3_model_evaluation()
