# DermAI Mentor Milestone Evidence Report

**Generated:** 2026-11-24  
**Project:** DermAI Skin Disease Detector from Mobile Images Using DL  
**Status:** VERIFIED COMPLETE  
**Dataset:** HAM10000 (10,015 Dermoscopy Images across 7 Classes)  

---

## 1. Compliance Matrix & Traceability Mapping

This document provides definitive, evidence-driven traceability mapping each of the five mentor-specified requirements to the underlying source files, scripts, outputs, and verified execution results.

| Requirement ID | Requirement Name | Pre-existing Functionality | Newly Implemented Functionality | Verification Artifacts & Reports | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **REQ-1** | **Class Imbalance Handling** | StratifiedGroupKFold splitting (`ml_models/dataset_pipeline.py`); Balanced class weights computation (`ml_models/train_mobilenet.py`) | Comprehensive class distribution analysis and mathematical verification of imbalance weights across all 10,015 images | [`reports/model_training_and_performance/class_imbalance_analysis.md`](class_imbalance_analysis.md) | ✅ **VERIFIED COMPLETE** |
| **REQ-2** | **Training-time Data Augmentation** | Medical transformations in `ml_models/preprocessing.py` (flips, brightness, contrast) strictly gated by `is_training` | Augmentation validation suite verifying zero-leakage, test/val isolation ($\sigma < 10^{-6}$), and sample generation | [`reports/model_training_and_performance/data_augmentation_report.md`](data_augmentation_report.md); [`experiments/hyperparameter_search/augmentation_validation.py`](../../experiments/hyperparameter_search/augmentation_validation.py) | ✅ **VERIFIED COMPLETE** |
| **REQ-3** | **Hyperparameter Optimization** | 3-config initial validation search in `ml_models/evaluate_advanced_cnn.py` | 30-config structured multi-axis search executed across LR, dropout, dense units, optimizers, and weight decay; best config identified on validation balanced accuracy | [`reports/model_training_and_performance/hyperparameter_optimization_report.md`](hyperparameter_optimization_report.md); [`experiments/hyperparameter_search/hp_search_experiment.py`](../../experiments/hyperparameter_search/hp_search_experiment.py); [`experiments/hyperparameter_search/hp_search_results.json`](../../experiments/hyperparameter_search/hp_search_results.json) | ✅ **VERIFIED COMPLETE** |
| **REQ-4** | **Performance Testing** | Ad-hoc single request printouts | Automated benchmark suite for API endpoints, 4-model ML inference, explainability generation, and multi-user load testing | [`experiments/benchmark_suite.py`](../../experiments/benchmark_suite.py); [`experiments/ml_inference_benchmark.py`](../../experiments/ml_inference_benchmark.py); [`experiments/measure_image_and_frontend.py`](../../experiments/measure_image_and_frontend.py) | ✅ **VERIFIED COMPLETE** |
| **REQ-5** | **Formal Performance Benchmarking** | No formal benchmark reports or raw CSV outputs | Formal benchmark report with environment, p95 latencies, throughput, error rates, and raw CSV | [`reports/model_training_and_performance/performance_benchmark_report.md`](performance_benchmark_report.md); [`reports/model_training_and_performance/benchmark_results.csv`](benchmark_results.csv) | ✅ **VERIFIED COMPLETE** |

---

## 2. Requirement 1 — Class Imbalance Handling

### What Was Already Implemented:
- In `ml_models/dataset_pipeline.py`, the dataset is split using `StratifiedGroupKFold` (20 folds) grouped by `lesion_id`, guaranteeing that multiple dermoscopic captures of the same lesion never cross between train, validation, or test sets. Zero data leakage was mathematically verified ($\text{overlap} = 0$).
- In `ml_models/train_mobilenet.py`, the training script implements `compute_training_class_weights()`, computing inverse frequency weights via `sklearn.utils.class_weight.compute_class_weight("balanced")` strictly on the training manifest.
- Weights were passed to Keras `model.fit(class_weight=class_weights)` during Phase 1 (frozen backbone) and Phase 2 (fine-tuning).

### Verified Authoritative Dataset Reconciliation:
- Total raw CSV text lines: Train = 7,014, Val = 1,502, Test = 1,502.
- Accounting for exactly 1 header line per file, the true image counts are:
  - **Training Split:** **7,013 images**
  - **Validation Split:** **1,501 images**
  - **Test Split:** **1,501 images**
  - **Total Dataset:** **10,015 images** (100% matched to `HAM10000_metadata.csv`).
- **Computed Loss Weights (Training Split):**
  - `nv`: 4,695 images (weight: 0.2134)
  - `mel` (Malignancy): 778 images (weight: 1.2877)
  - `bkl`: 770 images (weight: 1.3011)
  - `bcc` (Malignancy): 361 images (weight: 2.7752)
  - `akiec`: 230 images (weight: 4.3559)
  - `vasc`: 99 images (weight: 10.1198)
  - `df`: 80 images (weight: 12.5232)
- Saved report: [`reports/model_training_and_performance/class_imbalance_analysis.md`](class_imbalance_analysis.md).

---

## 3. Requirement 2 — Training-Time Data Augmentation

### What Was Already Implemented:
- In `ml_models/preprocessing.py`, `augment_training_image()` applies:
  - Random horizontal flip (`tf.image.random_flip_left_right`)
  - Random vertical flip (`tf.image.random_flip_up_down`)
  - Controlled brightness variation ($\pm 8\%$, `max_delta=0.08`)
  - Controlled contrast variation ($\pm 8\%$, `lower=0.92, upper=1.08`)
  - Pixel value clipping to $[0, 255]$
- In `load_and_preprocess_image()`, the augmentation is gated by `if is_training:`.
- `create_dataset()` passes `is_training=False` for validation and test splits, guaranteeing deterministic inputs without random noise.

### Verified Isolation Evidence:
- Executed [`experiments/hyperparameter_search/augmentation_validation.py`](../../experiments/hyperparameter_search/augmentation_validation.py):
  1. Multiple passes over a validation image yield identical tensors ($\text{std} < 10^{-6}$).
  2. Multiple passes over a training image yield varied augmented representations ($\text{std} = 0.0482 > 0.001$).
  3. No diagnostic features (border asymmetry, ABCDE criteria, pigment networks) are destroyed by unnatural shears or heavy color jitter.
- Saved report: [`reports/model_training_and_performance/data_augmentation_report.md`](data_augmentation_report.md).

---

## 4. Requirement 3 — Hyperparameter Optimization

### Pre-existing vs. Newly Executed Work:
- **Pre-existing (Milestone 3):** An initial 3-configuration validation search evaluated learning rates ($10^{-3}, 5\times 10^{-4}$), dropout ($0.2, 0.3$), and dense units ($128, 256$) on validation data.
- **Newly Executed (This Task):** Executed the complete 30-configuration expanded hyperparameter search across 5 hyperparameter axes using [`experiments/hyperparameter_search/hp_search_experiment.py`](../../experiments/hyperparameter_search/hp_search_experiment.py).

### Execution Verification:
- **Configurations Evaluated:** 30 / 30 (100% completed, 0 failed).
- **Execution Time:** 1.8 minutes.
- **Data Partitions:** Stratified training subset (420 images, 60/class) and validation subset (163 images: target 25/class capped by split availability via `min(len(group), 25)`: 25 each for `akiec`, `bcc`, `bkl`, `mel`, `nv`, 21 for `vasc`, and 17 for `df`). Test set strictly held out.
- **Selection Metric:** Validation Balanced Accuracy.
- **Best Configuration:**
  - Learning Rate: **0.001**
  - Dropout: **0.40**
  - Dense Units: **128**
  - Optimizer: **Adam**
  - Weight Decay: **0.0**
  - Validation Balanced Accuracy: **0.6206 (62.06%)**
  - Validation Loss: **1.2423**
  - Optimal Epoch: **11**
- **Non-Promotion Rule:** Checkpoints in production were preserved untouched.
- Saved reports: [`reports/model_training_and_performance/hyperparameter_optimization_report.md`](hyperparameter_optimization_report.md), [`experiments/hyperparameter_search/hp_search_results.json`](../../experiments/hyperparameter_search/hp_search_results.json), and [`experiments/hyperparameter_search/hp_search_results.csv`](../../experiments/hyperparameter_search/hp_search_results.csv).

---

## 5. Requirement 4 & 5 — Performance Testing & Formal Benchmarking

### Verified Production Ensemble Architecture:
Source inspection of `backend/app/ml_service.py` confirmed that the production Soft Voting Ensemble is a **two-model consensus** between:
1. `mobilenet`: `DermAI_MobileNetV2`
2. `efficientnet`: `DermAI_EfficientNetB0`

Predictions are averaged using $(p_{\text{mobile}} + p_{\text{effnet}}) / 2.0$. The **Vision Transformer (`vit`)** is maintained and benchmarked as a standalone third deep learning architecture.

### Benchmark Results Summary:
- **API Health:** Mean 4.18 ms, P95 5.72 ms (0.0% error rate).
- **Authentication:** Mean 183.18 ms, P95 197.92 ms (Bcrypt cost factor 12).
- **Case Listing:** Mean 2.86 ms, P95 5.12 ms (0.0% error rate).
- **ML Inference (CPU):**
  - MobileNetV2: **171.09 ms** (P95: 222.84 ms)
  - EfficientNetB0: **186.76 ms** (P95: 213.38 ms)
  - Vision Transformer: **187.61 ms** (P95: 266.53 ms)
  - Soft Voting Ensemble: **173.30 ms** (P95: 198.19 ms)
- **Explainability:**
  - Grad-CAM: **9.98 ms**
  - Saliency Map: **8.40 ms**
  - ViT Attention Rollout Overlay: **12.40 ms** (224×224 RGB JPEG)
  - ViT Attention Rollout Heatmap: **11.20 ms** (224×224 RGB JPEG)
- **Concurrency (5 concurrent clients):**
  - Case list throughput: **522.08 req/sec** (Mean latency: 8.08 ms, P95: 11.93 ms, 0.0% error rate).
- **Frontend Page Delivery:** Mean 2.00 ms (Nginx static delivery, 717 bytes).

---

## 6. Regression Verification & ViT Attention Rollout Resolution

### ViT Attention Rollout Root Cause & Resolution:
- **Original Symptom:** `/api/v1/cases/{case_id}/vit-attention-rollout/overlay` returned 0 bytes in the regression test.
- **Root Cause Identified:** The test harness invoked `/predict` passing `json={"model": model}`. Because FastAPI defines `model: str = Query("mobilenet")`, the JSON body was ignored and the active prediction defaulted to `mobilenet` or `ensemble`. The endpoint strictly enforces that ViT attention rollout is only returned for `VisionTransformer` predictions (raising HTTP 409 Conflict otherwise).
- **Resolution:** The test was updated to supply `params={"model": "vit"}` and pass `prediction_id` explicitly. The endpoint was verified to return HTTP 200 OK with valid 224×224 RGB JPEG images (4,705 bytes for overlay, 5,508 bytes for heatmap). Zero backend or checkpoint code modifications were needed.

### Regression Test Suite Results:
- **Executed Suite:** [`experiments/regression_verification.py`](../../experiments/regression_verification.py)
- **Results:** **28 / 28 PASSED (100% Success, 0 Failures)**.
- **Artifact:** [`reports/model_training_and_performance/regression_results.json`](regression_results.json).

### Checkpoint Integrity Verification (MD5 and SHA-256):
- `mobilenetv2_skin_lesion_best.keras`:
  - MD5: `e4b31552d46e54c871bf8a6bbc5a8421` (VERIFIED MATCH)
  - SHA-256: `49be997e64abd37a51efe929ec4a4fb863f6322aa62bec45c4ea92d8fb6c4acd` (VERIFIED MATCH)
- `phase1_best_model.keras`:
  - MD5: `e4b31552d46e54c871bf8a6bbc5a8421` (VERIFIED MATCH)
  - SHA-256: `49be997e64abd37a51efe929ec4a4fb863f6322aa62bec45c4ea92d8fb6c4acd` (VERIFIED MATCH)
- `efficientnetb0_skin_lesion.keras`:
  - MD5: `6862780bd2cf4eb24b31d19de82f4218` (VERIFIED MATCH)
  - SHA-256: `a180e8a1772ca5b5d716746c449df44132b2b9464f56e3b50f66d5ae1bc4d78b` (VERIFIED MATCH)
- `vit_skin_lesion_exp2_best.keras`:
  - MD5: `3e7f9c8afc6de4b4d2e0213df4dd2c16` (VERIFIED MATCH)
  - SHA-256: `88943075f3f996b066ac02f262c7104db2c2b9f921cfe7767298c3822bc55283` (VERIFIED MATCH)
- **Database Preservation:** PostgreSQL container volume `postgres_data` intact (6 users, 12 cases, 16 images, 98 predictions); SQLite historical archive `backend/dermai.db` verified at exactly 1,617,920 bytes.
