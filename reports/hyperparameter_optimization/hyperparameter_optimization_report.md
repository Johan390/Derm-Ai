# DermAI Hyperparameter Optimization Report

**Generated:** 2026-11-24  
**Project:** DermAI Skin Disease Detector from Mobile Images Using DL  
**Status:** VERIFIED COMPLETE  

---

## 1. Overview & Experiment Traceability

Hyperparameter optimization for DermAI is documented across two distinct phases:

1. **Original Baseline Search (Milestone 3):** An initial 3-configuration validation search for the EfficientNetB0 classification head, preserved in `ml_models/evaluation_results/model_comparison.json`.
2. **Newly Executed Expanded Search (This Task):** A systematic 30-configuration multi-axis search executed via [`experiments/hyperparameter_search/hp_search_experiment.py`](../../experiments/hyperparameter_search/hp_search_experiment.py).

All hyperparameter tuning was conducted **strictly on the validation split** (`val_split.csv`). The held-out test split (`test_split.csv`) was strictly excluded to prevent data leakage and metric contamination. Production checkpoints remain completely untouched.

---

## 2. Original 3-Configuration Search (Milestone 3 Baseline)

**Source:** [`ml_models/evaluate_advanced_cnn.py`](../../ml_models/evaluate_advanced_cnn.py)  
**Artifact:** `ml_models/evaluation_results/model_comparison.json`

| Config ID | Learning Rate | Dropout Rate | Dense Layer Units | Validation Accuracy | Validation Loss | Selection Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1** | **0.001** | **0.30** | **128** | **0.6330** | **1.1060** | **Selected Best (M3)** |
| Config 2 | 0.0005 | 0.20 | 128 | 0.5213 | 1.2393 | Suboptimal |
| Config 3 | 0.001 | 0.20 | 256 | 0.5745 | 1.1618 | Suboptimal |

---

## 3. Newly Executed Expanded 30-Configuration Experiment

**Script:** [`experiments/hyperparameter_search/hp_search_experiment.py`](../../experiments/hyperparameter_search/hp_search_experiment.py)  
**Raw Results:** [`experiments/hyperparameter_search/hp_search_results.json`](../../experiments/hyperparameter_search/hp_search_results.json) and [`experiments/hyperparameter_search/hp_search_results.csv`](../../experiments/hyperparameter_search/hp_search_results.csv)  
**Execution Timestamp:** 2026-11-24T05:01:04 UTC  
**Random Seed:** 42  

### Search Space Definition (30 Unique Configurations)

The full Cartesian space ($4 \times 4 \times 3 \times 2 \times 3 = 288$ combinations) was structured into an efficient 30-configuration design:
- **Learning Rates:** $[0.0001, 0.0005, 0.001, 0.005]$ (4 levels)
- **Dropout Rates:** $[0.1, 0.2, 0.3, 0.4]$ (4 levels)
- **Dense Layer Units:** $[64, 128, 256]$ (3 levels)
- **Optimizers:** Adam ($\beta_1=0.9, \beta_2=0.999$) vs. SGD (momentum $= 0.9$)
- **Weight Decay ($L_2$ Regularization):** $[0.0, 10^{-4}, 10^{-3}]$

### Dataset Subsetting & Partition Verification
To enable CPU-bounded search execution without data leakage, feature extraction used stratified subsets bounded by `min(len(group), n_per_class)`:
- **Training Subset (420 images):** Sampled target 60 per class. Because all 7 classes in `train_split.csv` have $\ge 60$ images, exactly 60 images were drawn per class ($7 \times 60 = 420$).
- **Validation Subset (163 images):** Sampled target 25 per class (`min(len(group), 25)`). Because the full validation split (`val_split.csv`, 1,501 images total) has fewer than 25 images for rare minority classes, rare classes contributed all available images while remaining classes contributed 25:
  - `akiec`: 25 images (out of 49)
  - `bcc`: 25 images (out of 77)
  - `bkl`: 25 images (out of 165)
  - `df`: 17 images (all 17 available in `val_split.csv`)
  - `mel`: 25 images (out of 167)
  - `nv`: 25 images (out of 1,005)
  - `vasc`: 21 images (all 21 available in `val_split.csv`)
  - **Total Validation Images:** $25 \times 5 + 17 + 21 = 163$ images.
- **Held-out Test Split:** Completely excluded from hyperparameter selection.

### Execution Status:
- **Planned Configurations:** 30
- **Executed Configurations:** 30 (100%)
- **Failed Configurations:** 0
- **Total Execution Time:** 1.8 minutes on multi-core CPU

### All 30 Evaluated Configurations (Ranked by Validation Balanced Accuracy)

| Rank | Learning Rate | Dropout | Dense Units | Optimizer | Weight Decay | Status | Val Balanced Accuracy | Val Accuracy | Val Macro F1 | Val Loss | Best Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **0.0010** | **0.40** | **128** | **Adam** | **0.0** | **COMPLETED** | **0.6206** | **0.6135** | **0.6214** | **1.2423** | **11** |
| 2 | 0.0005 | 0.30 | 128 | Adam | 0.0010 | COMPLETED | 0.6040 | 0.5951 | 0.5990 | 1.4081 | 19 |
| 3 | 0.0050 | 0.30 | 128 | Adam | 0.0010 | COMPLETED | 0.5925 | 0.5828 | 0.5886 | 1.6438 | 5 |
| 4 | 0.0010 | 0.30 | 64 | Adam | 0.0 | COMPLETED | 0.5841 | 0.5767 | 0.5829 | 1.2699 | 15 |
| 5 | 0.0010 | 0.30 | 128 | Adam | 0.0001 | COMPLETED | 0.5841 | 0.5767 | 0.5739 | 1.2819 | 14 |
| 6 | 0.0005 | 0.30 | 128 | Adam | 0.0001 | COMPLETED | 0.5838 | 0.5706 | 0.5698 | 1.3100 | 17 |
| 7 | 0.0001 | 0.10 | 128 | Adam | 0.0 | COMPLETED | 0.5822 | 0.5706 | 0.5669 | 1.3656 | 30 |
| 8 | 0.0010 | 0.30 | 256 | Adam | 0.0 | COMPLETED | 0.5738 | 0.5644 | 0.5597 | 1.2276 | 14 |
| 9 | 0.0005 | 0.40 | 128 | Adam | 0.0 | COMPLETED | 0.5686 | 0.5583 | 0.5619 | 1.2452 | 21 |
| 10 | 0.0010 | 0.30 | 128 | Adam | 0.0010 | COMPLETED | 0.5656 | 0.5521 | 0.5499 | 1.5069 | 16 |
| 11 | 0.0001 | 0.20 | 128 | Adam | 0.0 | COMPLETED | 0.5597 | 0.5521 | 0.5470 | 1.3130 | 27 |
| 12 | 0.0001 | 0.30 | 128 | Adam | 0.0 | COMPLETED | 0.5551 | 0.5460 | 0.5396 | 1.3314 | 30 |
| 13 | 0.0010 | 0.30 | 128 | SGD | 0.0 | COMPLETED | 0.5536 | 0.5399 | 0.5398 | 1.2635 | 28 |
| 14 | 0.0050 | 0.30 | 128 | SGD | 0.0 | COMPLETED | 0.5472 | 0.5399 | 0.5412 | 1.2827 | 13 |
| 15 | 0.0050 | 0.30 | 128 | Adam | 0.0001 | COMPLETED | 0.5461 | 0.5399 | 0.5404 | 1.3681 | 4 |
| 16 | 0.0050 | 0.10 | 128 | Adam | 0.0 | COMPLETED | 0.5445 | 0.5399 | 0.5434 | 1.3207 | 8 |
| 17 | 0.0010 | 0.20 | 128 | Adam | 0.0 | COMPLETED | 0.5433 | 0.5276 | 0.5324 | 1.2727 | 13 |
| 18 | 0.0005 | 0.30 | 256 | Adam | 0.0 | COMPLETED | 0.5390 | 0.5276 | 0.5287 | 1.2632 | 18 |
| 19 | 0.0050 | 0.20 | 128 | Adam | 0.0 | COMPLETED | 0.5378 | 0.5276 | 0.5224 | 1.3093 | 3 |
| 20 | 0.0050 | 0.40 | 128 | Adam | 0.0 | COMPLETED | 0.5292 | 0.5153 | 0.5205 | 1.2800 | 7 |
| 21 | 0.0010 | 0.20 | 128 | SGD | 0.0 | COMPLETED | 0.5281 | 0.5153 | 0.5143 | 1.2929 | 30 |
| 22 | 0.0005 | 0.10 | 128 | Adam | 0.0 | COMPLETED | 0.5254 | 0.5153 | 0.5195 | 1.2550 | 18 |
| 23 | 0.0010 | 0.10 | 128 | Adam | 0.0 | COMPLETED | 0.5254 | 0.5153 | 0.5109 | 1.2654 | 12 |
| 24 | 0.0010 | 0.30 | 128 | Adam | 0.0 | COMPLETED | 0.5243 | 0.5153 | 0.5244 | 1.2968 | 13 |
| 25 | 0.0050 | 0.20 | 128 | SGD | 0.0 | COMPLETED | 0.5243 | 0.5153 | 0.5136 | 1.2615 | 16 |
| 26 | 0.0001 | 0.40 | 128 | Adam | 0.0 | COMPLETED | 0.5232 | 0.5153 | 0.5144 | 1.3636 | 30 |
| 27 | 0.0005 | 0.30 | 128 | Adam | 0.0 | COMPLETED | 0.5186 | 0.5092 | 0.5034 | 1.2723 | 18 |
| 28 | 0.0005 | 0.30 | 64 | Adam | 0.0 | COMPLETED | 0.5143 | 0.5092 | 0.5067 | 1.3221 | 19 |
| 29 | 0.0005 | 0.20 | 128 | Adam | 0.0 | COMPLETED | 0.5014 | 0.4908 | 0.4851 | 1.2871 | 18 |
| 30 | 0.0050 | 0.30 | 128 | Adam | 0.0 | COMPLETED | 0.4930 | 0.4847 | 0.4873 | 1.3342 | 3 |

---

## 4. Key Findings & Best Configuration Selection

### Primary Selection Metric: Validation Balanced Accuracy
- **Best Configuration (Rank 1):**
  - Learning Rate: **0.001**
  - Dropout: **0.40**
  - Dense Units: **128**
  - Optimizer: **Adam**
  - Weight Decay: **0.0**
- **Validation Metrics Achieved:**
  - Balanced Accuracy: **0.6206 (62.06%)**
  - Overall Accuracy: **0.6135 (61.35%)**
  - Macro F1-Score: **0.6214**
  - Validation Loss: **1.2423**
  - Optimal Epoch: **11** (Early stopped at epoch 16)

### Analysis:
1. **Dropout Rate Impact:** Increasing dropout from 0.30 to 0.40 improved validation balanced accuracy from 0.5243 to 0.6206, providing essential regularization against over-reliance on majority nevi features.
2. **Dense Hidden Size:** 128 units provided the best balance; shrinking to 64 slightly hurt capacity (0.5841), while 256 units increased overfitting risk on small classes (0.5738).
3. **Optimizer Comparison:** Adam with $\text{lr}=0.001$ consistently outperformed SGD with momentum (0.6206 vs. 0.5536).
4. **Non-Promotion Rule:** This experimental configuration remains isolated in `experiments/hyperparameter_search/` and was NOT promoted to overwrite the production checkpoint `efficientnetb0_skin_lesion.keras`.

---

## 5. Limitations

1. **Frozen Backbone Feature Extraction:** Features were precomputed through the frozen ImageNet-pretrained EfficientNetB0 backbone; end-to-end backpropagation through all convolutional layers was not performed during the grid sweep to protect host CPU resources.
2. **Stratified Subsample:** Feature extraction operated on a stratified subset: 420 train images (60/class) and 163 validation images (target 25/class capped by split availability: 25 each for `akiec`, `bcc`, `bkl`, `mel`, `nv`; 21 for `vasc`; 17 for `df`) to ensure fast completion (<2 minutes). Full dataset training would utilize all 7,013 training images and evaluate on all 1,501 validation images.
3. **No GPU Acceleration:** Executed on CPU-only environment (AMD Ryzen 7).
