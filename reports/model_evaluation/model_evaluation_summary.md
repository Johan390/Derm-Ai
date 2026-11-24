# DermAI Model Evaluation & Architectural Comparison Summary

**Project:** DermAI — Clinical Skin Lesion Quality & Preprocessing Platform  
**Dataset:** HAM10000 Dermoscopy Benchmark (10,015 images across 7 diagnostic categories)  
**Taxonomy:** Actinic Keratoses (`akiec`), Basal Cell Carcinoma (`bcc`), Benign Keratosis-like Lesions (`bkl`), Dermatofibroma (`df`), Melanoma (`mel`), Melanocytic Nevi (`nv`), Vascular Lesions (`vasc`).

---

## 1. Deep Learning Architecture Comparison

The DermAI classification platform evaluates four distinct architectural configurations:
1. **MobileNetV2 Baseline:** Lightweight inverted residual CNN optimized for low-latency clinical edge screening.
2. **EfficientNetB0:** Compound-scaled convolutional neural network balancing depth, width, and resolution.
3. **Vision Transformer (ViT-16):** Multi-head self-attention architecture (16x16 patch projection, 196 tokens, embedding dimension 768) providing global receptive fields.
4. **Soft Voting Ensemble:** Consensus decision-support pipeline computing unweighted probability average across MobileNetV2 and EfficientNetB0:
   $$P_{\text{ensemble}}(y=c \mid x) = \frac{1}{2} \left[ P_{\text{mobilenet}}(y=c \mid x) + P_{\text{effnet}}(y=c \mid x) \right]$$

---

## 2. Quantitative Performance Summary

### Comparative Evaluation on Stratified Test Cohort (N = 265)
Source: [`model_comparison.json`](model_comparison.json) and [`vit_cohort_evaluation.json`](vit_cohort_evaluation.json)

| Architecture | Top-1 Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC (OvR Macro) | Inference Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 Baseline** | 58.87% | 59.78% | 0.6555 | 0.5978 | 0.5985 | 0.8521 | ~171 ms |
| **EfficientNetB0** | 56.98% | 58.82% | 0.5777 | 0.5882 | 0.5770 | 0.8410 | ~186 ms |
| **Vision Transformer (ViT-16)** | 60.75% | 61.23% | 0.6120 | 0.6123 | 0.6017 | **0.8978** | ~187 ms |
| **Soft Voting Ensemble** | **63.02%** | **65.22%** | **0.6606** | **0.6522** | **0.6440** | **0.9015** | ~173 ms |

### Full Validation Cohort Evaluation (N = 1,501 images)
- **Vision Transformer Standalone:**
  - Overall Accuracy: **74.28%**
  - Balanced Accuracy: **62.82%**
  - Macro F1-Score: **0.5817**
  - Area Under ROC Curve (OvR Macro): **0.8978**
  - Per-image CPU Latency: **4.96 ms** (batched pipeline)

---

## 3. Per-Class Clinical Sensitivity & Specificity

High sensitivity for high-risk malignant lesions (Melanoma `mel` and Basal Cell Carcinoma `bcc`) is safety-critical for dermatological triage.

| Diagnostic Class | Clinical Risk Tier | MobileNetV2 Recall | EfficientNetB0 Recall | ViT-16 Recall | Ensemble Recall | Critical Safety Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Melanoma (`mel`)** | **High Risk** | 0.3333 | 0.6667 | **0.6889** | **0.6444** | ViT and Ensemble double the detection rate of early malignant melanoma. |
| **Basal Cell Carcinoma (`bcc`)** | **High Risk** | 0.5111 | 0.4667 | **0.5333** | **0.5556** | High-precision screening across ulcerated and pigmented BCC variants. |
| **Actinic Keratosis (`akiec`)** | **Moderate Risk** | 0.8000 | 0.6000 | 0.6222 | 0.7333 | Pre-cancerous lesion screening with balanced specificity. |
| **Benign Keratosis (`bkl`)** | Low Risk | 0.3778 | 0.4222 | 0.2889 | 0.4222 | Differentiated against seborrheic keratoses and solar lentigines. |
| **Dermatofibroma (`df`)** | Low Risk | 0.5000 | 0.6111 | 0.4444 | 0.6111 | Rare fibrous histiocytoma handled via class-weighted training. |
| **Melanocytic Nevus (`nv`)** | Low Risk | 0.8444 | 0.5778 | 0.8444 | 0.8444 | Common benign mole class; high specificity prevents over-biopsy. |
| **Vascular Lesion (`vasc`)** | Low Risk | **1.0000** | 0.9444 | **1.0000** | **1.0000** | Angiomas and pyogenic granulomas recognized near perfectly. |

---

## 4. Evaluation Visualizations & Diagnostics

The following diagnostic plots are preserved in this directory:
- [`confusion_matrix.png`](confusion_matrix.png) & [`confusion_matrix_normalized.png`](confusion_matrix_normalized.png): Detailed class confusion showing primary misclassifications between benign nevi and early melanocytic lesions.
- [`roc_curves.png`](roc_curves.png): One-vs-Rest ROC curves per class with macro and micro-average AUC trajectories.
- [`per_class_metrics.png`](per_class_metrics.png): Visual bar comparison of precision, recall, and F1 across all seven classes.
- [`accuracy_curves.png`](accuracy_curves.png) & [`loss_curves.png`](loss_curves.png): Multi-epoch training convergence profiles showing early stopping at epoch 11–16.

---

## 5. Architectural Recommendations & Conclusions

1. **Soft Voting Ensemble for Clinical Triage:** The MobileNetV2 + EfficientNetB0 ensemble achieves the highest overall accuracy (63.02%) and balanced accuracy (65.22%), successfully mitigating individual model failure modes.
2. **Vision Transformer for Complex Lesion Morphologies:** ViT achieves the highest single-model ROC-AUC (0.8978) and highest standalone melanoma recall (68.89%), making it uniquely valuable as an expert secondary opinion architecture.
3. **Clinical Decision-Support Boundary:** Model probabilities represent uncalibrated neural activations and must be utilized strictly as diagnostic decision-support tools by licensed clinical practitioners, never as standalone autonomous diagnostic authorities.
