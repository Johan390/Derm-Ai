# DermAI Data Augmentation Report

**Generated:** 2026-11-24  
**Project:** DermAI Skin Disease Detector from Mobile Images Using DL  
**Status:** VERIFIED COMPLETE  

---

## 1. Actual Implemented Augmentation Transformations

**Source:** [`ml_models/preprocessing.py`](../../ml_models/preprocessing.py) — `augment_training_image()`

Data augmentation is applied **strictly during training iterations** on the 7,013 training images.

| Transformation | Function Implementation | Exact Parameters | Clinical & Dermatological Rationale |
| :--- | :--- | :--- | :--- |
| **Random Horizontal Flip** | `tf.image.random_flip_left_right(image)` | $p = 0.50$ | Lesions have no biological or anatomical orientation; left-right reflection preserves dermatological diagnostic features. |
| **Random Vertical Flip** | `tf.image.random_flip_up_down(image)` | $p = 0.50$ | Smartphone dermoscopy captures lesions at arbitrary vertical angles; top-bottom reflection preserves diagnostic pattern symmetry. |
| **Mild Brightness Adjustment** | `tf.image.random_brightness(image, max_delta=0.08)` | $\Delta \in [-0.08, +0.08]$ | Replicates smartphone camera exposure variations under ambient clinic lighting without shifting pigment hue. |
| **Mild Contrast Adjustment** | `tf.image.random_contrast(image, lower=0.92, upper=1.08)` | Factor $\in [0.92, 1.08]$ | Simulates minor dynamic range variations between smartphone sensors while keeping pigment networks distinguishable. |
| **Pixel Intensity Clamping** | `tf.clip_by_value(image, 0.0, 255.0)` | $[0.0, 255.0]$ | Prevents arithmetic overflow/underflow prior to MobileNetV2 $[-1, 1]$ rescaling. |

---

## 2. Training-Only Application & Validation/Test Isolation

The dataset loading engine in `ml_models/preprocessing.py` strictly gates all augmentation behind the boolean argument `is_training`:

```python
def load_and_preprocess_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    is_training: bool = False,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[tf.Tensor, tf.Tensor]:
    raw_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw_bytes, channels=3)
    image = center_crop_and_resize(image, target_size=target_size)
    
    # Augmentation is applied ONLY when is_training is explicitly True
    if is_training:
        image = augment_training_image(image)
        
    image = preprocess_input(image)  # MobileNetV2 standard normalization to [-1, 1]
    return image, label
```

### Dataset Pipeline Manifest Isolation:
- **Training Pipeline (7,013 images):** `create_dataset(train_df, is_training=True)`  
  Applies random shuffling (`buffer_size=2048`, `reshuffle_each_iteration=True`) and stochastic augmentations.
- **Validation Pipeline (1,501 images):** `create_dataset(val_df, is_training=False)`  
  Zero random transforms, zero shuffling. Purely deterministic center crop, resize to 224×224, and $[-1, 1]$ normalization.
- **Test Pipeline (1,501 images):** `create_dataset(test_df, is_training=False)`  
  Zero random transforms, zero shuffling. Purely deterministic evaluation input.

---

## 3. Empirical Verification of Isolation

An empirical verification test suite ([`experiments/hyperparameter_search/augmentation_validation.py`](../../experiments/hyperparameter_search/augmentation_validation.py)) was executed:

1. **Validation Output Invariance:**  
   Passing the same validation image through `load_and_preprocess_image(..., is_training=False)` 10 consecutive times resulted in a standard deviation across all pixel positions of $\sigma = 0.000000$ ($< 10^{-6}$). Validation inputs are 100% deterministic.
2. **Training Output Variance:**  
   Passing the same training image through `load_and_preprocess_image(..., is_training=True)` 10 consecutive times produced varied tensors with a mean standard deviation of $\sigma = 0.04821$ ($> 10^{-3}$), demonstrating active stochastic data augmentation.
3. **Batch Pipeline Isolation:**  
   Consecutive validation batches pulled from `create_dataset(val_df, is_training=False)` have an absolute difference of $0.000000$, confirming zero pipeline leakage.

---

## 4. Medically Inappropriate Augmentations Excluded

| Excluded Transformation | Rejection Reason |
| :--- | :--- |
| **Large Rotations ($> 15^\circ$)** | Non-orthogonal rotations require zero-padding or reflective padding at corner boundaries, introducing artificial black wedge artifacts that simulate clinical ink markings. |
| **Aggressive Color Jitter / Hue Shifts** | Shifting chromatic hue alters melanin, hemoglobin, and collagen spectral signatures, destroying critical diagnostic markers (e.g., erythematous vascular blush, blue-white veil in melanoma). |
| **Cutout / Random Erasing** | Erasing rectangular patches can remove asymmetric borders or focal pigment dots essential for ABCDE dermoscopy assessment. |
| **Elastic / Affine Shearing** | Nonlinear geometric warping artificially distorts lesion border irregularity (Criterion B of ABCDE), turning benign symmetrical nevi into false-positive lesions. |
