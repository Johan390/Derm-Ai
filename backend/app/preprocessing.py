"""
DermAI Skin Image Preprocessing & Quality Assessment Engine

Milestone 1 Implementation:
- Image format, resolution, and size validation
- Image Quality Assessment (Blur, Brightness, Contrast, AI suitability)
- Geometric aspect-preserved center cropping for Region of Interest (ROI) preservation
- 224x224 standardization resizing with interpolation selection
- CLAHE (Contrast-Limited Adaptive Histogram Equalization) on L-channel in CIELAB color space
- Bilateral filtering for skin grain smoothing while preserving lesion boundary edges
- Diagnostic RGB channel statistics calculation (Model tensor normalization deferred to Milestone 2 DataLoader)
- Generation of high-fidelity preview thumbnail and preprocessed comparison image

IMPORTANT ARCHITECTURAL DISTINCTIONS:
1. Cropping vs. Lesion Localization:
   The center crop performed here is a GEOMETRIC aspect-square crop centered on the image frame.
   It does NOT perform automated lesion localization (bounding-box detection) or lesion boundary
   segmentation (e.g. U-Net/Mask R-CNN), which are scheduled for Milestone 2 and 3.
   Patients/clinicians are advised via clinical guidelines to center the lesion in the viewfinder.
2. Model Tensor Normalization:
   Standard 8-bit image formats (PNG/JPEG) store unsigned integers in [0, 255].
   True deep learning tensor normalization ((x - mean) / std) produces negative float32 values
   and is properly applied inside the PyTorch/TensorFlow DataLoader pipeline during training/inference.
   The preprocessed image saved on disk is an 8-bit visual asset for clinical side-by-side inspection.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, List
from .config import settings

def assess_image_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates mobile skin image quality indicators against configurable clinical thresholds:
    - Blur score using Laplacian operator variance (Var(ΔI))
    - Brightness score (mean grayscale pixel intensity)
    - Contrast score (standard deviation of intensity)
    - Minimum resolution compliance
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 1. Blur Detection (Laplacian Variance)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Brightness Check
    brightness_score = float(np.mean(gray))
    
    # 3. Contrast Check
    contrast_score = float(np.std(gray))

    notes: List[str] = []
    is_poor = False

    # Blur threshold evaluation
    if blur_score < settings.IQA_BLUR_THRESHOLD_POOR:
        notes.append("Noticeable motion blur or soft focus detected. Re-take recommended.")
        is_poor = True
    elif blur_score < settings.IQA_BLUR_THRESHOLD_ACCEPTABLE:
        notes.append("Moderate sharpness; acceptable for preliminary screening.")
    else:
        notes.append("Image is sharply focused.")

    # Lighting threshold evaluation
    if brightness_score < settings.IQA_BRIGHTNESS_MIN:
        notes.append("Image is underexposed (insufficient illumination/shadowed).")
        is_poor = True
    elif brightness_score > settings.IQA_BRIGHTNESS_MAX:
        notes.append("Image is overexposed (direct flash glare or washout).")
        is_poor = True

    # Contrast threshold evaluation
    if contrast_score < settings.IQA_CONTRAST_MIN:
        notes.append("Low contrast between skin lesion and surrounding tissue.")
        if not is_poor and blur_score < settings.IQA_BLUR_THRESHOLD_ACCEPTABLE:
            is_poor = True

    # Resolution threshold evaluation
    if width < settings.IQA_MIN_DIMENSION or height < settings.IQA_MIN_DIMENSION:
        notes.append(f"Resolution ({width}x{height}) is below minimum requirement ({settings.IQA_MIN_DIMENSION}x{settings.IQA_MIN_DIMENSION}).")
        is_poor = True

    if is_poor:
        quality_status = "Poor Quality"
        is_suitable = False
    elif blur_score < settings.IQA_BLUR_THRESHOLD_ACCEPTABLE or contrast_score < (settings.IQA_CONTRAST_MIN + 10):
        quality_status = "Acceptable Quality"
        is_suitable = True
    else:
        quality_status = "Good Quality"
        is_suitable = True

    return {
        "blur_score": round(blur_score, 2),
        "brightness_score": round(brightness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "quality_status": quality_status,
        "is_suitable_for_ai": is_suitable,
        "notes": notes
    }

def preprocess_skin_image(
    original_filepath: str,
    output_filename: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Executes the DermAI Milestone 1 preprocessing pipeline:
    1. Validation & Reading from preserved original
    2. Image Quality Assessment
    3. Aspect-Preserved Geometric Center Cropping (ROI preservation)
    4. Adaptive Resizing (INTER_AREA for decimation, INTER_LINEAR for interpolation)
    5. Bilateral Filtering for epidermal smoothing with boundary preservation
    6. CLAHE (Contrast-Limited Adaptive Histogram Equalization) in CIELAB color space
    7. Diagnostic RGB channel statistics calculation
    8. Exports preview thumbnail and preprocessed comparison image
    """
    image_bgr = cv2.imread(original_filepath)
    if image_bgr is None:
        raise ValueError(f"Failed to read image file: {original_filepath}")

    orig_h, orig_w = image_bgr.shape[:2]

    # Perform Image Quality Assessment
    quality_metrics = assess_image_quality(image_bgr)
    quality_metrics["width"] = orig_w
    quality_metrics["height"] = orig_h
    quality_metrics["file_size_bytes"] = os.path.getsize(original_filepath)

    # 1. Aspect Ratio Centering & Square Cropping
    min_dim = min(orig_h, orig_w)
    start_y = (orig_h - min_dim) // 2
    start_x = (orig_w - min_dim) // 2
    cropped = image_bgr[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # 2. Resizing with Adaptive Interpolation
    if min_dim >= target_size[0] and min_dim >= target_size[1]:
        interp = cv2.INTER_AREA  # Superior downsampling
    else:
        interp = cv2.INTER_LINEAR  # Smooth upsampling if input is smaller than 224x224
    resized = cv2.resize(cropped, target_size, interpolation=interp)

    # 3. Noise Reduction (Bilateral Filter preserves lesion borders while smoothing skin grain)
    denoised = cv2.bilateralFilter(resized, d=5, sigmaColor=50, sigmaSpace=50)

    # 4. Contrast Enhancement via CLAHE on L-channel in CIELAB Color Space
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    preprocessed_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 5. Diagnostic Normalization Statistics (RGB representation)
    rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_channels = [float(round(np.mean(rgb[:, :, c]), 4)) for c in range(3)]
    std_channels = [float(round(np.std(rgb[:, :, c]), 4)) for c in range(3)]

    # 6. Save Preprocessed Comparison File
    preprocessed_path = os.path.join(settings.PREPROCESSED_DIR, output_filename)
    cv2.imwrite(preprocessed_path, preprocessed_bgr)

    # 7. Generate Preview Thumbnail
    preview_path = os.path.join(settings.PREVIEW_DIR, output_filename)
    preview_w = 400
    preview_h = max(1, int(400 * (orig_h / orig_w)))
    preview_img = cv2.resize(image_bgr, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(preview_path, preview_img)

    return {
        "original_dimensions": [orig_w, orig_h],
        "preprocessed_dimensions": [target_size[0], target_size[1]],
        "cropping_method": "geometric_center_crop",
        "lesion_localization_applied": False,
        "quality_metrics": quality_metrics,
        "rgb_channel_mean": mean_channels,
        "rgb_channel_std": std_channels,
        "preprocessed_filepath": preprocessed_path,
        "preview_filepath": preview_path
    }
