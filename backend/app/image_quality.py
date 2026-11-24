"""
DermAI Independent Image Quality Assessment (IQA) Module

This module provides explainable, deterministic, and measurable image-quality indicators
for dermatological mobile photographs before inference, in accordance with the mentor
specification (pages 6, 13-14, 29, 31).

Evaluated Quality Dimensions:
1. Blur Level (Laplacian Operator Variance - Var(ΔI)): Quantifies high-frequency spatial edge definition.
2. Brightness Assessment (Mean Grayscale Intensity): Detects illumination suitability.
3. Lighting Aberrations:
   - Underexposure Detection (Mean intensity < threshold): Insufficient lighting, shadowing.
   - Overexposure Detection (Mean intensity > threshold or highlight clipping): Specular reflection, flash washout.
4. Contrast Level (Intensity Standard Deviation - σ): Evaluates dynamic range and lesion-to-skin contrast.
5. Spatial Resolution Compliance: Verifies minimum pixel width and height against minimum spatial sampling bounds.
6. File Format & Integrity: Validates binary consistency against allowed extensions.

Quality Categories:
- GOOD: Optimal sharpness, balanced illumination, strong dynamic range. Suitable for high-confidence AI analysis.
- ACCEPTABLE: Moderate sharpness or illumination; diagnostic features remain discernible. AI analysis permitted with caution.
- POOR: Severe motion blur, extreme under/overexposure, or deficient resolution. Re-capture strongly recommended.
"""

import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image

from .config import settings


class ImageQualityAssessmentEngine:
    """Independent, configurable image quality assessment engine."""

    def __init__(
        self,
        blur_poor: float = 20.0,
        blur_acceptable: float = 40.0,
        brightness_min: float = 40.0,
        brightness_max: float = 225.0,
        contrast_min: float = 12.0,
        min_dimension: int = 150,
        max_file_size_mb: int = 15
    ):
        self.blur_poor = blur_poor
        self.blur_acceptable = blur_acceptable
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.contrast_min = contrast_min
        self.min_dimension = min_dimension
        self.max_file_size_mb = max_file_size_mb

    def evaluate_array(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates an in-memory BGR numpy array against explainable visual indicators.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image array is empty or None.")

        height, width = image_bgr.shape[:2]
        if image_bgr.ndim == 2:
            gray = image_bgr
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
            gray = image_bgr[:, :, 0]
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Aspect ratio heuristic
        min_dim = max(min(width, height), 1)
        aspect_ratio = max(width, height) / min_dim
        is_extreme_aspect = aspect_ratio > 3.0

        # 1. Blur Detection (Laplacian Variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # 2. Brightness Assessment (Mean grayscale intensity in [0, 255])
        brightness_score = float(np.mean(gray))

        # 3. Contrast Assessment (Standard deviation of intensity in [0, 255])
        contrast_score = float(np.std(gray))

        # 4. Highlight & Shadow Clipping (Percentage of pixels at boundaries)
        shadow_clip_pct = float(np.sum(gray < 10) / gray.size * 100.0)
        highlight_clip_pct = float(np.sum(gray > 245) / gray.size * 100.0)

        # 5. Diagnostic Notes and Anomaly Flags
        notes: List[str] = []
        is_underexposed = brightness_score < self.brightness_min or shadow_clip_pct > 25.0
        is_overexposed = brightness_score > self.brightness_max or highlight_clip_pct > 20.0
        is_blurred = blur_score < self.blur_poor
        is_low_contrast = contrast_score < self.contrast_min
        is_sub_resolution = width < self.min_dimension or height < self.min_dimension

        is_poor = False

        if is_blurred:
            notes.append(f"Significant motion blur detected (Laplacian score: {blur_score:.1f} < {self.blur_poor:.1f}). Focus lesion carefully.")
            is_poor = True
        elif blur_score < self.blur_acceptable:
            notes.append(f"Moderate focus definition (Laplacian score: {blur_score:.1f}). Acceptable for preliminary evaluation.")
        else:
            notes.append(f"Optimal sharp focus (Laplacian score: {blur_score:.1f}).")

        if is_underexposed:
            notes.append(f"Image is underexposed or shadowed (Mean brightness: {brightness_score:.1f} < {self.brightness_min:.1f}). Increase ambient lighting.")
            is_poor = True
        elif is_overexposed:
            notes.append(f"Image is overexposed with glare or flash washout (Mean brightness: {brightness_score:.1f} > {self.brightness_max:.1f}). Avoid direct flash reflections.")
            is_poor = True
        else:
            notes.append(f"Balanced illumination (Mean brightness: {brightness_score:.1f}).")

        if is_low_contrast:
            notes.append(f"Low intensity contrast (Standard deviation: {contrast_score:.1f} < {self.contrast_min:.1f}). Pigment delineation may be weak.")
            if not is_poor and blur_score < self.blur_acceptable:
                is_poor = True
        else:
            notes.append(f"Good lesion-to-skin contrast (Standard deviation: {contrast_score:.1f}).")

        if is_sub_resolution:
            notes.append(f"Spatial resolution ({width}x{height}) is below minimum requirement ({self.min_dimension}x{self.min_dimension}).")
            is_poor = True

        if is_extreme_aspect:
            notes.append(f"Unusual aspect ratio ({aspect_ratio:.1f}:1). Significant lesion peripheral area may be lost during square center-crop.")
            is_poor = True

        # 6. Quality Tier Classification
        if is_poor:
            quality_status = "POOR"
            is_suitable_for_ai = False
            recommendation = "Image quality is suboptimal for reliable AI evaluation. Re-taking with balanced lighting and sharp focus is recommended."
        elif blur_score < self.blur_acceptable or contrast_score < (self.contrast_min + 10.0) or shadow_clip_pct > 15.0 or highlight_clip_pct > 10.0:
            quality_status = "ACCEPTABLE"
            is_suitable_for_ai = True
            recommendation = "Image quality meets minimum clinical criteria. AI inference is supported with standard clinical caveats."
        else:
            quality_status = "GOOD"
            is_suitable_for_ai = True
            recommendation = "Image quality is optimal. Fully suitable for multi-model AI decision support."

        return {
            "quality_status": quality_status,
            "is_suitable_for_ai": is_suitable_for_ai,
            "blur_score": round(blur_score, 2),
            "brightness_score": round(brightness_score, 2),
            "contrast_score": round(contrast_score, 2),
            "shadow_clip_pct": round(shadow_clip_pct, 2),
            "highlight_clip_pct": round(highlight_clip_pct, 2),
            "width": int(width),
            "height": int(height),
            "aspect_ratio": round(aspect_ratio, 2),
            "is_blurred": bool(is_blurred),
            "is_underexposed": bool(is_underexposed),
            "is_overexposed": bool(is_overexposed),
            "is_low_contrast": bool(is_low_contrast),
            "is_sub_resolution": bool(is_sub_resolution),
            "is_extreme_aspect": bool(is_extreme_aspect),
            "recommendation": recommendation,
            "notes": notes
        }

    def evaluate_file(self, filepath: str) -> Dict[str, Any]:
        """
        Evaluates a file on disk including file size, format, and image quality.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        file_size = os.path.getsize(filepath)
        max_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(f"File size ({file_size / (1024*1024):.1f}MB) exceeds limit of {self.max_file_size_mb}MB")

        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise ValueError(f"Could not decode image at {filepath}")

        res = self.evaluate_array(img_bgr)
        res["file_size_bytes"] = file_size
        res["file_size_mb"] = round(file_size / (1024 * 1024), 2)
        return res


# Global singleton using application settings
iqa_engine = ImageQualityAssessmentEngine(
    blur_poor=settings.IQA_BLUR_THRESHOLD_POOR,
    blur_acceptable=settings.IQA_BLUR_THRESHOLD_ACCEPTABLE,
    brightness_min=settings.IQA_BRIGHTNESS_MIN,
    brightness_max=settings.IQA_BRIGHTNESS_MAX,
    contrast_min=settings.IQA_CONTRAST_MIN,
    min_dimension=settings.IQA_MIN_DIMENSION,
    max_file_size_mb=settings.MAX_FILE_SIZE_MB
)
