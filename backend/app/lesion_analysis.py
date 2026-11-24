"""Classical-computer-vision lesion candidate ROI analysis.

This is an unsupervised visual aid, NOT a trained or clinically validated lesion
segmentation model. HAM10000 classification labels do not provide pixel masks.
"""
from typing import Dict, Tuple
import cv2
import numpy as np


def lesion_candidate_mask(image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("A valid BGR image is required")
    h, w = image_bgr.shape[:2]
    # Work at a bounded resolution to keep API latency predictable.
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else image_bgr.copy()
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    # Estimate background skin tone from a narrow border, then measure color distance.
    border = np.concatenate([lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]], axis=0)
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(lab.astype(np.float32) - background[None, None, :], axis=2)
    distance_u8 = np.uint8(np.clip(distance * (255.0 / max(float(np.percentile(distance, 99)), 1.0)), 0, 255))
    # Otsu threshold + morphology; restrict candidate to the central 90% to reduce edge artifacts.
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    margin_y, margin_x = int(mask.shape[0] * 0.05), int(mask.shape[1] * 0.05)
    mask[:margin_y, :] = 0; mask[-margin_y:, :] = 0
    mask[:, :margin_x] = 0; mask[:, -margin_x:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    if scale < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    area_ratio = float(np.count_nonzero(mask) / (h * w))
    valid = 0.005 <= area_ratio <= 0.80
    if not valid:
        # Empty mask is more honest than returning an arbitrary region when contrast is inadequate.
        mask[:] = 0
    return mask, {
        "method": "border_color_distance_otsu_morphology",
        "is_heuristic_candidate_roi": True,
        "is_supervised_segmentation": False,
        "candidate_detected": bool(valid),
        "candidate_area_percent": round(area_ratio * 100.0, 2) if valid else 0.0,
        "image_dimensions": [w, h],
        "limitations": "Heuristic color-contrast ROI only. Not a trained segmentation model, lesion boundary measurement, or clinical finding."
    }


def create_roi_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if np.any(mask):
        tint = np.zeros_like(image_bgr)
        tint[:, :, 1] = 220
        selected = mask > 0
        overlay[selected] = cv2.addWeighted(image_bgr[selected], 0.55, tint[selected], 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), max(1, image_bgr.shape[1] // 500))
    return overlay
