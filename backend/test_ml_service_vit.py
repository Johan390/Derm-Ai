"""
DermAI - ML Service ViT & Multi-Model Inference Test Suite
Validates deep learning inference across Vision Transformer (ViT), MobileNetV2,
and EfficientNetB0 models using a self-contained synthetic dermoscopy fixture.
"""

import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ml_service import ml_service


def create_synthetic_lesion_file():
    """Generates a temporary synthetic dermoscopy image for standalone verification."""
    img = np.full((224, 224, 3), (180, 200, 240), dtype=np.uint8)
    cv2.circle(img, (112, 112), 45, (40, 60, 100), -1)
    cv2.ellipse(img, (112, 112), (55, 35), 30, 0, 360, (30, 45, 80), -1)

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    temp_file.close()
    cv2.imwrite(temp_path, img)
    return temp_path


def run_vit_and_multimodel_tests():
    print("\n" + "=" * 70)
    print("DERMAI - ML SERVICE ViT & MULTI-MODEL INFERENCE VERIFICATION")
    print("=" * 70)

    temp_image_path = create_synthetic_lesion_file()
    try:
        # 1. ViT Model Loading
        print("\n1. Testing ml_service._load_vit()...")
        model = ml_service._load_vit()
        assert model is not None, "Failed to load ViT model"
        print(f"   Loaded model: {model.name}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        assert model.output_shape[-1] == 7, f"Expected 7 classes, got {model.output_shape[-1]}"

        # 2. ViT Prediction
        print("\n2. Testing ml_service.predict_image with ViT...")
        res_vit = ml_service.predict_image(temp_image_path, model="vit")
        assert res_vit is not None
        assert "predicted_class" in res_vit
        assert "confidence_score" in res_vit
        assert "class_probabilities" in res_vit
        assert len(res_vit["class_probabilities"]) == 7
        prob_sum = sum(res_vit["class_probabilities"].values())
        assert 0.98 <= prob_sum <= 1.02, f"ViT probabilities sum out of range: {prob_sum}"
        print(f"   Model: {res_vit['model_name']}")
        print(f"   Predicted class: {res_vit['predicted_class']} ({res_vit['predicted_disease_name']})")
        print(f"   Confidence score: {res_vit['confidence_score']:.4f}")
        print(f"   Probabilities sum: {prob_sum:.4f}")

        # 3. MobileNetV2 Prediction
        print("\n3. Testing ml_service.predict_image with MobileNetV2...")
        res_mob = ml_service.predict_image(temp_image_path, model="mobilenet")
        assert res_mob is not None
        assert "predicted_class" in res_mob
        assert len(res_mob["class_probabilities"]) == 7
        print(f"   MobileNetV2 predicted: {res_mob['predicted_class']} ({res_mob['confidence_score']:.4f})")

        # 4. EfficientNetB0 Prediction
        print("\n4. Testing ml_service.predict_image with EfficientNetB0...")
        res_eff = ml_service.predict_image(temp_image_path, model="efficientnet")
        assert res_eff is not None
        assert "predicted_class" in res_eff
        assert len(res_eff["class_probabilities"]) == 7
        print(f"   EfficientNetB0 predicted: {res_eff['predicted_class']} ({res_eff['confidence_score']:.4f})")

        # 5. ViT Attention Map / Explainability
        print("\n5. Testing ml_service.generate_vit_attention_map...")
        heatmap, overlay = ml_service.generate_vit_attention_map(temp_image_path)
        assert heatmap is not None
        assert overlay is not None
        assert heatmap.shape in [(224, 224), (224, 224, 3)], f"Unexpected heatmap shape: {heatmap.shape}"
        assert overlay.shape == (224, 224, 3), f"Expected overlay shape (224, 224, 3), got {overlay.shape}"
        print(f"   Heatmap shape: {heatmap.shape}, Overlay shape: {overlay.shape}")

        print("\n" + "=" * 70)
        print("OVERALL RESULT: ALL ML SERVICE VERIFICATION CHECKS PASSED (5/5)")
        print("=" * 70 + "\n")
        return True

    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except OSError:
                pass


if __name__ == "__main__":
    success = run_vit_and_multimodel_tests()
    exit(0 if success else 1)
