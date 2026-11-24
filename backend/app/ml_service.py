"""DermAI inference service: MobileNetV2, EfficientNetB0, ViT and soft voting.

The original MobileNetV2 path and preprocessing remain the default for backwards
compatibility. Alternative models are loaded lazily and never overwrite the baseline.
"""
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_tf = None
_keras = None
_preprocess_input = None


def _get_tf_and_keras():
    global _tf, _keras, _preprocess_input
    if _tf is None:
        import tensorflow as tf
        import keras
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        _tf, _keras, _preprocess_input = tf, keras, preprocess_input
    return _tf, _keras, _preprocess_input


class ModelServiceError(Exception):
    pass


class ModelNotFoundError(ModelServiceError):
    pass


class ImageProcessingError(ModelServiceError):
    pass


class SkinLesionClassifierService:
    _instance: Optional["SkinLesionClassifierService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            saved_dir = os.path.join(self.base_dir, "ml_models", "saved_models")
            self.model_path = model_path or os.path.join(saved_dir, "mobilenetv2_skin_lesion_best.keras")
            self.metadata_path = metadata_path or os.path.join(saved_dir, "model_metadata.json")
            self.efficientnet_path = os.path.join(saved_dir, "efficientnetb0_skin_lesion.keras")
            self.vit_path = os.path.join(saved_dir, "vit_skin_lesion_exp2_best.keras")
            self.model = None
            self.efficientnet_model = None
            self.vit_model = None
            self.model_name = "DermAI_MobileNetV2"
            self.class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            self.class_names = {
                "akiec": "Actinic Keratosis", "bcc": "Basal Cell Carcinoma",
                "bkl": "Benign Keratosis", "df": "Dermatofibroma", "mel": "Melanoma",
                "nv": "Melanocytic Nevus", "vasc": "Vascular Lesion"
            }
            self.risk_levels = {
                "mel": "High Risk", "bcc": "High Risk", "akiec": "Moderate Risk",
                "nv": "Low Risk", "bkl": "Low Risk", "df": "Low Risk", "vasc": "Low Risk"
            }
            self.disclaimer = (
                "AI screening decision-support only; not a definitive clinical diagnosis. "
                "Softmax scores are model-relative mathematical scores, not calibrated medical probabilities. "
                "Categorical risk levels reflect epidemiological histopathology severity tiers of the predicted lesion class, "
                "not confirmed biopsy diagnoses or individualized patient clinical risk. "
                "In-person clinical evaluation by a qualified medical professional is required."
            )
            self._load_lock = threading.RLock()
            self._inference_lock = threading.RLock()
            self._load_metadata()
            # Load large TensorFlow artifacts on first inference, not during API import.
            self._initialized = True

    def _load_metadata(self):
        if os.path.isfile(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.class_labels = meta.get("class_labels", self.class_labels)
                self.class_names = meta.get("class_names", self.class_names)
                self.risk_levels = meta.get("risk_levels", self.risk_levels)
                self.model_name = meta.get("model_name", self.model_name)
            except Exception as exc:
                print(f"[ML Service] Metadata warning: {exc}")

    def _validate_model(self, model, name):
        if tuple(model.input_shape[1:]) != (224, 224, 3):
            raise ModelServiceError(f"{name} input shape must be (224,224,3); got {model.input_shape}")
        if int(model.output_shape[-1]) != len(self.class_labels):
            raise ModelServiceError(f"{name} must output {len(self.class_labels)} classes; got {model.output_shape}")

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            raise ModelNotFoundError(f"Trained MobileNetV2 model not found: {self.model_path}")
        with self._load_lock:
            if self.model is not None:
                return
            _, keras, _ = _get_tf_and_keras()
            try:
                self.model = keras.models.load_model(self.model_path, compile=False)
                self._validate_model(self.model, "MobileNetV2")
                self.model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
                print("[ML Service] MobileNetV2 loaded and warmed up.")
            except ModelServiceError:
                self.model = None
                raise
            except Exception as exc:
                self.model = None
                raise ModelServiceError(f"Could not load MobileNetV2 checkpoint: {exc}") from exc

    def _load_efficientnet(self):
        with self._load_lock:
            if self.efficientnet_model is not None:
                return self.efficientnet_model
            if not os.path.isfile(self.efficientnet_path):
                raise ModelNotFoundError(f"EfficientNetB0 checkpoint not found: {self.efficientnet_path}")
            _, keras, _ = _get_tf_and_keras()
            try:
                model = keras.models.load_model(self.efficientnet_path, compile=False)
                self._validate_model(model, "EfficientNetB0")
                self.efficientnet_model = model
                return model
            except Exception as exc:
                raise ModelServiceError(f"Could not load EfficientNetB0 checkpoint: {exc}") from exc

    def _load_vit(self):
        with self._load_lock:
            if self.vit_model is not None:
                return self.vit_model
            if not os.path.isfile(self.vit_path):
                raise ModelNotFoundError(f"Vision Transformer checkpoint not found: {self.vit_path}")
            _, keras, _ = _get_tf_and_keras()
            # Register the project's custom PatchExtractor/PatchEncoder classes before deserialization.
            if self.base_dir not in sys.path:
                sys.path.insert(0, self.base_dir)
            try:
                from ml_models import vision_transformer  # noqa: F401
                custom_objects = {
                    "PatchExtractor": getattr(vision_transformer, "PatchExtractor", None),
                    "PatchEncoder": getattr(vision_transformer, "PatchEncoder", None)
                }
                custom_objects = {k: v for k, v in custom_objects.items() if v is not None}
                model = keras.models.load_model(self.vit_path, custom_objects=custom_objects, compile=False)
                self._validate_model(model, "Vision Transformer")
                self.vit_model = model
                return model
            except Exception as exc:
                raise ModelServiceError(
                    "Could not load the trained ViT checkpoint. Check Keras compatibility and "
                    f"custom layer registration: {exc}"
                ) from exc

    def is_model_ready(self) -> bool:
        return self.model is not None

    def _read_rgb(self, image_path: str):
        if not os.path.isfile(image_path):
            raise ImageProcessingError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ImageProcessingError(f"Could not decode image: {image_path}")
        h, w = image.shape[:2]
        side = min(h, w)
        y, x = (h - side) // 2, (w - side) // 2
        crop = image[y:y + side, x:x + side]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32), image

    def preprocess_image(self, image_path: str) -> np.ndarray:
        rgb, _ = self._read_rgb(image_path)
        _, _, preprocess_input = _get_tf_and_keras()
        return np.expand_dims(preprocess_input(rgb.copy()), axis=0)

    def _format_result(self, probs, model_name, component_probabilities=None):
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0:
            raise ModelServiceError("Model returned an invalid probability vector.")
        probs = probs / total
        idx = int(np.argmax(probs))
        class_probs = {label: float(round(float(p), 6)) for label, p in zip(self.class_labels, probs)}
        sorted_items = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
        top3 = sorted_items[:3]
        top5 = sorted_items[:5]
        return {
            "predicted_class": self.class_labels[idx],
            "predicted_disease_name": self.class_names.get(self.class_labels[idx], self.class_labels[idx]),
            "confidence_score": float(round(float(probs[idx]), 6)),
            "risk_level": self.risk_levels.get(self.class_labels[idx], "Unknown Risk"),
            "class_probabilities": class_probs,
            "top_three_predictions": [
                {"class_label": label, "disease_name": self.class_names.get(label, label),
                 "probability": probability, "risk_level": self.risk_levels.get(label, "Unknown Risk")}
                for label, probability in top3
            ],
            "top_five_predictions": [
                {"class_label": label, "disease_name": self.class_names.get(label, label),
                 "probability": probability, "risk_level": self.risk_levels.get(label, "Unknown Risk")}
                for label, probability in top5
            ],
            "ranked_predictions": [
                {"class_label": label, "disease_name": self.class_names.get(label, label),
                 "probability": probability, "risk_level": self.risk_levels.get(label, "Unknown Risk")}
                for label, probability in sorted_items
            ],
            "model_name": model_name,
            "model_components": component_probabilities or {},
            "disclaimer": self.disclaimer
        }

    def predict_image(self, image_path: str, model: str = "mobilenet") -> Dict[str, Any]:
        model = (model or "mobilenet").lower().replace("-", "_")
        rgb, _ = self._read_rgb(image_path)
        tf, _, preprocess_input = _get_tf_and_keras()
        batch = np.expand_dims(preprocess_input(rgb.copy()), axis=0)
        with self._inference_lock:
            if model in ("mobilenet", "mobilenetv2", "baseline"):
                if not self.is_model_ready():
                    self._load_model()
                probs = self.model(batch, training=False).numpy()[0]
                name = "DermAI_MobileNetV2"
                components = None
            elif model in ("vit", "vision_transformer"):
                net = self._load_vit()
                probs = net(batch, training=False).numpy()[0]
                name = "DermAI_VisionTransformer"
                components = None
            elif model in ("efficientnet", "efficientnetb0"):
                net = self._load_efficientnet()
                # EfficientNetB0's Keras application includes its own input rescaling.
                probs = net(np.expand_dims(rgb, 0), training=False).numpy()[0]
                name = "DermAI_EfficientNetB0"
                components = None
            elif model in ("ensemble", "soft_voting"):
                if not self.is_model_ready():
                    self._load_model()
                effnet = self._load_efficientnet()
                p_mobile = self.model(batch, training=False).numpy()[0]
                p_effnet = effnet(np.expand_dims(rgb, 0), training=False).numpy()[0]
                probs = (np.asarray(p_mobile) + np.asarray(p_effnet)) / 2.0
                name = "DermAI_SoftVoting_Ensemble_MobileNetV2_EfficientNetB0"
                components = {"mobilenetv2": [float(x) for x in p_mobile],
                              "efficientnetb0": [float(x) for x in p_effnet]}
            else:
                raise ValueError("model must be one of: mobilenet, efficientnet, vit, ensemble")
        return self._format_result(probs, name, components)

    def _compute_cnn_cam(
        self,
        image_path: str,
        model_type: str,
        class_index: Optional[int] = None
    ):
        """
        Calculates raw spatial Grad-CAM activation map for a constituent CNN model
        (MobileNetV2 with 'out_relu' or EfficientNetB0 with 'top_activation').
        Returns (cam_224_float, rgb).
        """
        tf, keras, _ = _get_tf_and_keras()
        rgb, _ = self._read_rgb(image_path)

        if model_type == "mobilenet":
            if not self.is_model_ready():
                self._load_model()
            model = self.model
            backbone_name = "mobilenetv2_1.00_224"
            target_layer_name = "out_relu"
            batch = self.preprocess_image(image_path)
        elif model_type == "efficientnet":
            model = self._load_efficientnet()
            backbone_name = "efficientnetb0"
            target_layer_name = "top_activation"
            batch = np.expand_dims(rgb.astype(np.float32), axis=0)
        else:
            raise ValueError(f"Unsupported CNN model_type: {model_type}")

        backbone = model.get_layer(backbone_name)
        target_layer = backbone.get_layer(target_layer_name)

        feature_model = keras.Model(
            backbone.input,
            [target_layer.output, backbone.output]
        )

        feature_shape = tuple(backbone.output_shape[1:])
        feature_input = keras.Input(shape=feature_shape)

        x = feature_input
        for layer in model.layers[2:]:
            x = layer(x, training=False)

        head_model = keras.Model(feature_input, x)

        with tf.GradientTape() as tape:
            conv_features, backbone_features = feature_model(
                batch, training=False
            )
            tape.watch(conv_features)

            predictions = head_model(
                backbone_features, training=False
            )

            target_index = int(
                class_index
                if class_index is not None
                else tf.argmax(predictions[0])
            )

            if target_index < 0 or target_index >= len(self.class_labels):
                raise ValueError("Invalid Grad-CAM class index.")

            score = predictions[:, target_index]

        gradients = tape.gradient(score, conv_features)
        if gradients is None:
            raise ModelServiceError("Grad-CAM gradients could not be calculated.")

        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_features, axis=-1)[0]
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())

        heat = cam.numpy()
        heat = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_LINEAR)
        return heat, rgb

    def _compute_cnn_cam_plus_plus(
        self,
        image_path: str,
        model_type: str,
        class_index: Optional[int] = None
    ):
        """
        Calculates raw spatial Grad-CAM++ activation map for a constituent CNN model.
        Returns (cam_224_float, rgb).
        """
        tf, keras, _ = _get_tf_and_keras()
        rgb, _ = self._read_rgb(image_path)

        if model_type == "mobilenet":
            if not self.is_model_ready():
                self._load_model()
            model = self.model
            backbone_name = "mobilenetv2_1.00_224"
            target_layer_name = "out_relu"
            batch = self.preprocess_image(image_path)
        elif model_type == "efficientnet":
            model = self._load_efficientnet()
            backbone_name = "efficientnetb0"
            target_layer_name = "top_activation"
            batch = np.expand_dims(rgb.astype(np.float32), axis=0)
        else:
            raise ValueError(f"Unsupported CNN model_type: {model_type}")

        backbone = model.get_layer(backbone_name)
        target_layer = backbone.get_layer(target_layer_name)

        feature_model = keras.Model(
            backbone.input,
            [target_layer.output, backbone.output]
        )

        feature_shape = tuple(backbone.output_shape[1:])
        feature_input = keras.Input(shape=feature_shape)
        x = feature_input
        for layer in model.layers[2:]:
            x = layer(x, training=False)
        head_model = keras.Model(feature_input, x)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_feats, backbone_feats = feature_model(batch, training=False)
                    tape3.watch(conv_feats)
                    tape2.watch(conv_feats)
                    tape1.watch(conv_feats)

                    preds = head_model(backbone_feats, training=False)
                    target_idx = int(
                        class_index
                        if class_index is not None
                        else tf.argmax(preds[0])
                    )
                    if target_idx < 0 or target_idx >= len(self.class_labels):
                        raise ValueError("Invalid Grad-CAM++ class index.")
                    score = preds[:, target_idx]

                first_grad = tape3.gradient(score, conv_feats)
            second_grad = tape2.gradient(first_grad, conv_feats)
        third_grad = tape1.gradient(second_grad, conv_feats)

        if first_grad is None or second_grad is None:
            raise ModelServiceError("Higher-order gradients could not be computed for Grad-CAM++.")

        conv_first_grad = tf.maximum(first_grad, 0.0)
        denom = 2.0 * second_grad + conv_feats * third_grad
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
        alpha = second_grad / denom

        weights = tf.reduce_sum(alpha * conv_first_grad, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_feats, axis=-1)[0]
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())

        heat = cam.numpy()
        heat = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_LINEAR)
        return heat, rgb

    def generate_gradcam(
        self,
        image_path: str,
        class_index: Optional[int] = None,
        model_type: str = "mobilenet",
        constituent: Optional[str] = None
    ):
        """
        Generate genuine Grad-CAM for MobileNetV2, EfficientNetB0, or Soft Voting Ensemble.
        For Ensemble, combines constituent CNN feature attributions (MobileNetV2 'out_relu'
        and EfficientNetB0 'top_activation') weighted by soft-voting consensus, or returns
        the selected constituent explanation.
        """
        try:
            if model_type in ("mobilenet", "efficientnet"):
                heat, rgb = self._compute_cnn_cam(image_path, model_type, class_index)
            elif model_type in ("ensemble", "soft_voting"):
                if class_index is None:
                    ens_res = self.predict_image(image_path, model="ensemble")
                    class_index = self.class_labels.index(ens_res["predicted_class"])

                if constituent == "mobilenet":
                    heat, rgb = self._compute_cnn_cam(image_path, "mobilenet", class_index)
                elif constituent == "efficientnet":
                    heat, rgb = self._compute_cnn_cam(image_path, "efficientnet", class_index)
                else:
                    heat_mob, rgb = self._compute_cnn_cam(image_path, "mobilenet", class_index)
                    heat_eff, _ = self._compute_cnn_cam(image_path, "efficientnet", class_index)
                    heat_blend = 0.5 * heat_mob + 0.5 * heat_eff
                    h_min = float(np.min(heat_blend))
                    h_max = float(np.max(heat_blend))
                    heat = (heat_blend - h_min) / (h_max - h_min + 1e-8)
            else:
                raise ValueError(
                    f"Grad-CAM supports 'mobilenet', 'efficientnet', and 'ensemble'. Got '{model_type}'."
                )

            heat_u8 = np.uint8(np.clip(heat * 255, 0, 255))
            heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
                0.60,
                heatmap,
                0.40,
                0
            )
            return overlay, heatmap
        except ModelServiceError:
            raise
        except Exception as exc:
            raise ModelServiceError(f"{model_type} Grad-CAM generation failed: {exc}") from exc

    def generate_gradcam_plus_plus(
        self,
        image_path: str,
        class_index: Optional[int] = None,
        model_type: str = "mobilenet",
        constituent: Optional[str] = None
    ):
        """
        Generate Grad-CAM++ (Chattopadhyay et al., 2018) for MobileNetV2, EfficientNetB0, or Soft Voting Ensemble.
        Uses higher-order gradients for finer spatial weighting of lesion features.
        """
        try:
            if model_type in ("mobilenet", "efficientnet"):
                heat, rgb = self._compute_cnn_cam_plus_plus(image_path, model_type, class_index)
            elif model_type in ("ensemble", "soft_voting"):
                if class_index is None:
                    ens_res = self.predict_image(image_path, model="ensemble")
                    class_index = self.class_labels.index(ens_res["predicted_class"])

                if constituent == "mobilenet":
                    heat, rgb = self._compute_cnn_cam_plus_plus(image_path, "mobilenet", class_index)
                elif constituent == "efficientnet":
                    heat, rgb = self._compute_cnn_cam_plus_plus(image_path, "efficientnet", class_index)
                else:
                    heat_mob, rgb = self._compute_cnn_cam_plus_plus(image_path, "mobilenet", class_index)
                    heat_eff, _ = self._compute_cnn_cam_plus_plus(image_path, "efficientnet", class_index)
                    heat_blend = 0.5 * heat_mob + 0.5 * heat_eff
                    h_min = float(np.min(heat_blend))
                    h_max = float(np.max(heat_blend))
                    heat = (heat_blend - h_min) / (h_max - h_min + 1e-8)
            else:
                raise ValueError(
                    f"Grad-CAM++ supports 'mobilenet', 'efficientnet', and 'ensemble'. Got '{model_type}'."
                )

            heat_u8 = np.uint8(np.clip(heat * 255, 0, 255))
            heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
                0.60,
                heatmap,
                0.40,
                0
            )
            return overlay, heatmap
        except ModelServiceError:
            raise
        except Exception as exc:
            raise ModelServiceError(f"{model_type} Grad-CAM++ generation failed: {exc}") from exc

    def generate_saliency_map(
        self,
        image_path: str,
        class_index: Optional[int] = None,
        model_type: str = "mobilenet"
    ):
        """
        Generate pixel-level gradient saliency map (Simonyan et al., 2013).
        Computes the gradient of the predicted class score with respect to input pixels.
        Supported for MobileNetV2, EfficientNetB0, Vision Transformer, and Soft Voting Ensemble.
        """
        tf, _, preprocess_input = _get_tf_and_keras()
        rgb, _ = self._read_rgb(image_path)

        try:
            if model_type in ("mobilenet", "mobilenetv2"):
                if not self.is_model_ready():
                    self._load_model()
                model = self.model
                input_tensor = np.expand_dims(preprocess_input(rgb.copy()), axis=0)
                tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(tensor)
                    preds = model(tensor, training=False)
                    target_idx = int(
                        class_index
                        if class_index is not None
                        else tf.argmax(preds[0])
                    )
                    if target_idx < 0 or target_idx >= len(self.class_labels):
                        raise ValueError("Invalid class index for saliency map.")
                    score = preds[:, target_idx]
                grads = tape.gradient(score, tensor)

            elif model_type in ("efficientnet", "efficientnetb0"):
                model = self._load_efficientnet()
                input_tensor = np.expand_dims(rgb.astype(np.float32), axis=0)
                tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(tensor)
                    preds = model(tensor, training=False)
                    target_idx = int(
                        class_index
                        if class_index is not None
                        else tf.argmax(preds[0])
                    )
                    if target_idx < 0 or target_idx >= len(self.class_labels):
                        raise ValueError("Invalid class index for saliency map.")
                    score = preds[:, target_idx]
                grads = tape.gradient(score, tensor)

            elif model_type in ("vit", "vision_transformer"):
                model = self._load_vit()
                input_tensor = np.expand_dims(preprocess_input(rgb.copy()), axis=0)
                tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(tensor)
                    preds = model(tensor, training=False)
                    target_idx = int(
                        class_index
                        if class_index is not None
                        else tf.argmax(preds[0])
                    )
                    if target_idx < 0 or target_idx >= len(self.class_labels):
                        raise ValueError("Invalid class index for saliency map.")
                    score = preds[:, target_idx]
                grads = tape.gradient(score, tensor)

            elif model_type in ("ensemble", "soft_voting"):
                if not self.is_model_ready():
                    self._load_model()
                effnet = self._load_efficientnet()
                x_tensor = tf.convert_to_tensor(rgb.astype(np.float32))
                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)
                    batch_x = tf.expand_dims(x_tensor, 0)
                    x_mob = (batch_x / 127.5) - 1.0
                    x_eff = batch_x

                    p_mob = self.model(x_mob, training=False)
                    p_eff = effnet(x_eff, training=False)
                    p_ens = (p_mob + p_eff) / 2.0

                    target_idx = int(
                        class_index
                        if class_index is not None
                        else tf.argmax(p_ens[0])
                    )
                    if target_idx < 0 or target_idx >= len(self.class_labels):
                        raise ValueError("Invalid class index for saliency map.")
                    score = p_ens[:, target_idx]
                grads = tape.gradient(score, x_tensor)

            else:
                raise ValueError(
                    f"Saliency maps support 'mobilenet', 'efficientnet', 'vit', and 'ensemble'. Got '{model_type}'."
                )

            if grads is None:
                raise ModelServiceError("Input gradients could not be calculated.")

            # Max absolute gradient across RGB color channels
            saliency = tf.reduce_max(tf.abs(grads), axis=-1)
            if len(saliency.shape) == 3:
                saliency = saliency[0]
            sal_min = tf.reduce_min(saliency)
            sal_max = tf.reduce_max(saliency)
            saliency_norm = (saliency - sal_min) / (sal_max - sal_min + 1e-8)

            sal_np = saliency_norm.numpy()
            sal_resized = cv2.resize(sal_np, (224, 224), interpolation=cv2.INTER_LINEAR)
            sal_u8 = np.uint8(np.clip(sal_resized * 255, 0, 255))
            heatmap = cv2.applyColorMap(sal_u8, cv2.COLORMAP_HOT)

            overlay = cv2.addWeighted(
                cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
                0.60,
                heatmap,
                0.40,
                0
            )
            return overlay, heatmap
        except ModelServiceError:
            raise
        except Exception as exc:
            raise ModelServiceError(f"{model_type} Saliency map generation failed: {exc}") from exc

    def generate_vit_attention_rollout(self, image_path: str):
        """
        Calculates Attention Rollout (Abnar & Zuidema, 2020) specifically for the DermAI Vision Transformer.
        Recursively accumulates multi-head self-attention matrices across all 3 Transformer
        blocks, accounting for residual skip connections and global token pooling across the 196 (14x14) patches.
        Produces (overlay, heatmap) scaled to (224, 224, 3).
        """
        tf, keras, preprocess_input = _get_tf_and_keras()
        rgb, _ = self._read_rgb(image_path)
        batch = np.expand_dims(preprocess_input(rgb.copy()), axis=0)

        vit = self._load_vit()

        try:
            input_layer = vit.input
            extractor = vit.get_layer("patch_extractor")
            encoder = vit.get_layer("patch_encoder")

            x = encoder(extractor(batch))

            matrices = []
            for i in range(3):
                ln1 = vit.get_layer(f"ln1_{i}")
                mhsa = vit.get_layer(f"mhsa_{i}")
                add1 = vit.get_layer(f"add1_{i}")
                ln2 = vit.get_layer(f"ln2_{i}")
                mlp_d1 = vit.get_layer(f"mlp_dense1_{i}")
                mlp_d2 = vit.get_layer(f"mlp_dense2_{i}")
                add2 = vit.get_layer(f"add2_{i}")

                norm_x = ln1(x)
                attn_out, attn_weights = mhsa(
                    norm_x, norm_x, return_attention_scores=True, training=False
                )
                # attn_weights shape: (1, num_heads=4, seq_len=196, seq_len=196)
                # Average across attention heads: shape (196, 196)
                attn_mean = tf.reduce_mean(attn_weights[0], axis=0).numpy()
                matrices.append(attn_mean)

                x2 = add1([attn_out, x])
                norm_x2 = ln2(x2)
                mlp_out = mlp_d2(mlp_d1(norm_x2))
                x = add2([mlp_out, x2])

            # Compute Attention Rollout: account for residual connections (W = 0.5 * A + 0.5 * I)
            identity = np.eye(196, dtype=np.float32)
            joint_attention = None
            for A in matrices:
                # Add residual connection
                A_res = A + identity
                # Row normalization so rows sum to 1
                A_norm = A_res / (A_res.sum(axis=-1, keepdims=True) + 1e-8)
                if joint_attention is None:
                    joint_attention = A_norm
                else:
                    joint_attention = np.matmul(A_norm, joint_attention)

            # Global average pooling across query tokens to derive input patch relevance
            patch_relevance = joint_attention.mean(axis=0)  # (196,)
            patch_grid = patch_relevance.reshape((14, 14))

            # Min-Max normalization to [0, 1]
            p_min = patch_grid.min()
            p_max = patch_grid.max()
            patch_norm = (patch_grid - p_min) / (p_max - p_min + 1e-8)

            # Resize 14x14 patch relevance to 224x224 input image dimensions
            attention_224 = cv2.resize(
                patch_norm.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC
            )
            attention_u8 = np.uint8(np.clip(attention_224 * 255, 0, 255))
            heatmap = cv2.applyColorMap(attention_u8, cv2.COLORMAP_VIRIDIS)

            overlay = cv2.addWeighted(
                cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
                0.60,
                heatmap,
                0.40,
                0
            )
            return overlay, heatmap

        except Exception as exc:
            raise ModelServiceError(f"ViT Attention Rollout extraction failed: {exc}") from exc

    def generate_vit_attention_map(self, image_path: str):
        """
        Alias for generate_vit_attention_rollout for backward compatibility.
        """
        return self.generate_vit_attention_rollout(image_path)


ml_service = SkinLesionClassifierService()
