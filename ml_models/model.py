"""
DermAI MobileNetV2 Model Definition & Fine-Tuning Manager

Milestone 2 - Stage 2 Implementation:
- Pre-trained MobileNetV2 backbone initialized with ImageNet weights
- Custom classification head:
    GlobalAveragePooling2D -> BatchNormalization -> Dropout(0.3) ->
    Dense(128, ReLU) -> Dropout(0.2) -> Dense(7, Softmax)
- Phase 1: Fully frozen backbone for fast feature adaptation
- Phase 2: Top-30 layer unfreezing with BatchNormalization layers frozen
"""

from typing import Tuple
import keras
from keras import layers
from keras.applications import MobileNetV2


def build_mobilenetv2_classifier(
    num_classes: int = 7,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate_1: float = 0.3,
    dropout_rate_2: float = 0.2,
    dense_units: int = 128
) -> Tuple[keras.Model, keras.Model]:
    """
    Builds the DermAI skin lesion classification model using transfer learning.
    
    Returns:
        full_model: End-to-end Keras model ready for Phase 1 training
        base_model: Pre-trained MobileNetV2 backbone (frozen)
    """
    # 1. Instantiate MobileNetV2 backbone with ImageNet weights (excluding original top)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # 2. Freeze the entire backbone for Phase 1
    base_model.trainable = False

    # 3. Define Input and build Custom Classification Head
    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # Pass through frozen backbone (training=False ensures BatchNorm running stats are untouched)
    x = base_model(inputs, training=False)
    
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate_1, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_rate_2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    full_model = keras.Model(inputs=inputs, outputs=outputs, name="DermAI_MobileNetV2")
    return full_model, base_model


def unfreeze_mobilenetv2_top_layers(
    model: keras.Model,
    base_model: keras.Model,
    num_layers_to_unfreeze: int = 30
) -> keras.Model:
    """
    Prepares the model for Phase 2 fine-tuning:
    - Unfreezes the top `num_layers_to_unfreeze` layers of MobileNetV2
    - Keeps lower feature-extraction layers frozen
    - Keeps all BatchNormalization layers frozen to protect pretrained batch statistics
    """
    # Enable training on base model
    base_model.trainable = True

    # Freeze lower layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # For the top layers, keep BatchNorm layers frozen
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return model

