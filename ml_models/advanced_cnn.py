"""
DermAI Advanced CNN: EfficientNet-B0 Skin Lesion Classifier

Milestone 3 Implementation:
- Transfer learning with EfficientNetB0 backbone (ImageNet weights)
- Custom clinical classification head matching 7-class HAM10000 taxonomy
- Compatible preprocessing (square center crop, 224x224, standard normalization)
- Compatible input (224, 224, 3) and output (7, softmax)
- Quantitative evaluation and model persistence
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import EfficientNetB0


def build_efficientnetb0_classifier(
    num_classes: int = 7,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate_1: float = 0.3,
    dropout_rate_2: float = 0.2,
    dense_units: int = 128
) -> Tuple[keras.Model, keras.Model]:
    """
    Builds the DermAI skin lesion classification model using EfficientNetB0 backbone.
    
    Returns:
        full_model: Keras model with EfficientNetB0 backbone and custom 7-class head.
        base_model: Pre-trained EfficientNetB0 feature extraction backbone.
    """
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate_1, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_rate_2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    full_model = keras.Model(inputs=inputs, outputs=outputs, name="DermAI_EfficientNetB0")
    return full_model, base_model


def get_efficientnet_target_conv_layer(base_model: keras.Model) -> keras.layers.Layer:
    """Returns the final convolutional feature layer for Grad-CAM in EfficientNetB0."""
    return base_model.get_layer("top_activation")

