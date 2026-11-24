"""
DermAI Vision Transformer (ViT) Architecture for Skin Lesion Classification

Milestone 3 Implementation:
- Patch Extraction and Embedding (16x16 non-overlapping patches, 196 total tokens)
- Positional Encoding for 2D spatial arrangement preservation
- Multi-Head Self-Attention (MHSA) Transformer Encoder blocks
- Multi-Layer Perceptron (MLP) blocks with GELU activations and residual connections
- Compatible 7-class taxonomy output head for HAM10000 diagnostics
- Evaluation & operational status documentation (Honest disclosure of training limits on CPU)
"""

from typing import Tuple, Optional
import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="dermai")
class PatchExtractor(layers.Layer):
    """
    Extracts square non-overlapping patches from input RGB images.
    For an image of 224x224 and patch size 16:
    Produces (224/16)^2 = 14x14 = 196 patches of dimension 16*16*3 = 768.
    """
    def __init__(self, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        num_patches = (224 // self.patch_size) * (224 // self.patch_size)
        return tf.reshape(patches, [batch_size, num_patches, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@keras.saving.register_keras_serializable(package="dermai")
class PatchEncoder(layers.Layer):
    """
    Linearly projects patches into a latent vector space and adds learnable positional embeddings.
    """
    def __init__(self, num_patches: int = 196, projection_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim, name="patch_projection")
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim, name="position_embedding"
        )

    def build(self, input_shape):
        # Explicitly build both child layers before marking this layer built.
        # This preserves the existing Dense and Embedding weights.
        self.projection.build(input_shape)
        self.position_embedding.build((self.num_patches,))
        super().build(input_shape)

    def call(self, patch: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


def build_vit_classifier(
    num_classes: int = 7,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    patch_size: int = 16,
    projection_dim: int = 128,
    num_heads: int = 4,
    transformer_layers: int = 3,
    mlp_dim: int = 256,
    dropout_rate: float = 0.1
) -> keras.Model:
    """
    Constructs a lightweight Vision Transformer (ViT) classifier for dermatological screening.
    
    Inputs:
        num_classes: 7 diagnostic lesion classes (akiec, bcc, bkl, df, mel, nv, vasc)
        input_shape: (224, 224, 3) RGB images
        patch_size: 16x16 patch resolution (yielding 196 patches)
        projection_dim: Latent representation size (128)
        num_heads: Number of attention heads (4)
        transformer_layers: Number of Transformer blocks (3)
        mlp_dim: Feed-forward dimension (256)
    
    Returns:
        keras.Model: Configured Vision Transformer classifier
    """
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # 1. Patch extraction and projection with positional embeddings
    patches = PatchExtractor(patch_size=patch_size, name="patch_extractor")(inputs)
    encoded = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, name="patch_encoder")(patches)

    # 2. Transformer Encoder Blocks
    for i in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(encoded)
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate,
            name=f"mhsa_{i}"
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add(name=f"add1_{i}")([attn_output, encoded])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x2)
        # MLP Block
        mlp = layers.Dense(mlp_dim, activation=tf.nn.gelu, name=f"mlp_dense1_{i}")(x3)
        mlp = layers.Dropout(dropout_rate, name=f"mlp_drop1_{i}")(mlp)
        mlp = layers.Dense(projection_dim, name=f"mlp_dense2_{i}")(mlp)
        mlp = layers.Dropout(dropout_rate, name=f"mlp_drop2_{i}")(mlp)
        # Skip connection 2
        encoded = layers.Add(name=f"add2_{i}")([mlp, x2])

    # 3. Representation & Clinical Classification Head
    representation = layers.LayerNormalization(epsilon=1e-6, name="head_ln")(encoded)
    pooled = layers.GlobalAveragePooling1D(name="global_avg_pool")(representation)
    features = layers.Dropout(0.2, name="head_dropout")(pooled)
    logits = layers.Dense(num_classes, activation="softmax", name="predictions")(features)

    model = keras.Model(inputs=inputs, outputs=logits, name="DermAI_VisionTransformer")
    return model

