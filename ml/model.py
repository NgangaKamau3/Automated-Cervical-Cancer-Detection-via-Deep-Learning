"""
Model Architecture Module
Defines the cervical cancer classification model using transfer learning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


def create_efficientnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    trainable_base_layers: int = 20
) -> keras.Model:
    """
    Create an EfficientNetB3-based model for cervical cell classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        trainable_base_layers: Number of top layers to make trainable
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_image')
    
    # Data augmentation layer (only active during training)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Load EfficientNetB3 pre-trained on ImageNet
    base_model = keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling=None
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Get output from base model
    x = base_model.output
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers with regularization
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNetB3_CervicalCancer')
    
    print(f"✅ Created EfficientNetB3 model")
    print(f"   • Input shape: {input_shape}")
    print(f"   • Output classes: {num_classes}")
    print(f"   • Dropout rate: {dropout_rate}")
    print(f"   • Base model trainable: {base_model.trainable}")
    
    return model, base_model


def create_resnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001
) -> keras.Model:
    """
    Create a ResNet50V2-based model for cervical cell classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_image')
    
    # Data augmentation layer
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Load ResNet50V2 pre-trained on ImageNet
    base_model = keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling=None
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Get output from base model
    x = base_model.output
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers with regularization
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet50V2_CervicalCancer')
    
    print(f"✅ Created ResNet50V2 model")
    print(f"   • Input shape: {input_shape}")
    print(f"   • Output classes: {num_classes}")
    print(f"   • Dropout rate: {dropout_rate}")
    
    return model, base_model


def unfreeze_base_model(base_model, num_layers: int = 20):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    
    Args:
        base_model: The base model to unfreeze
        num_layers: Number of top layers to unfreeze
    """
    base_model.trainable = True
    
    # Freeze all layers except the top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    total_count = len(base_model.layers)
    
    print(f"\n🔓 Unfroze top {num_layers} layers of base model")
    print(f"   • Trainable layers: {trainable_count}/{total_count}")


def get_model(
    architecture: str = 'efficientnet',
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    **kwargs
) -> Tuple[keras.Model, Optional[keras.Model]]:
    """
    Factory function to create models based on architecture name.
    
    Args:
        architecture: 'efficientnet' or 'resnet'
        input_shape: Input image shape
        num_classes: Number of output classes
        **kwargs: Additional arguments for model creation
        
    Returns:
        Tuple of (model, base_model)
    """
    architecture = architecture.lower()
    
    if architecture == 'efficientnet':
        return create_efficientnet_model(input_shape, num_classes, **kwargs)
    elif architecture == 'resnet':
        return create_resnet_model(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'efficientnet' or 'resnet'")


def main():
    """Test model creation."""
    print("=" * 60)
    print("  Model Architecture Test")
    print("=" * 60)
    print()
    
    # Test EfficientNet model
    print("Testing EfficientNetB3 model:")
    print("-" * 60)
    model, base = create_efficientnet_model()
    model.summary()
    
    print(f"\n📊 Model statistics:")
    print(f"   • Total parameters: {model.count_params():,}")
    print(f"   • Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    # Test with dummy input
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = model(dummy_input, training=False)
    print(f"\n🧪 Test prediction:")
    print(f"   • Input shape: {dummy_input.shape}")
    print(f"   • Output shape: {output.shape}")
    print(f"   • Output sum: {tf.reduce_sum(output).numpy():.6f} (should be ~1.0)")
    
    print("\n" + "=" * 60)
    print("✅ Model creation successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()
