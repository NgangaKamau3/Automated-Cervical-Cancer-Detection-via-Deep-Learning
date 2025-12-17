"""
Test configuration and fixtures for pytest.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_image():
    """Create a dummy test image."""
    # Create random 224x224x3 image
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return image


@pytest.fixture(scope="session")
def test_image_bytes(test_image):
    """Convert test image to bytes."""
    from PIL import Image
    import io
    
    img = Image.fromarray(test_image)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture(scope="session")
def mock_model():
    """Create a simple mock model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model


@pytest.fixture(scope="session")
def temp_model_dir():
    """Create temporary directory for test models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def class_names():
    """Return standard class names."""
    return [
        "Dyskeratotic",
        "Koilocytotic",
        "Metaplastic",
        "Parabasal",
        "Superficial-Intermediate"
    ]


@pytest.fixture
def sample_metadata(class_names):
    """Sample model metadata."""
    return {
        "class_names": class_names,
        "num_classes": 5,
        "input_size": 224,
        "version": "1.0.0",
        "preprocessing": {
            "resize": 224,
            "normalization": "divide_by_255",
            "color_mode": "rgb"
        },
        "performance": {
            "test_accuracy": 0.9427
        }
    }
