"""
Tests for image preprocessing module.
"""

import pytest
import numpy as np
from PIL import Image
import io

from services.inference.preprocessing import ImagePreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = ImagePreprocessor(target_size=224)
    assert preprocessor.target_size == 224


def test_preprocess_from_bytes(test_image_bytes):
    """Test preprocessing from image bytes."""
    preprocessor = ImagePreprocessor(target_size=224)
    result = preprocessor.preprocess_from_bytes(test_image_bytes)
    
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_array(test_image):
    """Test preprocessing from numpy array."""
    preprocessor = ImagePreprocessor(target_size=224)
    result = preprocessor.preprocess_array(test_image)
    
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32


def test_preprocess_resize():
    """Test that images are resized correctly."""
    preprocessor = ImagePreprocessor(target_size=224)
    
    # Create image of different size
    large_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    result = preprocessor.preprocess_array(large_image)
    
    assert result.shape == (224, 224, 3)


def test_preprocess_batch(test_image_bytes):
    """Test batch preprocessing."""
    preprocessor = ImagePreprocessor(target_size=224)
    
    # Create batch of 3 images
    batch_bytes = [test_image_bytes] * 3
    result = preprocessor.preprocess_batch(batch_bytes)
    
    assert result.shape == (3, 224, 224, 3)
    assert result.dtype == np.float32


def test_validate_image_valid(test_image_bytes):
    """Test validation with valid image."""
    preprocessor = ImagePreprocessor(target_size=224)
    is_valid, error_msg = preprocessor.validate_image(test_image_bytes)
    
    assert is_valid is True
    assert error_msg == ""


def test_validate_image_empty():
    """Test validation with empty data."""
    preprocessor = ImagePreprocessor(target_size=224)
    is_valid, error_msg = preprocessor.validate_image(b"")
    
    assert is_valid is False
    assert "empty" in error_msg.lower()


def test_validate_image_too_large():
    """Test validation with oversized image."""
    preprocessor = ImagePreprocessor(target_size=224)
    
    # Create very large byte array (>10MB)
    large_bytes = b"0" * (11 * 1024 * 1024)
    is_valid, error_msg = preprocessor.validate_image(large_bytes)
    
    assert is_valid is False
    assert "large" in error_msg.lower()


def test_validate_image_invalid_format():
    """Test validation with invalid format."""
    preprocessor = ImagePreprocessor(target_size=224)
    
    # Not an image
    invalid_bytes = b"This is not an image"
    is_valid, error_msg = preprocessor.validate_image(invalid_bytes)
    
    assert is_valid is False


def test_preprocess_grayscale_to_rgb():
    """Test that grayscale images are converted to RGB."""
    preprocessor = ImagePreprocessor(target_size=224)
    
    # Create grayscale image
    gray_img = Image.new('L', (256, 256), color=128)
    img_bytes = io.BytesIO()
    gray_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = preprocessor.preprocess_from_bytes(img_bytes.getvalue())
    
    assert result.shape == (224, 224, 3)
