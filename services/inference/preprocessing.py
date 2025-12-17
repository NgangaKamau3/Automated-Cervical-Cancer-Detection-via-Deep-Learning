"""
Image Preprocessing Module
Handles image preprocessing for inference.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing for cervical cell images."""
    
    def __init__(self, target_size: int = 224):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (height and width)
        """
        self.target_size = target_size
    
    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image from bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Decode image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return self.preprocess_array(image_array)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image from bytes: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def preprocess_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess image array.
        
        Args:
            image_array: Numpy array of the image
            
        Returns:
            Preprocessed image array
        """
        try:
            # Resize image
            if image_array.shape[:2] != (self.target_size, self.target_size):
                image_array = cv2.resize(
                    image_array,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Ensure correct dtype and range
            image_array = image_array.astype(np.float32)
            
            # Normalize to [0, 1]
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess image array: {e}")
            raise ValueError(f"Image preprocessing failed: {e}")
    
    def preprocess_batch(self, image_bytes_list: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            image_bytes_list: List of raw image bytes
            
        Returns:
            Batch of preprocessed images as numpy array
        """
        preprocessed_images = []
        
        for image_bytes in image_bytes_list:
            preprocessed_image = self.preprocess_from_bytes(image_bytes)
            preprocessed_images.append(preprocessed_image)
        
        # Stack into batch
        batch = np.stack(preprocessed_images, axis=0)
        
        return batch
    
    def validate_image(self, image_bytes: bytes) -> Tuple[bool, str]:
        """
        Validate image data.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check size
            if len(image_bytes) == 0:
                return False, "Empty image data"
            
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                return False, "Image too large (max 10MB)"
            
            # Try to open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check dimensions
            width, height = image.size
            if width < 32 or height < 32:
                return False, f"Image too small ({width}x{height}), minimum is 32x32"
            
            if width > 4096 or height > 4096:
                return False, f"Image too large ({width}x{height}), maximum is 4096x4096"
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                return False, f"Unsupported image format: {image.format}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
