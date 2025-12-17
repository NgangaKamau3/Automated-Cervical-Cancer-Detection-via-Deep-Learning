"""
Model Loader Module
Handles model loading and caching for the inference service.
"""

import os
import json
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton model loader class."""
    
    _instance = None
    _model = None
    _metadata = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: Optional[str] = None) -> Tuple[tf.keras.Model, Dict]:
        """
        Load the trained model and its metadata.
        
        Args:
            model_path: Path to the model file. If None, uses default locations.
            
        Returns:
            Tuple of (model, metadata)
        """
        if self._model is not None and model_path == self._model_path:
            logger.info("Using cached model")
            return self._model, self._metadata
        
        # Determine model path
        if model_path is None:
            model_path = self._find_model_path()
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Load the model
            self._model = tf.keras.models.load_model(model_path)
            self._model_path = model_path
            
            # Load metadata
            self._metadata = self._load_metadata(model_path)
            
            # Warm up the model
            self._warmup_model()
            
            logger.info("Model loaded successfully")
            return self._model, self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _find_model_path(self) -> str:
        """
        Find the model path from various possible locations.
        
        Returns:
            Path to the model
        """
        # Possible model locations
        possible_paths = [
            "models/export/saved_model",  # Exported SavedModel
            "models/efficientnet_final.keras",
            "models/resnet_final.keras",
            "sipakmed_best_2.keras",  # Legacy model
            os.environ.get("MODEL_PATH", "")
        ]
        
        for path in possible_paths:
            if path and Path(path).exists():
                logger.info(f"Found model at: {path}")
                return path
        
        raise FileNotFoundError(
            "No model found. Please train a model first or set MODEL_PATH environment variable."
        )
    
    def _load_metadata(self, model_path: str) -> Dict:
        """
        Load model metadata.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dictionary with model metadata
        """
        # Try to load from export directory
        metadata_path = Path("models/export/model_metadata.json")
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info("Loaded model metadata from export")
        else:
            # Use default metadata
            metadata = {
                "class_names": [
                    "Dyskeratotic",
                    "Koilocytotic",
                    "Metaplastic",
                    "Parabasal",
                    "Superficial-Intermediate"
                ],
                "num_classes": 5,
                "input_size": 224,
                "version": "1.0.0",
                "preprocessing": {
                    "resize": 224,
                    "normalization": "divide_by_255",
                    "color_mode": "rgb"
                }
            }
            logger.warning("Using default metadata (model_metadata.json not found)")
        
        return metadata
    
    def _warmup_model(self):
        """Warm up the model with a dummy prediction."""
        if self._model is None:
            return
        
        try:
            input_size = self._metadata.get('input_size', 224)
            dummy_input = tf.random.normal((1, input_size, input_size, 3))
            _ = self._model.predict(dummy_input, verbose=0)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def get_model(self) -> tf.keras.Model:
        """Get the loaded model."""
        if self._model is None:
            self.load_model()
        return self._model
    
    def get_metadata(self) -> Dict:
        """Get the model metadata."""
        if self._metadata is None:
            self.load_model()
        return self._metadata
    
    def get_class_names(self) -> list:
        """Get list of class names."""
        metadata = self.get_metadata()
        return metadata.get('class_names', [])
    
    def get_input_size(self) -> int:
        """Get the expected input size."""
        metadata = self.get_metadata()
        return metadata.get('input_size', 224)


# Global model loader instance
model_loader = ModelLoader()
