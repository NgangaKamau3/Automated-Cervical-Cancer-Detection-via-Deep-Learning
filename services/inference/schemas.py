"""
Pydantic Schemas for Request/Response Validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    
    predicted_class: str = Field(..., description="Predicted cell type class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(default="1.0.0", description="Model version used")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "Superficial-Intermediate",
                "confidence": 0.92,
                "all_probabilities": {
                    "Dyskeratotic": 0.02,
                    "Koilocytotic": 0.01,
                    "Metaplastic": 0.03,
                    "Parabasal": 0.02,
                    "Superficial-Intermediate": 0.92
                },
                "processing_time_ms": 45.3,
                "model_version": "1.0.0",
                "timestamp": "2025-12-17T18:00:00"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_images: int = Field(..., description="Total number of images processed")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "predicted_class": "Superficial-Intermediate",
                        "confidence": 0.92,
                        "all_probabilities": {},
                        "processing_time_ms": 45.3
                    }
                ],
                "total_images": 1,
                "total_processing_time_ms": 50.2
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    gpu_available: bool = Field(default=False, description="Whether GPU is available")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "uptime_seconds": 3600.5,
                "gpu_available": True
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_version: str = Field(..., description="Model version")
    architecture: str = Field(..., description="Model architecture")
    input_size: int = Field(..., description="Expected input image size")
    num_classes: int = Field(..., description="Number of output classes")
    class_names: List[str] = Field(..., description="List of class names")
    test_accuracy: Optional[float] = Field(None, description="Test set accuracy")
    framework: str = Field(default="tensorflow", description="ML framework")
    
    class Config:
        schema_extra = {
            "example": {
                "model_version": "1.0.0",
                "architecture": "efficientnet",
                "input_size": 224,
                "num_classes": 5,
                "class_names": [
                    "Dyskeratotic",
                    "Koilocytotic",
                    "Metaplastic",
                    "Parabasal",
                    "Superficial-Intermediate"
                ],
                "test_accuracy": 0.9427,
                "framework": "tensorflow"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ImageProcessingError",
                "message": "Failed to decode image",
                "detail": "Image file is corrupted or invalid format",
                "timestamp": "2025-12-17T18:00:00"
            }
        }
