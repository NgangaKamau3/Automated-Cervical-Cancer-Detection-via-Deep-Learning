"""
FastAPI Inference Service
Production-ready REST API for cervical cancer cell classification.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from services.inference.schemas import (
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from services.inference.model_loader import model_loader
from services.inference.preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cervical Cancer Detection API",
    description="Production API for automated cervical cytology classification using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
START_TIME = time.time()
preprocessor = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and preprocessor on startup."""
    global preprocessor
    
    logger.info("Starting up inference service...")
    
    try:
        # Load model
        model, metadata = model_loader.load_model()
        logger.info(f"Model loaded successfully: {metadata.get('version', 'unknown')}")
        
        # Initialize preprocessor
        input_size = metadata.get('input_size', 224)
        preprocessor = ImagePreprocessor(target_size=input_size)
        logger.info(f"Preprocessor initialized with input size: {input_size}")
        
        logger.info("✅ Service startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down inference service...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Cervical Cancer Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and model information.
    """
    try:
        model = model_loader.get_model()
        metadata = model_loader.get_metadata()
        model_loaded = model is not None
        
        # Check GPU availability
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        uptime = time.time() - START_TIME 
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            model_version=metadata.get('version', '1.0.0'),
            uptime_seconds=uptime,
            gpu_available=gpu_available
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e)
            }
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model information and metadata.
    
    Returns details about the model architecture, classes, and performance.
    """
    try:
        metadata = model_loader.get_metadata()
        
        return ModelInfoResponse(
            model_version=metadata.get('version', '1.0.0'),
            architecture=metadata.get('training', {}).get('architecture', 'efficientnet'),
            input_size=metadata.get('input_size', 224),
            num_classes=metadata.get('num_classes', 5),
            class_names=metadata.get('class_names', []),
            test_accuracy=metadata.get('performance', {}).get('test_accuracy'),
            framework=metadata.get('framework', 'tensorflow')
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Cervical cell image file")):
    """
    Predict cell type from a single image.
    
    Upload a Pap smear cell image to get classification results.
    
    - **file**: Image file (JPEG, PNG, BMP) of cervical cell
    
    Returns predicted class, confidence score, and probabilities for all classes.
    """
    start_time = time.time()
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Validate image
        is_valid, error_msg = preprocessor.validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Preprocess image
        try:
            processed_image = preprocessor.preprocess_from_bytes(image_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image preprocessing failed: {str(e)}"
            )
        
        # Add batch dimension
        batch_image = np.expand_dims(processed_image, axis=0)
        
        # Get prediction
        model = model_loader.get_model()
        predictions = model.predict(batch_image, verbose=0)
        
        # Get class names
        class_names = model_loader.get_class_names()
        
        # Extract results
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Create probability dictionary
        all_probabilities = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get model version
        metadata = model_loader.get_metadata()
        model_version = metadata.get('version', '1.0.0')
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probabilities,
            processing_time_ms=processing_time_ms,
            model_version=model_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(files: List[UploadFile] = File(..., description="Multiple cervical cell images")):
    """
    Predict cell types from multiple images.
    
    Upload multiple Pap smear cell images to get batch classification results.
    
    - **files**: List of image files (JPEG, PNG, BMP)
    
    Returns predictions for all uploaded images.
    """
    start_time = time.time()
    
    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 images per batch"
        )
    
    try:
        # Read all images
        image_bytes_list = []
        for file in files:
            image_bytes = await file.read()
            
            # Validate
            is_valid, error_msg = preprocessor.validate_image(image_bytes)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid image '{file.filename}': {error_msg}"
                )
            
            image_bytes_list.append(image_bytes)
        
        # Preprocess batch
        batch_images = preprocessor.preprocess_batch(image_bytes_list)
        
        # Get predictions
        model = model_loader.get_model()
        predictions = model.predict(batch_images, verbose=0)
        
        # Get class names
        class_names = model_loader.get_class_names()
        metadata = model_loader.get_metadata()
        model_version = metadata.get('version', '1.0.0')
        
        # Create responses
        prediction_responses = []
        for i, pred in enumerate(predictions):
            pred_start_time = time.time()
            
            predicted_class_idx = int(np.argmax(pred))
            predicted_class = class_names[predicted_class_idx]
            confidence = float(pred[predicted_class_idx])
            
            all_probabilities = {
                class_names[j]: float(pred[j])
                for j in range(len(class_names))
            }
            
            pred_time_ms = (time.time() - pred_start_time) * 1000
            
            prediction_responses.append(
                PredictionResponse(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    all_probabilities=all_probabilities,
                    processing_time_ms=pred_time_ms,
                    model_version=model_version
                )
            )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Batch prediction complete: {len(files)} images in {total_time_ms:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_images=len(files),
            total_processing_time_ms=total_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def main():
    """Run the inference service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "services.inference.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
