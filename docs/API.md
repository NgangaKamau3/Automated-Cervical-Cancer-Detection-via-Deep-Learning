# API Reference

Complete API documentation for the Cervical Cancer Detection inference service.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.cervical-cancer.example.com`

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API key authentication or OAuth2.

## Endpoints

### 1. Root Endpoint

**GET /**

Returns service information and available endpoints.

**Response:**
```json
{
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
```

---

### 2. Health Check

**GET /health**

Check service health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 3600.5,
  "gpu_available": true
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

---

### 3. Single Image Prediction

**POST /predict**

Classify a single cervical cell image.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body Parameters**:
  - `file` (required): Image file (JPEG, PNG, BMP)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@cell_image.jpg"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("cell_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "predicted_class": "Superficial-Intermediate",
  "confidence": 0.9234,
  "all_probabilities": {
    "Dyskeratotic": 0.0123,
    "Koilocytotic": 0.0156,
    "Metaplastic": 0.0287,
    "Parabasal": 0.0200,
    "Superficial-Intermediate": 0.9234
  },
  "processing_time_ms": 45.3,
  "model_version": "1.0.0",
  "timestamp": "2025-12-17T18:00:00.000Z"
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid image format or size
- `500 Internal Server Error`: Prediction failed

---

### 4. Batch Prediction

**POST /predict/batch**

Classify multiple cervical cell images in one request.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body Parameters**:
  - `files` (required, multiple): Array of image files (max 50 images)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "accept: application/json" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/predict/batch"
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "predictions": [
    {
      "predicted_class": "Superficial-Intermediate",
      "confidence": 0.9234,
      "all_probabilities": { ... },
      "processing_time_ms": 15.2,
      "model_version": "1.0.0",
      "timestamp": "2025-12-17T18:00:00.000Z"
    },
    // ... more predictions
  ],
  "total_images": 3,
  "total_processing_time_ms": 52.8
}
```

**Status Codes:**
- `200 OK`: Batch prediction successful
- `400 Bad Request`: Invalid images, too many files, or empty batch
- `500 Internal Server Error`: Batch prediction failed

**Limits:**
- Maximum 50 images per batch
- Maximum 10MB per image
- Supported formats: JPEG, PNG, BMP, TIFF

---

### 5. Model Information

**GET /model/info**

Get information about the loaded model.

**Response:**
```json
{
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
```

**Status Codes:**
- `200 OK`: Model info retrieved
- `500 Internal Server Error`: Failed to retrieve model info

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "detail": "Additional error details",
  "timestamp": "2025-12-17T18:00:00.000Z"
}
```

**Common Error Types:**
- `ImageProcessingError`: Invalid or corrupted image
- `ValidationError`: Invalid request parameters
- `ModelLoadError`: Model failed to load
- `InternalServerError`: Unexpected server error

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployments, consider implementing rate limiting using:
- API Gateway (e.g., Kong, AWS API Gateway)
- Middleware (e.g., slowapi)

Recommended limits:
- `/predict`: 100 requests/minute per IP
- `/predict/batch`: 20 requests/minute per IP
- `/health`: Unlimited

---

## Interactive Documentation

Visit `/docs` for interactive Swagger UI documentation where you can test endpoints directly in your browser.

Visit `/redoc` for alternative ReDoc documentation.

---

## Performance Considerations

**Latency:**
- Single prediction: ~45ms (with GPU), ~150ms (CPU only)
- Batch prediction: ~100ms for 10 images (with GPU)

**Optimization Tips:**
1. Use batch prediction for multiple images
2. Enable GPU for faster inference
3. Consider using TFLite models for edge deployment
4. Implement caching for frequently predicted images

---

## Examples

### Complete Python Client

```python
import requests
from pathlib import Path

class CervicalCancerClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_single(self, image_path):
        """Predict single image."""
        files = {"file": open(image_path, "rb")}
        response = requests.post(
            f"{self.base_url}/predict",
            files=files
        )
        return response.json()
    
    def predict_batch(self, image_paths):
        """Predict multiple images."""
        files = [
            ("files", open(path, "rb"))
            for path in image_paths
        ]
        response = requests.post(
            f"{self.base_url}/predict/batch",
            files=files
        )
        return response.json()
    
    def get_model_info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()

# Usage
client = CervicalCancerClient()

# Check health
print(client.health_check())

# Predict single image
result = client.predict_single("cell_image.jpg")
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Predict batch
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
batch_results = client.predict_batch(images)
for i, pred in enumerate(batch_results['predictions']):
    print(f"Image {i+1}: {pred['predicted_class']} ({pred['confidence']:.2%})")
```

### Node.js Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function predictImage(imagePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));
  
  const response = await axios.post(
    'http://localhost:8000/predict',
    form,
    { headers: form.getHeaders() }
  );
  
  return response.data;
}

// Usage
predictImage('cell_image.jpg')
  .then(result => {
    console.log(`Predicted: ${result.predicted_class}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  })
  .catch(error => console.error(error));
```
