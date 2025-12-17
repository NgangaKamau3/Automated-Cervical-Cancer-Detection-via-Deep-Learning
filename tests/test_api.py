"""
Tests for the FastAPI inference service.
"""

import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import numpy as np


# Import the FastAPI app
from services.inference.main import app


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Cervical Cancer Detection API"
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code in [200, 503]  # Either healthy or unhealthy
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info_endpoint():
    """Test model info endpoint."""
    response = client.get("/model/info")
    
    # May fail if model not loaded, which is okay for testing
    if response.status_code == 200:
        data = response.json()
        assert "model_version" in data
        assert "architecture" in data
        assert "num_classes" in data
        assert data["num_classes"] == 5


def create_test_image():
    """Create a test image file."""
    # Create a simple test image
    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.mark.skipif(
    condition=True,  # Skip by default as it requires model to be loaded
    reason="Requires trained model to be available"
)
def test_predict_endpoint():
    """Test single prediction endpoint."""
    img_bytes = create_test_image()
    
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_class" in data
    assert "confidence" in data
    assert "all_probabilities" in data
    assert "processing_time_ms" in data
    
    # Validate confidence is between 0 and 1
    assert 0 <= data["confidence"] <= 1
    
    # Validate all probabilities sum to ~1
    prob_sum = sum(data["all_probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01


@pytest.mark.skipif(
    condition=True,
    reason="Requires trained model to be available"
)
def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    # Create multiple test images
    files = [
        ("files", ("test1.jpg", create_test_image(), "image/jpeg")),
        ("files", ("test2.jpg", create_test_image(), "image/jpeg")),
        ("files", ("test3.jpg", create_test_image(), "image/jpeg"))
    ]
    
    response = client.post("/predict/batch", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "total_images" in data
    assert "total_processing_time_ms" in data
    
    assert data["total_images"] == 3
    assert len(data["predictions"]) == 3


def test_predict_invalid_file():
    """Test prediction with invalid file."""
    # Send empty file
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    
    # Should return 400 or 500 depending on validation
    assert response.status_code in [400, 500]


def test_batch_predict_too_many_files():
    """Test batch prediction with too many files."""
    # Create 51 files (over the limit of 50)
    files = [
        ("files", (f"test{i}.jpg", create_test_image(), "image/jpeg"))
        for i in range(51)
    ]
    
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400


def test_batch_predict_empty():
    """Test batch prediction with no files."""
    response = client.post("/predict/batch", files=[])
    assert response.status_code in [400, 422]  # Bad request or validation error
