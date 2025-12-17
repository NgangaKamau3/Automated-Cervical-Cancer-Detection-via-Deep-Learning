# Automated Cervical Cancer Detection via Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Production-grade deep learning system for automated cervical cytology classification using state-of-the-art computer vision.

## 🎯 Overview

This system provides an end-to-end solution for cervical cancer detection through automated Pap smear image analysis. It combines modern ML architecture, production-ready inference services, and comprehensive deployment tools.

**Key Features:**
- 🧠 **State-of-the-art Models**: EfficientNetB3/ResNet50V2 with transfer learning
- 🚀 **Production API**: FastAPI-based inference service with comprehensive validation
- 📊 **Interactive UI**: Streamlit dashboard for easy visualization
- 🐳 **Cloud-Ready**: Docker + Kubernetes deployment configurations
- 📈 **MLOps**: Comprehensive training pipeline with experiment tracking
- ✅ **High Accuracy**: >92% test accuracy on SIPaKMeD dataset

**Classification Categories:**
1. Dyskeratotic
2. Koilocytotic
3. Metaplastic
4. Parabasal
5. Superficial-Intermediate

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA support for faster training
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/NgangaKamau3/Automated-Cervical-Cancer-Detection-via-Deep-Learning.git
cd Automated-Cervical-Cancer-Detection-via-Deep-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download SIPaKMeD dataset from Kaggle
python scripts/download_dataset.py
```

### Train Model

```bash
# Train with EfficientNetB3 (recommended)
python ml/train.py --architecture efficientnet --batch-size 32

# Or train with ResNet50V2 (faster)
python ml/train.py --architecture resnet --batch-size 32
```

### Run Inference Service

```bash
# Start FastAPI server
python services/inference/main.py --host 0.0.0.0 --port 8000

# Or use uvicorn directly
uvicorn services.inference.main:app --host 0.0.0.0 --port 8000
```

### Run Gradio UI (HuggingFace Spaces)
This project features a clinical-grade Gradio interface suitable for HuggingFace Spaces.

```bash
# Run locally
python app.py
```
**Deploy to HuggingFace:**
1. Create a new Space (Gradio SDK)
2. Upload `app.py`, `requirements.txt`, and `models/` (or use `ml` folder)
3. Launch!

## 📊 Model Performance

| Architecture | Test Accuracy | Parameters | Inference Time |
|-------------|---------------|------------|----------------|
| EfficientNetB3 | ~94% | ~12M | ~45ms |
| ResNet50V2 | ~92% | ~25M | ~35ms |
| Legacy CNN | ~91.8% | ~1M | ~15ms |

*Benchmarks on NVIDIA T4 GPU*

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Access services
# API: http://localhost:8000
# UI: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### Manual Docker Build

```bash
# Build image
docker build -t cervical-cancer-detection:latest .

# Run API service
docker run -p 8000:8000 cervical-cancer-detection:latest

# Run Streamlit UI
docker run -p 8501:8501 cervical-cancer-detection:latest \
    streamlit run App.py --server.port=8501 --server.address=0.0.0.0
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

## 📚 API Documentation

### Interactive API Docs
Visit `http://localhost:8000/docs` when running the service for interactive Swagger UI documentation.

### Key Endpoints

#### Health Check
```bash
GET /health
```

#### Single Image Prediction
```bash
POST /predict
Content-Type: multipart/form-data

# Example with curl
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: multipart/form-data

# Example with curl
curl -X POST "http://localhost:8000/predict/batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

#### Model Information
```bash
GET /model/info
```

## 🛠️ Development

### Project Structure
```
.
├── ml/                         # Machine learning components
│   ├── data_loader.py         # Data pipeline
│   ├── model.py               # Model architectures
│   ├── train.py               # Training script
│   └── export.py              # Model export utilities
├── services/                   # Production services
│   └── inference/             # FastAPI inference service
│       ├── main.py            # API server
│       ├── schemas.py         # Pydantic models
│       ├── model_loader.py    # Model management
│       └── preprocessing.py   # Image preprocessing
├── scripts/                    # Utility scripts
│   └── download_dataset.py    # Dataset downloader
├── k8s/                        # Kubernetes manifests
├── config/                     # Configuration files
├── tests/                      # Test suite
├── App.py                      # Streamlit UI
├── Dockerfile                  # Container image
└── docker-compose.yml         # Multi-service deployment
```

### Running Tests
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ml --cov=services --cov-report=html
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 ml/ services/ tests/

# Type checking
mypy ml/ services/
```

## 📖 Additional Documentation

- [Training Guide](docs/TRAINING.md) - Detailed training instructions
- [API Reference](docs/API.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Cloud deployment walkthrough
- [Development Guide](docs/DEVELOPMENT.md) - Contributing guidelines

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{wanjohi2024cervical,
  title={Automated Cervical Cytology Classification Using Deep Learning},
  author={Wanjohi, Mikenickson and Kamau, Nganga},
  booktitle={16th KEMRI Annual Scientific \& Health Conference},
  year={2024}
}
```

## 📊 Dataset

This project uses the [SIPaKMeD dataset](https://www.cs.uoi.gr/~marina/sipakmed.html):
- 4,049 isolated cells
- 5 cell categories
- High-resolution images (original dimensions preserved in training)

**Reference:**
> Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., & Charchanti, A. (2018). SIPAKMED: A new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images. *25th IEEE International Conference on Image Processing (ICIP)*.

## 🙏 Acknowledgments

- **Dataset**: SIPaKMeD dataset creators (University of Ioannina)
- **Framework**: TensorFlow/Keras team
- **Pretrained Models**: ImageNet contributors
- **Conference**: 16th KEMRI Annual Scientific & Health Conference

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Mikenickson Wanjohi** - System Architecture & Implementation
- **Nganga Kamau** - ML Development & Optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**⚕️ Medical Disclaimer**: This software is for research and educational purposes only. It is not intended for clinical diagnostic use. Always consult with qualified medical professionals for medical advice and diagnosis.