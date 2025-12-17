# Deployment Guide

Complete guide for deploying the Cervical Cancer Detection system to production.

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development

### Prerequisites
- Python 3.10+
- 8GB+ RAM
- (Optional) NVIDIA GPU with CUDA 11.8+

### Setup

```bash
# Clone repository
git clone https://github.com/NgangaKamau3/Automated-Cervical-Cancer-Detection-via-Deep-Learning.git
cd Automated-Cervical-Cancer-Detection-via-Deep-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset and train model
python scripts/download_dataset.py
python ml/train.py
```

### Running Services

```bash
# Terminal 1: API Service
python services/inference/main.py --port 8000

# Terminal 2: Streamlit UI
streamlit run App.py
```

---

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t cervical-cancer-detection:latest .

# Run API service
docker run -d \
  --name cervical-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  cervical-cancer-detection:latest

# Run Streamlit UI
docker run -d \
  --name cervical-ui \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  cervical-cancer-detection:latest \
  streamlit run App.py --server.port=8501 --server.address=0.0.0.0
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

**Access:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

---

## Kubernetes Deployment

### Prerequisites
- kubectl configured
- Kubernetes cluster (GKE, EKS, AKS, or local minikube)
- Container registry access

### Step 1: Build and Push Image

```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT/cervical-cancer-detection:v1.0.0 .

# Push to registry
docker push gcr.io/YOUR_PROJECT/cervical-cancer-detection:v1.0.0

# Update image in k8s/deployment.yaml
sed -i 's|cervical-cancer-detection:latest|gcr.io/YOUR_PROJECT/cervical-cancer-detection:v1.0.0|' k8s/deployment.yaml
```

### Step 2: Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace cervical-cancer

# Apply configurations
kubectl apply -f k8s/service.yaml -n cervical-cancer
kubectl apply -f k8s/deployment.yaml -n cervical-cancer
kubectl apply -f k8s/hpa.yaml -n cervical-cancer

# Optional: Apply ingress for HTTPS
kubectl apply -f k8s/ingress.yaml -n cervical-cancer
```

### Step 3: Verify Deployment

```bash
# Check pods
kubectl get pods -n cervical-cancer

# Check services
kubectl get svc -n cervical-cancer

# View logs
kubectl logs -f deployment/cervical-cancer-api -n cervical-cancer

# Get service URL
kubectl get svc cervical-cancer-api -n cervical-cancer
```

### Step 4: Upload Model to Persistent Volume

```bash
# Create PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cervical-cancer-models-pvc
  namespace: cervical-cancer
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
EOF

# Copy model files (using a temporary pod)
kubectl run -n cervical-cancer model-uploader \
  --image=busybox \
  --restart=Never \
  --command -- sleep 3600

kubectl cp models/ cervical-cancer/model-uploader:/models
```

---

## Cloud Platforms

### Google Cloud Platform (GKE)

#### 1. Setup GKE Cluster

```bash
# Create cluster
gcloud container clusters create cervical-cancer-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials cervical-cancer-cluster \
  --zone us-central1-a
```

#### 2. Build and Push to GCR

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/YOUR_PROJECT_ID/cervical-cancer-detection:v1.0.0 .
docker push gcr.io/YOUR_PROJECT_ID/cervical-cancer-detection:v1.0.0
```

#### 3. Deploy Application

```bash
# Deploy
kubectl apply -f k8s/ -n cervical-cancer

# Get external IP
kubectl get svc cervical-cancer-api -n cervical-cancer
```

#### 4. Setup Cloud Load Balancer

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml -n cervical-cancer
```

### AWS (EKS)

```bash
# Create cluster with eksctl
eksctl create cluster \
  --name cervical-cancer-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10

# Build and push to ECR
aws ecr create-repository --repository-name cervical-cancer-detection
docker build -t YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/cervical-cancer-detection:v1.0.0 .
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/cervical-cancer-detection:v1.0.0

# Deploy
kubectl apply -f k8s/ -n cervical-cancer
```

### Azure (AKS)

```bash
# Create cluster
az aks create \
  --resource-group cervical-cancer-rg \
  --name cervical-cancer-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity

# Get credentials
az aks get-credentials \
  --resource-group cervical-cancer-rg \
  --name cervical-cancer-cluster

# Build and push to ACR
az acr create --resource-group cervical-cancer-rg --name cervicalcanceracr --sku Basic
az acr build --registry cervicalcanceracr --image cervical-cancer-detection:v1.0.0 .

# Deploy
kubectl apply -f k8s/ -n cervical-cancer
```

---

## Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://YOUR_SERVICE_IP:8000/health

# Kubernetes
kubectl get pods -n cervical-cancer
kubectl top pods -n cervical-cancer
```

### Logs

```bash
# Docker
docker logs -f cervical-api

# Kubernetes
kubectl logs -f deployment/cervical-cancer-api -n cervical-cancer
kubectl logs -f deployment/cervical-cancer-ui -n cervical-cancer
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment cervical-cancer-api --replicas=5 -n cervical-cancer

# Auto-scaling is configured via HPA (k8s/hpa.yaml)
kubectl get hpa -n cervical-cancer
```

### Updates

```bash
# Build new version
docker build -t YOUR_REGISTRY/cervical-cancer-detection:v1.1.0 .
docker push YOUR_REGISTRY/cervical-cancer-detection:v1.1.0

# Update deployment
kubectl set image deployment/cervical-cancer-api \
  api=YOUR_REGISTRY/cervical-cancer-detection:v1.1.0 \
  -n cervical-cancer

# Rollback if needed
kubectl rollout undo deployment/cervical-cancer-api -n cervical-cancer
```

### Backup

```bash
# Backup model files
kubectl cp cervical-cancer/POD_NAME:/app/models ./models-backup

# Backup database (if applicable)
# Setup regular backups using cloud provider tools
```

---

## Production Checklist

- [ ] SSL/TLS certificates configured
- [ ] Authentication/authorization implemented
- [ ] Rate limiting enabled
- [ ] Monitoring and alerting setup (Prometheus/Grafana)
- [ ] Backup strategy in place
- [ ] Disaster recovery plan documented
- [ ] Security scanning enabled
- [ ] Resource limits configured
- [ ] Auto-scaling tested
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team training completed

---

## Troubleshooting

### Model Not Loading

**Problem:** API returns "model_loaded": false

**Solution:**
1. Check if model files exist in the container
2. Verify MODEL_PATH environment variable
3. Check pod logs for errors
4. Ensure sufficient memory allocation

### High Latency

**Problem:** Predictions taking >1 second

**Solution:**
1. Enable GPU support
2. Increase pod resources
3. Use TFLite models for inference
4. Enable batch prediction
5. Add caching layer

### Out of Memory

**Problem:** Pods being killed due to OOM

**Solution:**
1. Increase memory limits in deployment.yaml
2. Reduce batch size
3. Enable model quantization
4. Use smaller model architecture

---

## Support

For issues or questions:
1. Check logs first
2. Review documentation
3. Open GitHub issue
4. Contact team via email

**Emergency Contacts:**
- DevOps: devops@example.com
- ML Team: ml-team@example.com
