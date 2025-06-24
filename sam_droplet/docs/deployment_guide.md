# SAM Droplet Segmentation - Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Development Deployment](#development-deployment)
5. [Production Deployment](#production-deployment)
6. [Docker Deployment](#docker-deployment)
7. [Cloud Deployment](#cloud-deployment)
8. [Performance Tuning](#performance-tuning)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Security Considerations](#security-considerations)
11. [Troubleshooting](#troubleshooting)

## Overview

This guide covers deploying the SAM Droplet Segmentation application in various environments, from local development to production cloud deployments.

### Deployment Options

| Option | Use Case | Complexity | Performance |
|--------|----------|------------|-------------|
| Flask Dev | Development/Testing | Low | Basic |
| Waitress | Small Production | Medium | Good |
| Gunicorn | Linux Production | Medium | Excellent |
| Docker | Containerized | High | Excellent |
| Cloud | Scalable Production | High | Excellent |

## System Requirements

### Minimum Requirements

- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **CPU**: 4+ cores recommended

### Recommended Requirements

- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.9+
- **RAM**: 32GB+
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA RTX 3060+ with 8GB+ VRAM
- **CPU**: 8+ cores

### GPU Requirements (Optional but Recommended)

- **NVIDIA GPU** with CUDA 11.0+ support
- **VRAM**: 6GB+ (8GB+ recommended)
- **CUDA**: 11.0 or later
- **cuDNN**: Compatible version

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd sam_droplet

# Install dependencies
pip install -r requirements_server.txt

# Download SAM model (if not included)
# Place mobile_sam.pt in model/ directory
```

### 2. Run Development Server

```bash
python app.py
```

Navigate to `http://localhost:9487`

## Development Deployment

### Flask Development Server

Best for: Local development, testing, debugging

```bash
# Basic run
python app.py

# Custom host/port
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
python app.py

# Debug mode
export FLASK_DEBUG=True
python app.py
```

### Configuration

Create `.env` file:

```env
# Server configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=9487
FLASK_DEBUG=True

# Model configuration
SAM_MODEL_PATH=model/mobile_sam.pt

# Performance
OMP_NUM_THREADS=4
```

Load in `app.py`:

```python
from dotenv import load_dotenv
load_dotenv()

host = os.getenv('FLASK_HOST', '0.0.0.0')
port = int(os.getenv('FLASK_PORT', 9487))
debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

app.run(host=host, port=port, debug=debug)
```

## Production Deployment

### Option 1: Waitress (Cross-Platform)

Best for: Windows servers, small-medium traffic

#### Basic Setup

```python
# waitress_config.py
from waitress import serve
from app import app
import os

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    serve(app, 
          host=host, 
          port=port,
          threads=6,
          connection_limit=1000,
          channel_timeout=300,
          max_request_body_size=104857600,  # 100MB
          cleanup_interval=30)
```

#### Run Waitress

```bash
python waitress_config.py
```

#### Advanced Configuration

```python
# waitress_production.py
from waitress import serve
from app import app
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('waitress')

if __name__ == '__main__':
    serve(app,
          host='0.0.0.0',
          port=5000,
          
          # Threading
          threads=8,
          
          # Connection handling
          connection_limit=1000,
          cleanup_interval=30,
          
          # Request handling
          channel_timeout=600,
          max_request_body_size=104857600,  # 100MB
          
          # Logging
          call_log_format='%(REMOTE_ADDR)s "%(REQUEST_METHOD)s %(REQUEST_URI)s %(HTTP_VERSION)s" %(status)s %(bytes)s "%(HTTP_REFERER)s" "%(HTTP_USER_AGENT)s"')
```

### Option 2: Gunicorn (Linux/macOS)

Best for: Linux production servers, high traffic

#### Basic Setup

```bash
# Install gunicorn
pip install gunicorn

# Run with basic config
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
```

#### Production Configuration

Create `gunicorn.conf.py`:

```python
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = min(4, multiprocessing.cpu_count())
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Restart workers
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/sam_droplet/access.log"
errorlog = "/var/log/sam_droplet/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'sam_droplet'

# Server mechanics
daemon = False
pidfile = '/var/run/sam_droplet.pid'
user = 'www-data'
group = 'www-data'
tmp_upload_dir = None

# SSL (if needed)
# keyfile = '/path/to/private.key'
# certfile = '/path/to/certificate.crt'
```

Run with config:

```bash
gunicorn -c gunicorn.conf.py app:app
```

#### Systemd Service

Create `/etc/systemd/system/sam-droplet.service`:

```ini
[Unit]
Description=SAM Droplet Segmentation Service
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/sam_droplet
Environment=PATH=/opt/sam_droplet/venv/bin
ExecStart=/opt/sam_droplet/venv/bin/gunicorn -c gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable sam-droplet
sudo systemctl start sam-droplet
sudo systemctl status sam-droplet
```

### Option 3: uWSGI

Best for: High-performance production with Nginx

#### uWSGI Configuration

Create `uwsgi.ini`:

```ini
[uwsgi]
module = app:app
master = true
processes = 4
threads = 2
socket = /tmp/sam_droplet.sock
chmod-socket = 666
vacuum = true
die-on-term = true
harakiri = 300
max-requests = 1000
buffer-size = 32768
```

#### Nginx Configuration

Create `/etc/nginx/sites-available/sam_droplet`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 100M;
    
    location / {
        include uwsgi_params;
        uwsgi_pass unix:/tmp/sam_droplet.sock;
        uwsgi_read_timeout 300s;
        proxy_send_timeout 300s;
    }
    
    location /static {
        alias /opt/sam_droplet/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/sam_droplet /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Docker Deployment

### Single Container

#### Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 sam_user && chown -R sam_user:sam_user /app
USER sam_user

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "waitress_config.py"]
```

#### Build and Run

```bash
# Build image
docker build -t sam-droplet:latest .

# Run container
docker run -d \
    --name sam-droplet \
    -p 5000:5000 \
    -v /path/to/model:/app/model:ro \
    --memory=8g \
    --cpus=4 \
    sam-droplet:latest
```

### GPU Support

#### GPU Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    # ... other dependencies

# Install PyTorch with CUDA
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# ... rest of Dockerfile
```

#### Run with GPU

```bash
docker run -d \
    --name sam-droplet-gpu \
    --gpus all \
    -p 5000:5000 \
    -v /path/to/model:/app/model:ro \
    --memory=16g \
    sam-droplet:gpu
```

### Docker Compose

#### docker-compose.yml

```yaml
version: '3.8'

services:
  sam-droplet:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./model:/app/model:ro
      - ./logs:/app/logs
    environment:
      - FLASK_HOST=0.0.0.0
      - FLASK_PORT=5000
      - OMP_NUM_THREADS=4
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - sam-droplet
    restart: unless-stopped
```

#### Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream sam_droplet {
        server sam-droplet:5000;
    }

    server {
        listen 80;
        client_max_body_size 100M;
        
        location / {
            proxy_pass http://sam_droplet;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300;
            proxy_send_timeout 300;
        }
    }
}
```

#### Run with Compose

```bash
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

1. **Launch EC2 Instance**:
   - AMI: Ubuntu 20.04 LTS
   - Instance Type: g4dn.xlarge (GPU) or m5.2xlarge (CPU)
   - Storage: 50GB+ EBS GP3

2. **Setup Instance**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker ubuntu

# For GPU instances - install NVIDIA drivers
sudo apt install -y nvidia-driver-470
sudo reboot

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

3. **Deploy Application**:

```bash
# Clone repository
git clone <repo-url>
cd sam_droplet

# Build and run
docker-compose up -d
```

#### ECS Deployment

Create `task-definition.json`:

```json
{
  "family": "sam-droplet",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "sam-droplet",
      "image": "your-ecr-repo/sam-droplet:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_HOST",
          "value": "0.0.0.0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sam-droplet",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/sam-droplet

# Deploy to Cloud Run
gcloud run deploy sam-droplet \
    --image gcr.io/PROJECT_ID/sam-droplet \
    --platform managed \
    --region us-central1 \
    --memory 8Gi \
    --cpu 4 \
    --timeout 600 \
    --max-instances 10 \
    --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name sam-droplet-rg --location eastus

# Deploy container
az container create \
    --resource-group sam-droplet-rg \
    --name sam-droplet \
    --image your-registry/sam-droplet:latest \
    --cpu 4 \
    --memory 8 \
    --ports 5000 \
    --environment-variables FLASK_HOST=0.0.0.0 \
    --restart-policy Always
```

## Performance Tuning

### CPU Optimization

```bash
# Set CPU affinity
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# For production
taskset -c 0-3 python app.py
```

### Memory Optimization

```python
# In app.py
import gc
import torch

def cleanup_memory():
    """Clean up memory after processing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Call after each request
@app.after_request
def after_request(response):
    cleanup_memory()
    return response
```

### GPU Optimization

```python
# Optimize GPU memory usage
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Use mixed precision
from torch.cuda.amp import autocast

@autocast()
def generate_masks(image):
    return mask_generator.generate(image)
```

## Monitoring and Logging

### Application Logging

```python
import logging
import logging.handlers
import os

# Configure logging
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE', 'sam_droplet.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10485760, backupCount=5),
            logging.StreamHandler()
        ]
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)
```

### Metrics Collection

```python
import time
import psutil
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            status = 'success'
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            status = 'error'
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"Performance - {func.__name__}: "
                       f"time={end_time-start_time:.2f}s, "
                       f"memory_delta={end_memory-start_memory:.1f}MB, "
                       f"status={status}")
        
        return result
    return wrapper

# Apply to endpoints
@app.route('/segment_file', methods=['POST'])
@monitor_performance
def segment_uploaded_file():
    # ... function implementation
```

### Health Checks

```python
@app.route('/health')
def health_check():
    """Comprehensive health check."""
    health_data = {
        'status': 'healthy',
        'timestamp': time.time(),
        'model_loaded': sam_model is not None,
        'gpu_available': torch.cuda.is_available(),
        'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
        'cpu_percent': psutil.cpu_percent()
    }
    
    if torch.cuda.is_available():
        health_data['gpu_memory'] = {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
            'cached': torch.cuda.memory_reserved() / 1024 / 1024
        }
    
    return jsonify(health_data)
```

## Security Considerations

### Input Validation

```python
from flask import request
import magic

def validate_image_file(file):
    """Validate uploaded image file."""
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > 100 * 1024 * 1024:  # 100MB limit
        raise ValueError("File too large")
    
    # Check file type
    file_type = magic.from_buffer(file.read(1024), mime=True)
    file.seek(0)
    
    allowed_types = ['image/jpeg', 'image/png', 'image/webp']
    if file_type not in allowed_types:
        raise ValueError("Invalid file type")
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/segment_file', methods=['POST'])
@limiter.limit("10 per minute")
def segment_uploaded_file():
    # ... function implementation
```

### HTTPS Configuration

For production, always use HTTPS:

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # ... rest of configuration
}
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures

```bash
# Check model file
ls -la model/mobile_sam.pt

# Check permissions
chmod 644 model/mobile_sam.pt

# Verify file integrity
md5sum model/mobile_sam.pt
```

#### 2. Memory Issues

```bash
# Monitor memory usage
free -h
htop

# Check swap
swapon --show

# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. GPU Issues

```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

#### 4. Network Issues

```bash
# Check port availability
netstat -tuln | grep 5000

# Test API endpoints
curl -f http://localhost:5000/health

# Check firewall
sudo ufw status
```

### Debugging Steps

1. **Check logs**: Review application and system logs
2. **Test components**: Verify each component individually
3. **Resource monitoring**: Monitor CPU, memory, and GPU usage
4. **Network connectivity**: Test API endpoints
5. **Dependencies**: Verify all requirements are installed

### Log Analysis

```bash
# Monitor application logs
tail -f sam_droplet.log

# Check system logs
journalctl -f -u sam-droplet

# Docker logs
docker logs -f sam-droplet
```

This deployment guide provides comprehensive instructions for deploying the SAM Droplet Segmentation application in various environments. Choose the deployment option that best fits your requirements and scale. 