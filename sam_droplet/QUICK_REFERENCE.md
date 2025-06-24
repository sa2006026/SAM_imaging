# SAM Droplet - Quick Reference Guide

## 🚀 **Current Setup Status**

- **✅ Containerized Application**: Running in Docker with Gunicorn
- **✅ Cloudflare Tunnel**: DNS routing via `tally-o.gavinlou.com`
- **✅ API Authentication**: Multi-tier API key system
- **✅ Health Monitoring**: Container and application health checks
- **✅ Production Ready**: Scalable and secure deployment

## 🐳 **Docker Commands**

### Basic Operations
```bash
# Start all services
docker compose up -d

# Start with GPU support
docker compose -f docker-compose.gpu.yml up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
docker compose logs sam-app
docker compose logs cloudflare-tunnel

# Stop services
docker compose down

# Restart services
docker compose restart

# Scale application
docker compose up -d --scale sam-app=3
```

### Maintenance
```bash
# Rebuild containers
docker compose build --no-cache
docker compose up -d

# Update containers
git pull
docker compose down
docker compose up -d --build

# View resource usage
docker stats

# Execute into container
docker compose exec sam-app bash
```

## 🔑 **API Endpoints**

### Base URL
```
https://tally-o.gavinlou.com
```

### Authentication
```bash
# All protected endpoints require this header
-H "X-API-Key: your-api-key"
```

### Available API Keys
```
Demo Key:  sam-demo-key-123     (50 requests/hour)
Admin Key: sam-admin-key-456    (200 requests/hour)
```

### Endpoints

#### **Health Check** (Public)
```bash
curl https://tally-o.gavinlou.com/health
```
Response:
```json
{"model_loaded": false, "status": "healthy"}
```

#### **Image Segmentation** (Protected)
```bash
# Base64 image segmentation
curl -X POST \
  -H "X-API-Key: sam-demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgoAAAANSU..."}' \
  https://tally-o.gavinlou.com/segment

# File upload segmentation
curl -X POST \
  -H "X-API-Key: sam-demo-key-123" \
  -F "file=@image.jpg" \
  https://tally-o.gavinlou.com/segment_file
```

#### **Image Analysis** (Protected)
```bash
curl -X POST \
  -H "X-API-Key: sam-demo-key-123" \
  -F "file=@image.jpg" \
  https://tally-o.gavinlou.com/analyze_image
```

#### **Admin Endpoints** (Admin Only)
```bash
# Usage statistics
curl -H "X-Admin-Key: admin-secret-key-change-me" \
  https://tally-o.gavinlou.com/admin/stats

# Detailed health check
curl -H "X-Admin-Key: admin-secret-key-change-me" \
  https://tally-o.gavinlou.com/admin/health
```

## 🔧 **Configuration**

### Environment Variables (.env)
```env
# Security
API_KEYS=key1:name1:limit1,key2:name2:limit2
ADMIN_API_KEY=admin-secret-key-change-me
HEALTH_CHECK_KEY=health-monitor-key

# Performance
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
MAX_FILE_SIZE_MB=50
LOG_LEVEL=INFO

# GPU (if using GPU setup)
CUDA_VISIBLE_DEVICES=0
```

### Docker Compose Files
- `docker-compose.yml` - CPU-based deployment
- `docker-compose.gpu.yml` - GPU-enabled deployment

## 🌐 **Cloudflare Tunnel**

### Current Configuration
```yaml
tunnel: 0ac39a6b-fe66-4408-8597-31e996f9e896
credentials-file: /home/gavin/.cloudflared/0ac39a6b-fe66-4408-8597-31e996f9e896.json

ingress:
  - hostname: tally-o.gavinlou.com
    service: http://localhost:9487
  - service: http_status:404
```

### Tunnel Commands
```bash
# Check tunnel status
cloudflared tunnel info tally-o

# Run tunnel (if not using Docker)
cloudflared tunnel --config config.yml run tally-o

# Create DNS record
cloudflared tunnel route dns tally-o tally-o.gavinlou.com
```

## 🔍 **Monitoring & Debugging**

### Health Checks
```bash
# Container health
docker compose ps

# Application health
curl https://tally-o.gavinlou.com/health

# Detailed health (admin)
curl -H "X-Admin-Key: admin-secret-key-change-me" \
  https://tally-o.gavinlou.com/admin/health

# Local health check
curl -H "X-API-Key: sam-demo-key-123" http://localhost:9487/health
```

### Logs
```bash
# All logs
docker compose logs -f

# Application logs only
docker compose logs -f sam-app

# Tunnel logs only
docker compose logs -f cloudflare-tunnel

# Follow specific container
docker logs -f sam-droplet-app
```

### Performance Monitoring
```bash
# Resource usage
docker stats

# Container processes
docker compose exec sam-app ps aux

# Disk usage
docker system df

# Network status
docker network ls
```

## 🛠️ **Troubleshooting**

### Common Issues

#### **503 Errors**
```bash
# Check if container is running
docker compose ps

# Check application health
curl http://localhost:9487/health

# Check tunnel connection
docker compose logs cloudflare-tunnel
```

#### **Authentication Errors**
```bash
# Test without API key (should fail)
curl https://tally-o.gavinlou.com/segment

# Test with correct API key
curl -H "X-API-Key: sam-demo-key-123" https://tally-o.gavinlou.com/health
```

#### **Container Issues**
```bash
# Restart containers
docker compose restart

# Rebuild containers
docker compose down
docker compose build --no-cache
docker compose up -d

# Check container logs
docker compose logs sam-app
```

#### **Model Loading Issues**
```bash
# Check model file exists
docker compose exec sam-app ls -la /app/model/

# Check Python imports
docker compose exec sam-app python -c "import mobile_sam; print('OK')"

# Check GPU availability
docker compose exec sam-app python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 **Testing Examples**

### Test Image Segmentation
```bash
# Create test image (base64)
base64 -w 0 test_image.jpg > image_b64.txt

# Test segmentation
curl -X POST \
  -H "X-API-Key: sam-demo-key-123" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/jpeg;base64,$(cat image_b64.txt)\"}" \
  https://tally-o.gavinlou.com/segment
```

### Load Testing
```bash
# Simple load test
for i in {1..10}; do
  curl -H "X-API-Key: sam-demo-key-123" \
    https://tally-o.gavinlou.com/health &
done
wait
```

## 🔄 **Deployment Workflows**

### Development Deployment
```bash
# 1. Clone repository
git clone <repo-url>
cd sam_droplet

# 2. Set up environment
cp .dockerenv.example .env
nano .env  # Edit configuration

# 3. Start services
docker compose up -d

# 4. Test deployment
curl -H "X-API-Key: sam-demo-key-123" https://tally-o.gavinlou.com/health
```

### Production Deployment
```bash
# 1. Use production environment
cp .dockerenv.example .env
# Edit .env with production settings

# 2. Deploy with GPU support
docker compose -f docker-compose.gpu.yml up -d

# 3. Scale for load
docker compose up -d --scale sam-app=3

# 4. Monitor deployment
docker compose logs -f
```

### Update Deployment
```bash
# 1. Pull updates
git pull

# 2. Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d

# 3. Verify update
docker compose ps
curl -H "X-API-Key: sam-demo-key-123" https://tally-o.gavinlou.com/health
```

## 📝 **Files Overview**

### Docker Files
- `Dockerfile` - Main application container
- `Dockerfile.tunnel` - Cloudflare tunnel container  
- `docker-compose.yml` - CPU deployment
- `docker-compose.gpu.yml` - GPU deployment
- `.dockerignore` - Build optimization

### Configuration
- `.env` - Environment variables
- `config.yml` - Tunnel configuration
- `requirements_server.txt` - Python dependencies

### Documentation
- `README_DOCKER.md` - Detailed setup guide
- `CHANGELOG.md` - Version history and features
- `ARCHITECTURE.md` - System architecture
- `QUICK_REFERENCE.md` - This file

### Application Code
- `app.py` - Main Flask application
- `auth_middleware.py` - Authentication system
- `start_server.py` - Server startup script

---

**🎯 Quick Start**: `docker compose up -d` → Test: `curl -H "X-API-Key: sam-demo-key-123" https://tally-o.gavinlou.com/health` 