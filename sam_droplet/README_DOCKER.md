# SAM Droplet Segmentation - Docker Setup

This guide explains how to run the SAM Droplet Segmentation service using Docker and Docker Compose with Cloudflare Tunnel integration.

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed
2. **SAM Model File**: Place `mobile_sam.pt` in the `model/` directory
3. **Cloudflare Tunnel**: Configured tunnel credentials in `~/.cloudflared/`

### Basic Setup (CPU)

```bash
# Clone and navigate to the project
cd sam_droplet

# Create environment file
cp .dockerenv.example .env

# Edit your environment variables
nano .env

# Build and start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

### GPU Setup (NVIDIA)

For GPU acceleration (requires NVIDIA Docker runtime):

```bash
# Install NVIDIA Docker runtime first
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Use GPU compose file
docker-compose -f docker-compose.gpu.yml up -d
```

## 📋 Configuration

### Environment Variables

Copy `.dockerenv.example` to `.env` and configure:

```env
# Security
API_KEYS=your-prod-key:MainClient:100,backup-key:BackupClient:50
ADMIN_API_KEY=your-admin-secret-key
HEALTH_CHECK_KEY=your-health-key

# Performance
OMP_NUM_THREADS=4
MAX_FILE_SIZE_MB=50
LOG_LEVEL=INFO

# GPU (if using GPU setup)
CUDA_VISIBLE_DEVICES=0
```

### Required Files

Ensure these files exist:

- `model/mobile_sam.pt` - SAM model file
- `config.yml` - Cloudflare tunnel configuration
- `~/.cloudflared/` - Tunnel credentials directory

## 🔧 Usage

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start only the app (without tunnel)
docker-compose up -d sam-app

# View logs
docker-compose logs -f sam-app
docker-compose logs -f cloudflare-tunnel
```

### Testing the API

Once running, test the service:

```bash
# Health check
curl https://tally-o.gavinlou.com/health

# With API key
curl -H "X-API-Key: sam-demo-key-123" https://tally-o.gavinlou.com/health

# Upload and segment image
curl -X POST \
  -H "X-API-Key: sam-demo-key-123" \
  -F "file=@test_image.jpg" \
  https://tally-o.gavinlou.com/segment_file
```

### Monitoring

```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Logs
docker-compose logs -f

# Health checks
docker-compose exec sam-app curl http://localhost:9487/health
```

## 🔄 Management

### Updating

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Scaling

```bash
# Scale the app service
docker-compose up -d --scale sam-app=3

# Use load balancer for multiple instances
# (requires additional nginx/traefik setup)
```

### Backup

```bash
# Backup configuration
tar -czf sam-backup-$(date +%Y%m%d).tar.gz \
  config.yml .env model/ ~/.cloudflared/

# Backup logs
docker-compose exec sam-app tar -czf /app/logs/backup.tar.gz /app/logs/
```

## 🐛 Troubleshooting

### Common Issues

1. **503 Errors from Tunnel**
   ```bash
   # Check tunnel config
   docker-compose logs cloudflare-tunnel
   
   # Verify app is healthy
   docker-compose exec sam-app curl http://localhost:9487/health
   ```

2. **Model Not Found**
   ```bash
   # Check model file
   docker-compose exec sam-app ls -la /app/model/
   
   # Download model if missing
   # See main README for model download instructions
   ```

3. **GPU Not Detected**
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   
   # Verify GPU in container
   docker-compose exec sam-app python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Permission Issues**
   ```bash
   # Fix log directory permissions
   sudo chown -R $USER:$USER logs/
   chmod 755 logs/
   ```

### Performance Tuning

```yaml
# In docker-compose.yml, adjust resources:
services:
  sam-app:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Security

```bash
# Generate secure API keys
openssl rand -hex 32

# Use secrets for production
docker secret create api_keys api_keys.txt
```

## 📊 Production Deployment

For production deployments:

1. **Use secure API keys** - Generate with `openssl rand -hex 32`
2. **Enable HTTPS** - Cloudflare tunnel provides this automatically
3. **Monitor resources** - Use `docker stats` and log monitoring
4. **Backup regularly** - Model files and configuration
5. **Update security** - Keep Docker images updated

### Docker Swarm

For high availability:

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml sam-stack

# Scale services
docker service scale sam-stack_sam-app=3
```

## 📚 API Documentation

See the main documentation for API endpoints and usage examples:
- [API Reference](docs/api_reference.md)
- [Security Guide](docs/security_guide.md)
- [Deployment Guide](docs/deployment_guide.md) 