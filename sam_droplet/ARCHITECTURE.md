# SAM Droplet Segmentation - System Architecture

## 🏗️ Architecture Overview

The SAM Droplet Segmentation service is now a **containerized, production-ready system** with multiple layers of security, monitoring, and scalability.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Internet/Users                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTPS Requests
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Cloudflare Edge Network                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   DDoS Shield   │  │  SSL/TLS Cert   │  │   Global CDN    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Encrypted Tunnel
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Local Server Environment                      │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                Cloudflare Tunnel                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐               │  │
│  │  │  config.yml     │  │  credentials    │               │  │
│  │  │  tally-o.gavin  │  │  authentication │               │  │
│  │  └─────────────────┘  └─────────────────┘               │  │
│  └───────────────────┬───────────────────────────────────────┘  │
│                      │ HTTP to localhost:9487                   │
│                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Docker Container Network                     │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │           SAM Application Container                 │  │  │
│  │  │                                                     │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐          │  │  │
│  │  │  │   Gunicorn      │  │  Flask App      │          │  │  │
│  │  │  │   WSGI Server   │  │  (app.py)       │          │  │  │
│  │  │  │   Port: 9487    │  │                 │          │  │  │
│  │  │  └─────────────────┘  └─────────────────┘          │  │  │
│  │  │                                                     │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐          │  │  │
│  │  │  │ Auth Middleware │  │  SAM Model      │          │  │  │
│  │  │  │ API Keys        │  │  mobile_sam.pt  │          │  │  │
│  │  │  │ Rate Limiting   │  │  Segmentation   │          │  │  │
│  │  │  └─────────────────┘  └─────────────────┘          │  │  │
│  │  │                                                     │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐          │  │  │
│  │  │  │   Health Check  │  │    Logging      │          │  │  │
│  │  │  │   Monitoring    │  │    & Metrics    │          │  │  │
│  │  │  └─────────────────┘  └─────────────────┘          │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow

### 1. **External Request**
```
User → https://tally-o.gavinlou.com/segment
├── Headers: X-API-Key: sam-demo-key-123
├── Body: { "image": "data:image/png;base64,..." }
└── Method: POST
```

### 2. **Cloudflare Processing**
- **DNS Resolution**: `tally-o.gavinlou.com` → Cloudflare Edge
- **SSL Termination**: HTTPS certificate validation
- **Security Filtering**: DDoS protection, bot detection
- **CDN Caching**: Static content caching (if applicable)

### 3. **Tunnel Routing**
- **Tunnel Connection**: Encrypted connection to local server
- **Request Forwarding**: HTTP request to `localhost:9487`
- **Response Streaming**: Real-time response back through tunnel

### 4. **Container Processing**
- **Load Balancer**: Gunicorn distributes to worker processes
- **Authentication**: API key validation via middleware
- **Rate Limiting**: Request throttling per API key
- **Application Logic**: SAM model inference
- **Response Generation**: JSON response with segmentation masks

## 🏛️ Component Architecture

### **Container Layer**
```yaml
sam-droplet-app:
  image: sam_droplet-sam-app
  ports: ["9487:9487"]
  environment:
    - API_KEYS=sam-demo-key-123:Demo:50
    - ADMIN_API_KEY=admin-secret-key
    - SAM_MODEL_PATH=model/mobile_sam.pt
  volumes:
    - ./model:/app/model:ro
    - ./logs:/app/logs
  healthcheck:
    test: curl -f http://localhost:9487/health
    interval: 30s
```

### **Application Layer**
```python
Flask Application (app.py)
├── Authentication Middleware (auth_middleware.py)
│   ├── API Key Validation
│   ├── Rate Limiting
│   └── Usage Tracking
├── Core Endpoints
│   ├── /health (Public)
│   ├── /segment (Protected)
│   ├── /segment_file (Protected)
│   └── /analyze_image (Protected)
├── Admin Endpoints
│   ├── /admin/stats (Admin Only)
│   └── /admin/health (Admin Only)
└── SAM Model Integration
    ├── MobileSAM Loading
    ├── Image Processing
    └── Mask Generation
```

### **Security Layer**
```
┌─────────────────────────────────────────┐
│            Security Stack               │
├─────────────────────────────────────────┤
│ 1. Cloudflare DDoS Protection          │
│ 2. SSL/TLS Encryption                  │
│ 3. API Key Authentication              │
│ 4. Rate Limiting (Per Key)             │
│ 5. Request Size Validation             │
│ 6. File Type Validation                │
│ 7. Container Isolation                 │
│ 8. Network Segmentation                │
└─────────────────────────────────────────┘
```

## 🔑 Authentication Flow

### **API Key Validation**
```python
@require_api_key
def segment_image():
    # 1. Extract API key from X-API-Key header
    api_key = request.headers.get('X-API-Key')
    
    # 2. Validate against configured keys
    if api_key not in VALID_API_KEYS:
        return 401 Unauthorized
    
    # 3. Check if key is enabled
    if not key_info['enabled']:
        return 401 Disabled
    
    # 4. Rate limiting check
    if exceeded_rate_limit(api_key):
        return 429 Too Many Requests
    
    # 5. Process request
    return process_segmentation()
```

### **Rate Limiting Algorithm**
```python
def is_allowed(api_key, limit_per_hour):
    now = time.time()
    hour_ago = now - 3600
    
    # Clean old requests
    requests[api_key] = [req for req in requests[api_key] if req > hour_ago]
    
    # Check limit
    if len(requests[api_key]) >= limit_per_hour:
        return False
    
    # Record request
    requests[api_key].append(now)
    return True
```

## 🚀 Deployment Architecture

### **Development Environment**
```bash
# Single-node development
docker compose up -d

Components:
├── sam-app (1 instance)
├── Local Cloudflare tunnel
└── Local model storage
```

### **Production Environment**
```bash
# Scalable production deployment
docker compose -f docker-compose.gpu.yml up -d --scale sam-app=3

Components:
├── sam-app (3 instances + load balancer)
├── GPU acceleration
├── Persistent volume storage
├── Enhanced monitoring
└── Production tunnel configuration
```

### **High Availability Setup**
```bash
# Future: Kubernetes deployment
kubectl apply -f k8s/

Components:
├── Multiple pods across nodes
├── Horizontal Pod Autoscaler
├── Persistent Volume Claims
├── Service mesh (Istio)
├── Ingress controller
└── Monitoring stack (Prometheus/Grafana)
```

## 📊 Data Flow

### **Image Segmentation Process**
```
Input Image → Preprocessing → SAM Model → Post-processing → Response
     │              │            │            │              │
     ▼              ▼            ▼            ▼              ▼
Base64 Decode → Validation → Inference → Mask Filter → JSON Response
     │              │            │            │              │
File Upload    Size Check   GPU/CPU Proc   Statistics    API Response
   (50MB)       Format       (Model)       Analytics     (Masks)
```

### **Model Loading Strategy**
```python
# Lazy loading with caching
def initialize_sam():
    global sam_model, mask_generator
    
    if sam_model is None:
        # Load model on first request
        model_path = "model/mobile_sam.pt"
        sam_model = sam_model_registry["vit_t"](checkpoint=model_path)
        sam_model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        mask_generator = SamAutomaticMaskGenerator(sam_model)
```

## 🔧 Configuration Management

### **Environment-Based Configuration**
```env
# Runtime Configuration
API_KEYS=key1:name1:limit1,key2:name2:limit2
ADMIN_API_KEY=secure-admin-key
SAM_MODEL_PATH=model/mobile_sam.pt
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50

# Performance Tuning
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0

# Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=9487
FLASK_DEBUG=false
```

### **Docker Compose Configuration**
```yaml
# Multi-service orchestration
services:
  sam-app:
    build: .
    environment: ${ENV_VARS}
    volumes: ${VOLUME_MOUNTS}
    healthcheck: ${HEALTH_CONFIG}
    
  cloudflare-tunnel:
    build: 
      dockerfile: Dockerfile.tunnel
    depends_on:
      sam-app: { condition: service_healthy }
```

## 📈 Monitoring & Observability

### **Health Check Hierarchy**
```
┌─────────────────────────────────────────┐
│           Health Monitoring             │
├─────────────────────────────────────────┤
│ 1. Container Health (Docker)           │
│ 2. Application Health (/health)        │
│ 3. Model Health (SAM loading)          │
│ 4. Tunnel Health (Cloudflare)          │
│ 5. API Health (Response times)         │
└─────────────────────────────────────────┘
```

### **Metrics Collection**
```python
# Usage tracking per endpoint
def track_usage():
    metrics = {
        'api_key': request.headers.get('X-API-Key'),
        'endpoint': request.endpoint,
        'processing_time': time.time() - start_time,
        'status_code': response.status_code,
        'timestamp': datetime.utcnow()
    }
    # Store for analytics
```

## 🛡️ Security Architecture

### **Defense in Depth**
```
Internet → Cloudflare → Tunnel → Container → Application → Model
    │          │          │         │            │          │
    ▼          ▼          ▼         ▼            ▼          ▼
DDoS Prot   SSL Term   Encrypted  Isolation   Auth/Auth   Input Val
Bot Prot    WAF        Tunnel     Resource    Rate Limit  Sanitize
Geo Block   Cache      Access     Limits      API Keys    Model Sec
```

### **Network Security**
- **No Direct Exposure**: Only tunnel connection to external network
- **Container Isolation**: Application runs in isolated Docker network
- **Encrypted Communication**: All external traffic via HTTPS tunnel
- **Access Control**: API key-based authentication with granular permissions

## 🔄 Scaling Strategy

### **Horizontal Scaling**
```bash
# Scale application containers
docker compose up -d --scale sam-app=5

# Benefits:
- Increased request throughput
- Load distribution across instances
- Fault tolerance (instance failure)
- Resource utilization optimization
```

### **Vertical Scaling**
```yaml
# Resource allocation per container
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

### **Auto-scaling (Future)**
```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sam-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sam-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

This architecture provides a **production-ready, scalable, and secure** SAM segmentation service with comprehensive monitoring, authentication, and deployment capabilities. 