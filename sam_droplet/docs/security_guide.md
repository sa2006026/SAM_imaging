# Security Guide for Internet-Exposed SAM Service

## Overview

When exposing your SAM Droplet Segmentation service to the internet via Cloudflare Tunnel, implementing proper security measures is crucial to protect your GPU resources and prevent abuse.

## Authentication Methods

### 1. API Key Authentication (Recommended for APIs)

#### Implementation

Add to your `app.py`:

```python
import os
import hashlib
import hmac
from functools import wraps
from flask import request, jsonify

# Store API keys (in production, use a database or environment variables)
VALID_API_KEYS = {
    "your-secret-key-123": {
        "name": "Main Client",
        "rate_limit": 100,  # requests per hour
        "enabled": True
    },
    "client-2-key-456": {
        "name": "Secondary Client", 
        "rate_limit": 50,
        "enabled": True
    }
}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if api_key not in VALID_API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
            
        if not VALID_API_KEYS[api_key]['enabled']:
            return jsonify({'error': 'API key disabled'}), 401
        
        # Store API key info for rate limiting
        request.api_key_info = VALID_API_KEYS[api_key]
        
        return f(*args, **kwargs)
    return decorated_function

# Apply to your endpoints
@app.route('/segment_file', methods=['POST'])
@require_api_key
def segment_uploaded_file():
    # ... existing implementation
    pass

@app.route('/segment', methods=['POST'])
@require_api_key
def segment_image():
    # ... existing implementation
    pass
```

#### Client Usage

```bash
curl -X POST http://your-domain.com/segment_file \
  -H "X-API-Key: your-secret-key-123" \
  -F "file=@image.jpg"
```

```python
# Python client
import requests

headers = {"X-API-Key": "your-secret-key-123"}
files = {"file": open("image.jpg", "rb")}

response = requests.post(
    "http://your-domain.com/segment_file",
    headers=headers,
    files=files
)
```

### 2. JWT Token Authentication (More Secure)

```python
import jwt
from datetime import datetime, timedelta

JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-key')
JWT_ALGORITHM = 'HS256'

def generate_token(user_id, expires_hours=24):
    """Generate a JWT token for a user."""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=expires_hours),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def require_jwt_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token required'}), 401
            
        if token.startswith('Bearer '):
            token = token[7:]
        
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            request.user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Token generation endpoint
@app.route('/auth/token', methods=['POST'])
def get_token():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Verify credentials (implement your logic here)
    if verify_credentials(username, password):
        token = generate_token(username)
        return jsonify({'token': token})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401
```

### 3. Rate Limiting Implementation

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis

# Initialize rate limiter
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_api_key():
    return request.headers.get('X-API-Key', get_remote_address())

limiter = Limiter(
    app,
    key_func=get_api_key,
    storage_uri="redis://localhost:6379"
)

# Apply different limits based on API key
def dynamic_rate_limit():
    api_key = request.headers.get('X-API-Key')
    if api_key and api_key in VALID_API_KEYS:
        return f"{VALID_API_KEYS[api_key]['rate_limit']} per hour"
    return "10 per hour"  # Default for unauthenticated requests

@app.route('/segment_file', methods=['POST'])
@require_api_key
@limiter.limit(dynamic_rate_limit)
def segment_uploaded_file():
    # ... implementation
    pass
```

## Usage Tracking and Monitoring

```python
import time
from collections import defaultdict

# Usage tracking
usage_stats = defaultdict(lambda: {'requests': 0, 'gpu_time': 0})

def track_usage(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key', 'anonymous')
        start_time = time.time()
        
        try:
            result = f(*args, **kwargs)
            gpu_time = time.time() - start_time
            
            # Track usage
            usage_stats[api_key]['requests'] += 1
            usage_stats[api_key]['gpu_time'] += gpu_time
            
            return result
        except Exception as e:
            # Still track failed requests
            usage_stats[api_key]['requests'] += 1
            raise
    
    return decorated_function

@app.route('/admin/usage', methods=['GET'])
@require_admin_key
def get_usage_stats():
    return jsonify(dict(usage_stats))
```

## Input Validation and Sanitization

```python
import magic
from PIL import Image
import io

def validate_image_upload(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check file size (limit to 50MB)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)     # Seek back to start
        
        if size > 50 * 1024 * 1024:
            return jsonify({'error': 'File too large (max 50MB)'}), 400
        
        # Check file type
        file_content = file.read(1024)
        file.seek(0)
        
        file_type = magic.from_buffer(file_content, mime=True)
        allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff']
        
        if file_type not in allowed_types:
            return jsonify({'error': f'Invalid file type: {file_type}'}), 400
        
        # Validate image can be opened
        try:
            image = Image.open(file)
            image.verify()
            file.seek(0)  # Reset for processing
        except Exception:
            return jsonify({'error': 'Invalid image file'}), 400
        
        return f(*args, **kwargs)
    
    return decorated_function
```

## Cloudflare-Specific Security Features

### 1. Cloudflare Access (Zero Trust)

Configure Cloudflare Access for additional security:

```yaml
# cloudflare-access-policy.yaml
name: "SAM API Access"
decision: "allow"
includes:
  - email_domain: ["yourdomain.com"]
  - ip: ["your.office.ip/32"]
excludes: []
require: []
session_duration: "24h"
```

### 2. Cloudflare WAF Rules

```javascript
// Custom WAF rule for API endpoints
(http.request.uri.path contains "/segment") and 
(not http.request.headers["x-api-key"][0] matches "^[a-zA-Z0-9-]{20,}$")
```

### 3. Rate Limiting at Cloudflare Level

```json
{
  "threshold": 10,
  "period": 60,
  "action": "challenge",
  "match": {
    "request": {
      "methods": ["POST"],
      "schemes": ["HTTPS"],
      "url": "yourdomain.com/segment*"
    }
  }
}
```

## Environment Configuration

Create a `.env` file for production:

```env
# Security
JWT_SECRET=your-super-secret-jwt-key-here
API_KEYS=key1:client1:100,key2:client2:50
ADMIN_API_KEY=admin-super-secret-key

# Rate limiting
REDIS_URL=redis://localhost:6379
DEFAULT_RATE_LIMIT=10
AUTHENTICATED_RATE_LIMIT=100

# Monitoring
ENABLE_USAGE_TRACKING=true
LOG_LEVEL=INFO

# Service limits
MAX_FILE_SIZE_MB=50
MAX_GPU_TIME_SECONDS=300
```

## Complete Secure App Configuration

```python
import os
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
JWT_SECRET = os.getenv('JWT_SECRET')
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is required")

# Initialize rate limiter
limiter = Limiter(
    app,
    key_func=lambda: request.headers.get('X-API-Key', get_remote_address()),
    default_limits=[os.getenv('DEFAULT_RATE_LIMIT', '10') + " per hour"]
)

# Apply security to all endpoints
@app.before_request
def log_request():
    api_key = request.headers.get('X-API-Key', 'None')
    logger.info(f"Request: {request.method} {request.path} - API Key: {api_key[:8]}...")

@app.after_request
def log_response(response):
    logger.info(f"Response: {response.status_code}")
    return response
```

## Client SDK with Authentication

Update your Python client:

```python
import requests
import json

class SecureSAMClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'X-API-Key': api_key,
            'User-Agent': 'SAM-Client/1.0'
        }
    
    def segment_file(self, image_path, filters=None):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if filters:
                data['filters'] = json.dumps(filters)
            
            response = requests.post(
                f"{self.base_url}/segment_file",
                headers=self.headers,
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 429:
                raise Exception("Rate limit exceeded")
            elif response.status_code == 401:
                raise Exception("Invalid API key")
            
            response.raise_for_status()
            return response.json()

# Usage
client = SecureSAMClient("https://yourdomain.com", "your-api-key-123")
result = client.segment_file("image.jpg", {"mean_min": 140})
```

## Deployment with Security

Update your Docker configuration:

```dockerfile
# Add security dependencies
RUN pip install flask-limiter redis pyjwt python-magic

# Run as non-root user
USER sam_user

# Health check with authentication
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f -H "X-API-Key: $HEALTH_CHECK_KEY" http://localhost:5000/health || exit 1
```

## Monitoring and Alerting

```python
# Add to your app.py
@app.route('/admin/metrics', methods=['GET'])
@require_admin_key
def get_metrics():
    return jsonify({
        'active_connections': get_active_connections(),
        'gpu_utilization': get_gpu_utilization(),
        'request_rate': get_request_rate(),
        'error_rate': get_error_rate(),
        'top_users': get_top_users()
    })

def get_gpu_utilization():
    if torch.cuda.is_available():
        return {
            'memory_used': torch.cuda.memory_allocated(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'utilization_percent': torch.cuda.utilization()
        }
    return None
```

## Recommendations

1. **Start with API Key authentication** - simplest to implement
2. **Enable Cloudflare rate limiting** as a first line of defense
3. **Monitor GPU usage** and set alerts for unusual activity
4. **Implement usage quotas** per API key
5. **Log all requests** for security auditing
6. **Use HTTPS only** (Cloudflare handles this)
7. **Regularly rotate API keys**
8. **Set up alerts** for high usage or error rates

This setup will protect your expensive GPU resources while still allowing legitimate access to your service! 