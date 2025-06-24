"""
Authentication middleware for SAM Droplet Segmentation service.
Simple API key authentication with rate limiting.
"""

import os
import time
import json
from functools import wraps
from collections import defaultdict, deque
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)

# Simple in-memory storage (for production, use Redis or database)
class SimpleRateLimit:
    def __init__(self):
        self.requests = defaultdict(deque)
    
    def is_allowed(self, key, limit_per_hour=100):
        now = time.time()
        hour_ago = now - 3600
        
        # Clean old requests
        while self.requests[key] and self.requests[key][0] < hour_ago:
            self.requests[key].popleft()
        
        # Check if under limit
        if len(self.requests[key]) >= limit_per_hour:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Initialize rate limiter
rate_limiter = SimpleRateLimit()

# Load API keys from environment or use defaults
def load_api_keys():
    """Load API keys from environment variable or return defaults."""
    api_keys_env = os.getenv('API_KEYS', '')
    
    if api_keys_env:
        # Format: "key1:name1:limit1,key2:name2:limit2"
        keys = {}
        for key_config in api_keys_env.split(','):
            parts = key_config.split(':')
            if len(parts) >= 2:
                key = parts[0]
                name = parts[1]
                limit = int(parts[2]) if len(parts) > 2 else 100
                keys[key] = {"name": name, "rate_limit": limit, "enabled": True}
        return keys
    
    # Default keys for development (change these!)
    return {
        "sam-demo-key-123": {
            "name": "Demo Client",
            "rate_limit": 50,
            "enabled": True
        },
        "sam-admin-key-456": {
            "name": "Admin Client", 
            "rate_limit": 200,
            "enabled": True
        }
    }

VALID_API_KEYS = load_api_keys()

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(f"Request without API key from {request.remote_addr}")
            return jsonify({
                'error': 'API key required',
                'message': 'Include X-API-Key header with your request'
            }), 401
        
        # Validate API key
        if api_key not in VALID_API_KEYS:
            logger.warning(f"Invalid API key attempt: {api_key[:8]}... from {request.remote_addr}")
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 401
            
        key_info = VALID_API_KEYS[api_key]
        
        # Check if key is enabled
        if not key_info.get('enabled', True):
            logger.warning(f"Disabled API key used: {api_key[:8]}...")
            return jsonify({
                'error': 'API key disabled',
                'message': 'This API key has been disabled'
            }), 401
        
        # Rate limiting
        if not rate_limiter.is_allowed(api_key, key_info['rate_limit']):
            logger.warning(f"Rate limit exceeded for API key: {api_key[:8]}...")
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': f'You have exceeded the rate limit of {key_info["rate_limit"]} requests per hour'
            }), 429
        
        # Store API key info for use in the endpoint
        request.api_key_info = key_info
        
        # Log successful authentication
        logger.info(f"Authenticated request from {key_info['name']} ({api_key[:8]}...)")
        
        return f(*args, **kwargs)
    
    return decorated_function

def validate_file_upload(f):
    """Decorator to validate file uploads."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size (50MB limit)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)     # Seek back to start
        
        max_size = 50 * 1024 * 1024  # 50MB
        if size > max_size:
            return jsonify({
                'error': 'File too large',
                'message': f'File size ({size/1024/1024:.1f}MB) exceeds limit ({max_size/1024/1024}MB)'
            }), 400
        
        # Basic file type check
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp'}
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Please upload an image file (JPG, PNG, WebP, TIFF, BMP)'
            }), 400
        
        return f(*args, **kwargs)
    
    return decorated_function

def track_usage(f):
    """Decorator to track API usage."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key', 'anonymous')
        start_time = time.time()
        
        try:
            result = f(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Log usage
            logger.info(f"Request completed - API Key: {api_key[:8]}..., "
                       f"Processing time: {processing_time:.2f}s, "
                       f"Endpoint: {request.endpoint}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request failed - API Key: {api_key[:8]}..., "
                        f"Processing time: {processing_time:.2f}s, "
                        f"Error: {str(e)}")
            raise
    
    return decorated_function

# Admin functions
def require_admin_key(f):
    """Decorator to require admin API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_key = os.getenv('ADMIN_API_KEY', 'admin-secret-key-change-me')
        provided_key = request.headers.get('X-Admin-Key')
        
        if not provided_key or provided_key != admin_key:
            return jsonify({'error': 'Admin access required'}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

def get_usage_stats():
    """Get current usage statistics."""
    stats = {}
    for key, info in VALID_API_KEYS.items():
        current_requests = len([
            req for req in rate_limiter.requests[key] 
            if req > time.time() - 3600
        ])
        stats[key[:8] + "..."] = {
            'name': info['name'],
            'requests_last_hour': current_requests,
            'rate_limit': info['rate_limit'],
            'enabled': info['enabled']
        }
    return stats

# Example usage:
"""
To use this in your app.py, add these imports at the top:

from auth_middleware import require_api_key, validate_file_upload, track_usage

Then decorate your endpoints:

@app.route('/segment_file', methods=['POST'])
@require_api_key
@validate_file_upload  
@track_usage
def segment_uploaded_file():
    # ... your existing implementation
    pass

@app.route('/segment', methods=['POST'])
@require_api_key
@track_usage  
def segment_image():
    # ... your existing implementation
    pass

# Keep health check open (or add separate health key)
@app.route('/health', methods=['GET'])
def health_check():
    # ... your existing implementation
    pass

# Add admin endpoint for monitoring
@app.route('/admin/stats', methods=['GET'])
@require_admin_key
def admin_stats():
    return jsonify(get_usage_stats())
""" 