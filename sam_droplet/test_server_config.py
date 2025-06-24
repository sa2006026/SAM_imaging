#!/usr/bin/env python3
"""
Test script to verify that the SAM Droplet server is configured 
with the exact same parameters as mask_size_grouping.py
"""

import requests
import json
import base64
import sys
from pathlib import Path

def test_server_config():
    """Test that server configuration matches mask_size_grouping.py"""
    
    print("🔬 Testing SAM Droplet Server Configuration")
    print("="*50)
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:9487/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Server is running")
            print(f"   Model loaded: {health.get('model_loaded', False)}")
        else:
            print("❌ Server health check failed")
            return False
    except requests.RequestException as e:
        print("❌ Cannot connect to server - make sure it's running")
        print("   Run: python3 start_advanced_analysis.py")
        return False
    
    # Test that we can access the advanced analysis endpoint
    try:
        # Create a small test image (black square)
        import numpy as np
        from PIL import Image
        import io
        
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[20:80, 20:80] = [255, 255, 255]  # White square in center
        
        # Convert to base64
        pil_img = Image.fromarray(test_image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Test the advanced analysis with exact mask_size_grouping.py parameters
        test_data = {
            "image": f"data:image/png;base64,{img_base64}",
            "advanced_filters": {
                "edge_threshold": 5,
                "min_circularity": 0.53,
                "max_blob_distance": 80
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "sam-demo-key-123"
        }
        
        print("\n🧪 Testing advanced analysis endpoint...")
        response = requests.post(
            "http://localhost:9487/advanced_analysis", 
            json=test_data, 
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Advanced analysis endpoint working")
            
            if result.get('success'):
                analysis = result.get('analysis_results', {})
                filtering = analysis.get('filtering_criteria', {})
                clustering = analysis.get('clustering_info', {})
                
                print("\n📋 Configuration Verification:")
                print(f"   🏠 Edge threshold: {filtering.get('edge_threshold')} (expected: 5)")
                print(f"   ⭕ Min circularity: {filtering.get('min_circularity')} (expected: 0.53)")
                print(f"   🎈 Max blob distance: {filtering.get('max_blob_distance')} (expected: 80)")
                print(f"   🎯 Clustering method: {clustering.get('method')} (expected: K-means)")
                print(f"   📊 Number of clusters: {clustering.get('n_clusters')} (expected: 2)")
                
                # Verify parameters match
                params_match = (
                    filtering.get('edge_threshold') == 5 and
                    filtering.get('min_circularity') == 0.53 and
                    filtering.get('max_blob_distance') == 80 and
                    clustering.get('method') == 'K-means' and
                    clustering.get('n_clusters') == 2
                )
                
                if params_match:
                    print("\n🎉 SUCCESS: Server configuration matches mask_size_grouping.py!")
                    return True
                else:
                    print("\n⚠️  WARNING: Some parameters don't match expected values")
                    return False
            else:
                print("❌ Analysis failed")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("❌ Advanced analysis endpoint failed")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing advanced analysis: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the server"""
    print("\n" + "="*50)
    print("🚀 USAGE EXAMPLES")
    print("="*50)
    
    print("🌐 Web Interface:")
    print("   http://localhost:9487/static/advanced_analysis.html")
    
    print("\n📝 Python API Example:")
    print("""
import requests
import base64

# Load your image
with open('your_image.png', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

# Send to server with mask_size_grouping.py parameters
data = {
    "image": f"data:image/png;base64,{img_data}",
    "advanced_filters": {
        "edge_threshold": 5,
        "min_circularity": 0.53,
        "max_blob_distance": 80
    }
}

response = requests.post(
    "http://localhost:9487/advanced_analysis",
    json=data,
    headers={"X-API-Key": "sam-demo-key-123"}
)

result = response.json()
print(f"Found {len(result['csv_data'])} masks")
print(f"Cluster 0: {result['analysis_results']['clustering_info']['cluster_sizes'][0]} masks")
print(f"Cluster 1: {result['analysis_results']['clustering_info']['cluster_sizes'][1]} masks")
""")

if __name__ == "__main__":
    success = test_server_config()
    
    if success:
        show_usage_examples()
        print("\n🎊 Your mask_size_grouping.py is now fully hosted as a web server!")
    else:
        print("\n❌ Configuration test failed. Please check the server setup.")
        sys.exit(1) 