#!/usr/bin/env python3
"""
SAM Droplet Advanced Analysis Server - Two-Group Mask Clustering
Based on mask_size_grouping.py functionality for intelligent droplet classification.

This server provides advanced mask analysis with K-means clustering to automatically
separate masks into two distinct groups based on multiple morphological features.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        ('flask', 'flask'), 
        ('flask_cors', 'flask-cors'), 
        ('numpy', 'numpy'), 
        ('cv2', 'opencv-python'), 
        ('torch', 'torch'), 
        ('torchvision', 'torchvision'), 
        ('segment_anything', 'segment-anything'), 
        ('PIL', 'pillow'), 
        ('sklearn', 'scikit-learn'), 
        ('matplotlib', 'matplotlib'), 
        ('scipy', 'scipy'), 
        ('skimage', 'scikit-image')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements_server.txt")
        
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("   (Virtual environment detected - packages should install correctly)")
        else:
            print("   ⚠️  Consider using a virtual environment!")
        
        return False
    
    print("✅ All required packages are installed")
    return True

def check_models():
    """Check if SAM models are available."""
    model_dir = Path(__file__).parent / "model"
    model_files = [
        "sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth", 
        "sam_vit_b_01ec64.pth",
        "mobile_sam.pt"
    ]
    
    available_models = []
    for model_file in model_files:
        if (model_dir / model_file).exists():
            available_models.append(model_file)
    
    if not available_models:
        print("⚠️  No SAM models found in model/ directory")
        print("📥 Server will automatically download SAM vit_b model on first use")
        return False
    
    print(f"✅ Found {len(available_models)} SAM model(s):")
    for model in available_models:
        print(f"   - {model}")
    return True

def setup_environment():
    """Set up environment variables for development."""
    env_vars = {
        'API_KEYS': 'sam-demo-key-123:Demo:50,sam-admin-key-456:Admin:200',
        'ADMIN_API_KEY': 'admin-secret-key-change-me',
        'HEALTH_CHECK_KEY': 'health-check-key-123'
    }
    
    for key, value in env_vars.items():
        os.environ.setdefault(key, value)
    
    print("✅ Environment variables configured")

def display_startup_info():
    """Display startup information and URLs."""
    print("\n" + "="*70)
    print("🔬 SAM DROPLET ADVANCED ANALYSIS - TWO-GROUP CLUSTERING")
    print("="*70)
    print("🎯 INTELLIGENT MASK CLASSIFICATION BASED ON mask_size_grouping.py")
    print("="*70)
    
    print("🌐 Server URL: http://localhost:9487")
    print("\n📱 WEB INTERFACES:")
    print("   🔬 Advanced Analysis:  http://localhost:9487/static/advanced_analysis.html")
    print("   📊 Basic Interface:    http://localhost:9487/")
    
    print("\n🔑 API AUTHENTICATION:")
    print("   • Demo Key:    sam-demo-key-123")
    print("   • Admin Key:   sam-admin-key-456")
    
    print("\n🎯 TWO-GROUP CLUSTERING FEATURES:")
    print("   ✨ K-means clustering with 6 morphological features")
    print("   📏 Area, bbox dimensions, aspect ratio analysis")
    print("   🎪 Stability score and circularity metrics")
    print("   🔍 Automatic droplet size classification")
    print("   📈 Statistical analysis per group")
    
    print("\n🛠️ ADVANCED FILTERING OPTIONS:")
    print("   🏠 Edge proximity filtering (default: 5px threshold)")
    print("   ⭕ Circularity filtering (default: min 0.53)")
    print("   🎈 Multi-blob distance filtering (default: max 50px)")
    print("   📊 Comprehensive quality assessment")
    
    print("\n📋 API ENDPOINTS:")
    print("   • POST /advanced_analysis - Two-group mask clustering")
    print("   • POST /download_analysis - Complete analysis package")
    print("   • POST /segment - Basic SAM segmentation")
    print("   • GET  /health - Server health check")
    
    print("\n📦 OUTPUT FORMATS:")
    print("   📊 Interactive visualizations (cluster overview)")
    print("   📈 Statistical summaries (per-group analysis)")
    print("   💾 CSV export (all masks with group assignments)")
    print("   🗜️ ZIP download (complete analysis package)")
    
    print("\n🔄 CLUSTERING WORKFLOW:")
    print("   1️⃣ SAM generates initial masks")
    print("   2️⃣ Quality filtering (edge/circularity/blob)")
    print("   3️⃣ Feature extraction (6 morphological features)")
    print("   4️⃣ K-means clustering into 2 groups")
    print("   5️⃣ Statistical analysis and visualization")
    
    print("\n📚 DOCUMENTATION:")
    print("   📖 README_ADVANCED_ANALYSIS.md - Complete guide")
    print("   🚀 QUICK_REFERENCE.md - API reference")
    
    print("="*70)
    print("🎊 READY FOR INTELLIGENT DROPLET CLASSIFICATION!")
    print("="*70)

def display_clustering_info():
    """Display detailed information about the clustering algorithm."""
    print("\n" + "🧠 INTELLIGENT TWO-GROUP CLUSTERING ALGORITHM")
    print("─" * 60)
    
    print("\n📊 FEATURE EXTRACTION (6 dimensions):")
    print("   1. 📏 Area - Total mask area in pixels")
    print("   2. 📐 Bbox Width - Bounding box width")
    print("   3. 📐 Bbox Height - Bounding box height") 
    print("   4. 📊 Aspect Ratio - Width/height ratio")
    print("   5. ⭐ Stability Score - SAM confidence score")
    print("   6. ⭕ Circularity - Shape circularity (0-1)")
    
    print("\n🔬 PREPROCESSING PIPELINE:")
    print("   🏠 Edge Filter: Remove incomplete masks touching borders")
    print("   ⭕ Circularity Filter: Remove non-circular shapes")
    print("   🎈 Blob Filter: Remove masks with distant components")
    
    print("\n🎯 K-MEANS CLUSTERING:")
    print("   📈 Log transformation for area normalization")
    print("   🔧 Feature standardization (StandardScaler)")
    print("   🎪 Automatic cluster balancing")
    print("   📊 Cluster quality assessment")
    
    print("\n📈 OUTPUT ANALYSIS:")
    print("   🏷️ Group 0: Typically smaller/more circular droplets")
    print("   🏷️ Group 1: Typically larger/varied shape droplets") 
    print("   📊 Per-group statistics (area, circularity, etc.)")
    print("   🎨 Color-coded visualizations")
    
    print("─" * 60)

def start_server():
    """Start the Flask development server."""
    print("\n🔄 INITIALIZING SERVER...")
    try:
        # Import and run the Flask app
        from app import app, initialize_sam
        
        # Initialize SAM model
        print("🔧 Loading SAM model for mask generation...")
        initialize_sam()
        
        print("🎊 Advanced Analysis Server Ready!")
        print("📱 Open http://localhost:9487/static/advanced_analysis.html to start")
        print("🔍 Upload an image to see intelligent two-group clustering in action!")
        print("\n🚀 Starting Flask server...")
        
        app.run(host='0.0.0.0', port=9487, debug=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("Thank you for using SAM Droplet Advanced Analysis!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Try: pip install -r requirements_server.txt")
        sys.exit(1)

def main():
    """Main startup routine."""
    print("🔬 SAM Droplet Advanced Analysis - Two-Group Clustering Server")
    print("Based on mask_size_grouping.py intelligent classification")
    print("="*70)
    
    # Check requirements
    print("🔍 Checking system requirements...")
    if not check_requirements():
        print("\n❌ Please install missing packages and try again")
        sys.exit(1)
    
    # Check models
    print("\n🔍 Checking SAM models...")
    model_available = check_models()
    if not model_available:
        print("⚠️  Continuing anyway - will auto-download on first use")
    
    # Setup environment
    print("\n🔧 Setting up environment...")
    setup_environment()
    
    # Display comprehensive info
    display_startup_info()
    display_clustering_info()
    
    # Ask for confirmation
    try:
        print("\n" + "="*70)
        input("⏸️  Press Enter to start the Two-Group Clustering Server (Ctrl+C to cancel)...")
        print("="*70)
    except KeyboardInterrupt:
        print("\n🚫 Startup cancelled")
        sys.exit(0)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main() 