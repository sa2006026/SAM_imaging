#!/usr/bin/env python3
"""
Image Enhancement using Real-ESRGAN
This script enhances the resolution of GY_image.png using Real-ESRGAN models.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import urllib.request
import zipfile
import argparse
from pathlib import Path

def download_model(model_name="RealESRGAN_x4plus"):
    """Download Real-ESRGAN model if not present"""
    model_urls = {
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    }
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model...")
        try:
            urllib.request.urlretrieve(model_urls[model_name], model_path)
            print(f"Model downloaded successfully: {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    else:
        print(f"Model already exists: {model_path}")
    
    return model_path

def install_dependencies():
    """Install required dependencies"""
    try:
        import realesrgan
        print("Real-ESRGAN already installed")
        return True
    except ImportError:
        print("Installing Real-ESRGAN...")
        os.system("pip install realesrgan")
        try:
            import realesrgan
            print("Real-ESRGAN installed successfully")
            return True
        except ImportError:
            print("Failed to install Real-ESRGAN. Trying alternative method...")
            return False

def enhance_with_realesrgan(input_path, output_path, scale=4, model_name="RealESRGAN_x4plus"):
    """Enhance image using Real-ESRGAN library"""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        print(f"Enhancing image: {input_path}")
        print(f"Scale factor: {scale}x")
        print(f"Model: {model_name}")
        
        # Download model
        model_path = download_model(model_name)
        if not model_path:
            return False
        
        # Determine model parameters based on model name
        if "x4plus" in model_name:
            scale = 4
            num_block = 23
        elif "x2plus" in model_name:
            scale = 2
            num_block = 23
        else:
            scale = 4
            num_block = 23
        
        # Initialize model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=num_block,
            num_grow_ch=32,
            scale=scale
        )
        
        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=400,  # Tile size for memory efficiency
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available()  # Use half precision if GPU available
        )
        
        # Read input image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return False
        
        print(f"Original image size: {img.shape[:2]}")
        
        # Enhance image
        print("Processing image...")
        enhanced_img, _ = upsampler.enhance(img, outscale=scale)
        
        print(f"Enhanced image size: {enhanced_img.shape[:2]}")
        
        # Save enhanced image
        cv2.imwrite(output_path, enhanced_img)
        print(f"Enhanced image saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error in Real-ESRGAN enhancement: {e}")
        return False

def enhance_with_opencv_sr(input_path, output_path, scale=4):
    """Fallback enhancement using OpenCV Super Resolution"""
    try:
        print("Using OpenCV Super Resolution as fallback...")
        
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return False
        
        print(f"Original image size: {img.shape[:2]}")
        
        # Use different interpolation methods
        height, width = img.shape[:2]
        new_height, new_width = height * scale, width * scale
        
        # INTER_CUBIC for better quality
        enhanced_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        print(f"Enhanced image size: {enhanced_img.shape[:2]}")
        
        # Save enhanced image
        cv2.imwrite(output_path, enhanced_img)
        print(f"Enhanced image saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error in OpenCV enhancement: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Enhance image resolution using ESRGAN")
    parser.add_argument("--input", default="image/Test/GY_image.png", help="Input image path")
    parser.add_argument("--output", default="image/Output/GY_image_enhanced.png", help="Output image path")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4], help="Upscaling factor")
    parser.add_argument("--model", default="RealESRGAN_x4plus", 
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRNet_x4plus"],
                       help="Model to use")
    parser.add_argument("--fallback", action="store_true", help="Use OpenCV fallback method")
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("="*60)
    print("IMAGE ENHANCEMENT WITH ESRGAN")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Scale: {args.scale}x")
    print(f"Model: {args.model}")
    print("="*60)
    
    success = False
    
    if not args.fallback:
        # Try Real-ESRGAN first
        if install_dependencies():
            success = enhance_with_realesrgan(args.input, args.output, args.scale, args.model)
    
    if not success:
        print("\nFalling back to OpenCV Super Resolution...")
        success = enhance_with_opencv_sr(args.input, args.output, args.scale)
    
    if success:
        print("\n" + "="*60)
        print("ENHANCEMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Show file sizes
        input_size = os.path.getsize(args.input) / (1024 * 1024)
        output_size = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Input file size: {input_size:.1f} MB")
        print(f"Output file size: {output_size:.1f} MB")
        
        # Show image dimensions
        input_img = cv2.imread(args.input)
        output_img = cv2.imread(args.output)
        print(f"Input dimensions: {input_img.shape[1]}x{input_img.shape[0]}")
        print(f"Output dimensions: {output_img.shape[1]}x{output_img.shape[0]}")
        print(f"Scale achieved: {output_img.shape[1]/input_img.shape[1]:.1f}x")
        
    else:
        print("\nERROR: Enhancement failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 