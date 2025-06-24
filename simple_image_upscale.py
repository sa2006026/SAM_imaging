#!/usr/bin/env python3
"""
Simple Image Upscaling using PIL (Pillow)
This is a basic demonstration using only PIL/Pillow for image enhancement.
For best results, use the full ESRGAN script after installing dependencies.
"""

import os
from PIL import Image, ImageFilter, ImageEnhance
import argparse

def enhance_image_simple(input_path, output_path, scale=4, method='lanczos'):
    """
    Simple image enhancement using PIL
    """
    print(f"📥 Loading image: {input_path}")
    
    try:
        # Open the image
        with Image.open(input_path) as img:
            original_size = img.size
            print(f"📏 Original size: {original_size[0]}x{original_size[1]}")
            
            # Calculate new size
            new_width = original_size[0] * scale
            new_height = original_size[1] * scale
            new_size = (new_width, new_height)
            
            print(f"🎯 Target size: {new_size[0]}x{new_size[1]} ({scale}x upscaling)")
            
            # Choose resampling method
            resampling_methods = {
                'nearest': Image.NEAREST,
                'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS
            }
            
            resample = resampling_methods.get(method, Image.LANCZOS)
            print(f"🔧 Using {method.upper()} resampling")
            
            # Resize the image
            print("🚀 Processing...")
            enhanced_img = img.resize(new_size, resample)
            
            # Apply additional enhancements
            print("✨ Applying enhancements...")
            
            # Slight sharpening
            enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
            
            # Slight contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced_img)
            enhanced_img = enhancer.enhance(1.05)
            
            # Save the result
            enhanced_img.save(output_path, quality=95, optimize=True)
            print(f"💾 Enhanced image saved: {output_path}")
            
            # Show file size comparison
            original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            enhanced_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"\n📊 Results:")
            print(f"   Original: {original_size[0]}x{original_size[1]} ({original_size_mb:.1f} MB)")
            print(f"   Enhanced: {new_size[0]}x{new_size[1]} ({enhanced_size_mb:.1f} MB)")
            print(f"   Scale achieved: {scale}x")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple image upscaling using PIL")
    parser.add_argument("--input", default="image/Test/GY_image.png", help="Input image path")
    parser.add_argument("--output", default="image/Output/GY_image_upscaled_simple.png", help="Output image path")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscaling factor")
    parser.add_argument("--method", default="lanczos", 
                       choices=["nearest", "bilinear", "bicubic", "lanczos"],
                       help="Resampling method")
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} does not exist")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("🎨 Simple Image Upscaling")
    print("="*50)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"📈 Scale: {args.scale}x")
    print(f"⚙️  Method: {args.method}")
    print("="*50)
    
    success = enhance_image_simple(args.input, args.output, args.scale, args.method)
    
    if success:
        print("\n✅ Simple upscaling completed!")
        print("\n💡 Note: For best AI-enhanced results, use the full ESRGAN script:")
        print("   python3 enhance_image_esrgan.py")
    else:
        print("\n❌ Simple upscaling failed!")

if __name__ == "__main__":
    main() 