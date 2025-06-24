#!/usr/bin/env python3
"""
Simple launcher to enhance GY_image.png using ESRGAN
"""

import subprocess
import sys
import os

def main():
    input_file = "image/Test/GY_image.png"
    output_file = "image/Output/GY_image_enhanced_4x.png"
    
    print("🎯 GY Image Enhancement with ESRGAN")
    print("="*50)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} not found!")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"📥 Input: {input_file}")
    print(f"📤 Output: {output_file}")
    print(f"🔍 Scale: 4x enhancement")
    print(f"🤖 Model: RealESRGAN_x4plus")
    print("="*50)
    
    # Run the enhancement script
    try:
        cmd = [
            sys.executable, "enhance_image_esrgan.py",
            "--input", input_file,
            "--output", output_file,
            "--scale", "4",
            "--model", "RealESRGAN_x4plus"
        ]
        
        print("🚀 Starting enhancement process...")
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ Enhancement completed successfully!")
            if os.path.exists(output_file):
                print(f"📁 Enhanced image saved to: {output_file}")
            else:
                print("⚠️  Warning: Output file not found, but process completed.")
        else:
            print(f"\n❌ Enhancement failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running enhancement: {e}")

if __name__ == "__main__":
    main() 