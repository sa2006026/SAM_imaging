# Image Enhancement with ESRGAN - Setup Guide

## 🎯 Goal
Enhance the resolution of `image/Test/GY_image.png` using Real-ESRGAN for better quality upscaling.

## 📋 Prerequisites

### 1. Install Python Dependencies
```bash
pip install opencv-python torch torchvision pillow numpy realesrgan basicsr
```

Or install from requirements file:
```bash
pip install -r requirements_esrgan.txt
```

### 2. System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster processing)
- At least 4GB RAM
- 2GB free disk space for models

## 🚀 Quick Start

### Option 1: Simple Launcher (Recommended)
```bash
python3 enhance_gy_image.py
```

### Option 2: Full Control
```bash
python3 enhance_image_esrgan.py --input image/Test/GY_image.png --output image/Output/GY_image_enhanced_4x.png --scale 4
```

### Option 3: Fallback Method (if ESRGAN fails)
```bash
python3 enhance_image_esrgan.py --fallback --scale 4
```

## 🎛️ Available Options

### Scale Factors
- `--scale 2`: 2x upscaling (faster, smaller file)
- `--scale 4`: 4x upscaling (slower, larger file, better quality)

### Models Available
- `RealESRGAN_x4plus`: Best for general photos (default)
- `RealESRGAN_x2plus`: For 2x upscaling
- `RealESRNet_x4plus`: Alternative 4x model

### Example Commands
```bash
# 2x enhancement with x2plus model
python3 enhance_image_esrgan.py --scale 2 --model RealESRGAN_x2plus

# 4x enhancement with custom output path
python3 enhance_image_esrgan.py --scale 4 --output image/Output/GY_super_resolution.png

# Force use OpenCV if ESRGAN fails
python3 enhance_image_esrgan.py --fallback
```

## 📁 Expected Output

### Input
- File: `image/Test/GY_image.png` (1.6MB)
- Dimensions: Original size

### Output
- File: `image/Output/GY_image_enhanced_4x.png`
- Dimensions: 4x larger (width × 4, height × 4)
- Quality: Significantly enhanced with AI-based upscaling

## 🔧 What the Script Does

1. **Model Download**: Automatically downloads Real-ESRGAN model (~100MB) on first run
2. **Image Processing**: Uses advanced AI to enhance image resolution
3. **Memory Management**: Processes in tiles to handle large images
4. **Fallback Support**: Uses OpenCV if ESRGAN fails
5. **Progress Reporting**: Shows detailed progress and statistics

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install opencv-python realesrgan basicsr
   ```

2. **CUDA Out of Memory**
   - Use `--scale 2` instead of `--scale 4`
   - Close other applications using GPU

3. **Model Download Fails**
   - Check internet connection
   - Manually download from GitHub releases

4. **Low Quality Results**
   - Try different models (`--model RealESRNet_x4plus`)
   - Use smaller scale first (`--scale 2`)

### Fallback Method
If Real-ESRGAN fails, the script automatically falls back to OpenCV's cubic interpolation, which still provides decent results.

## 📊 Performance Expectations

### Processing Time (approximate)
- **CPU Only**: 2-5 minutes for 4x upscaling
- **GPU (CUDA)**: 30-60 seconds for 4x upscaling
- **Fallback (OpenCV)**: 5-10 seconds

### Quality Comparison
- **Real-ESRGAN**: Best quality, preserves details, reduces artifacts
- **OpenCV Fallback**: Good quality, simple interpolation

## 🎨 Results Preview

Your enhanced image will have:
- ✅ 4x larger dimensions
- ✅ Sharper details and edges
- ✅ Reduced blur and pixelation
- ✅ Better text readability (if any)
- ✅ Enhanced fine structures

Perfect for use with SAM mask generation as higher resolution typically produces more accurate segmentation results! 