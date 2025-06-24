# SAM Droplet Segmentation - Pixel-Level Filtering

This enhanced version of the SAM droplet segmentation tool includes advanced pixel-level filtering capabilities to identify and isolate specific types of objects based on their pixel intensity characteristics.

## 🚀 New Features

### Pixel Intensity Analysis
- **Mean Intensity**: Average brightness of pixels within each object
- **Pixel Range**: Minimum and maximum pixel values in each object  
- **Standard Deviation**: Measure of pixel intensity variation (texture)
- **Area Filtering**: Filter by object size in pixels

### Interactive Filtering Interface
- **Collapsible Filter Panel**: Easy-to-use controls for setting filter values
- **Real-time Analysis**: Analyze images to understand pixel statistics
- **Smart Suggestions**: Get recommended filter ranges based on image analysis
- **Filter Preview**: See pixel statistics for each detected object
- **Interactive Click Filtering**: Click on masks to manually filter out false positives
- **Visual Feedback**: Filtered masks are shown with reduced opacity and strikethrough
- **Undo/Redo Controls**: Undo last filter action or reset all filters

## 📋 Filter Types

### 1. Mean Intensity Filters
Filter objects based on average pixel brightness:
- `mean_min`: Include only objects brighter than this value (0-255)
- `mean_max`: Include only objects darker than this value (0-255)

**Use Cases:**
- Isolate bright droplets: `{"mean_min": 150}`
- Find dark particles: `{"mean_max": 100}`

### 2. Pixel Threshold Filters
Filter based on extreme pixel values within objects:
- `min_threshold`: Exclude objects with any pixels darker than this
- `max_threshold`: Exclude objects with any pixels brighter than this

**Use Cases:**
- Remove objects with very dark pixels: `{"min_threshold": 50}`
- Remove objects with very bright pixels: `{"max_threshold": 200}`

### 3. Texture/Variation Filters
Filter based on pixel intensity variation within objects:
- `std_min`: Include only objects with high internal contrast
- `std_max`: Include only objects with uniform intensity

**Use Cases:**
- Find textured objects: `{"std_min": 15}`
- Find uniform objects: `{"std_max": 10}`

### 4. Area Filters
Filter based on object size:
- `area_min`: Minimum object size in pixels
- `area_max`: Maximum object size in pixels

**Use Cases:**
- Remove small noise: `{"area_min": 100}`
- Find large objects: `{"area_min": 1000, "area_max": 50000}`

### 5. Edge Proximity Filters
Filter based on distance from image edges:
- `min_edge_distance`: Minimum distance in pixels from any image edge
- `exclude_edge_touching`: Set to `true` to exclude objects touching any edge

**Use Cases:**
- Remove incomplete masks: `{"exclude_edge_touching": true}`
- Keep only center objects: `{"min_edge_distance": 50}`
- Ensure complete droplets: `{"min_edge_distance": 20, "exclude_edge_touching": true}`

### 6. Interactive Click Filtering
Manually filter out false positives by clicking on masks:
- **Click to Filter**: Click on any mask to toggle its filter state
- **Visual Feedback**: Filtered masks become semi-transparent
- **Smart Hover**: Filtered masks are excluded from hover interactions
- **Undo Support**: Easily undo the last filtering action

**Use Cases:**
- Remove obvious false positives after automated filtering
- Fine-tune results by manually excluding unwanted detections
- Quick cleanup of segmentation results
- Interactive quality control workflow

## 🎯 Example Use Cases

### Bright Droplet Detection
```json
{
  "mean_min": 140,
  "area_min": 50,
  "area_max": 5000,
  "std_max": 20
}
```
Finds bright, medium-sized, uniform objects.

### Dark Particle Analysis
```json
{
  "mean_max": 80,
  "min_threshold": 10,
  "area_min": 20
}
```
Finds dark particles while excluding very small artifacts.

### High-Contrast Object Detection
```json
{
  "std_min": 25,
  "area_min": 200
}
```
Finds objects with significant internal texture or patterns.

### Size-Based Filtering
```json
{
  "area_min": 1000,
  "area_max": 10000
}
```
Isolates objects within a specific size range.

### Complete Droplets Only
```json
{
  "exclude_edge_touching": true,
  "min_edge_distance": 10,
  "area_min": 100
}
```
Finds only complete objects that don't touch image edges.

### Center Region Analysis
```json
{
  "min_edge_distance": 100,
  "mean_min": 120,
  "area_min": 200
}
```
Analyzes bright objects only in the center region of the image.

### Edge-Filtered Droplets
```json
{
  "mean_min": 140,
  "area_min": 50,
  "area_max": 5000,
  "std_max": 20,
  "exclude_edge_touching": true
}
```
Finds bright, medium-sized, uniform objects that are complete (not cut off by edges).

## 🔧 Usage Instructions

### Web Interface
1. **Start the Server**:
   ```bash
   cd sam_droplet
   python app.py
   ```

2. **Open Browser**: Navigate to `http://localhost:9487`

3. **Upload Image**: Drag and drop or select an image file

4. **Analyze Image** (Optional but Recommended):
   - Click "📊 Analyze Image" to see image statistics
   - Review the suggested filter ranges
   - Use this information to set appropriate filter values

5. **Set Filters**:
   - Expand the "🔧 Pixel Intensity Filters" panel
   - Enter desired filter values in the input fields
   - Leave fields blank to not apply that filter

6. **Apply Filters**: Click "🔍 Apply Filters" to segment with filtering

7. **Review Results**:
   - Hover over the original image to preview individual masks
   - View pixel statistics for each detected object
   - Adjust filters and re-run as needed

8. **Interactive Refinement** (Optional):
   - Click on masks in the overlay to filter out false positives
   - Use "Show Hidden" to review filtered masks
   - Use "Undo Last" to reverse recent filtering actions
   - Use "Reset All" to clear all interactive filters

### API Usage

#### Analyze Image Statistics
```bash
curl -X POST http://localhost:9487/analyze_image \
  -F "file=@your_image.jpg"
```

#### Segment with Filters
```bash
curl -X POST http://localhost:9487/segment_file \
  -F "file=@your_image.jpg" \
  -F 'filters={"mean_min": 150, "area_min": 100}'
```

## 📊 Understanding Results

### Pixel Statistics Display
Each detected object shows:
- **Mean Intensity**: Average brightness (0-255)
- **Range**: Min-Max pixel values in the object
- **Std Dev**: Standard deviation (higher = more textured)
- **Area**: Object size in pixels
- **Edge Distance**: Minimum distance to any image edge
- **Edge Status**: Whether the object touches any edge

### Filter Effectiveness
- **Green Message**: "Successfully filtered to X objects! (Filters applied)"
- **Object Count**: Compare before/after filtering
- **Hover Preview**: Inspect individual objects to verify filtering accuracy

## 🧪 Testing and Development

### Demo Script
```bash
python test_filters.py
```
Runs a demonstration of the filtering capabilities with examples.

### Python API
```python
from sam_droplet.filters import filter_masks_by_criteria

# Apply filters to existing masks
filtered_masks = filter_masks_by_criteria(
    masks=original_masks,
    image=image_array,
    filters={"mean_min": 150, "area_min": 100}
)
```

## 🎨 Filter Strategy Tips

1. **Start with Image Analysis**: Always analyze your image first to understand the pixel intensity distribution

2. **Use Conservative Filters**: Start with broader ranges and refine gradually

3. **Combine Filter Types**: Mix intensity, texture, and area filters for precise selection

4. **Iterative Refinement**: 
   - Apply filters
   - Review results with hover preview
   - Adjust values based on pixel statistics
   - Re-apply

5. **Common Patterns**:
   - **Droplet Detection**: Mean intensity + area filters
   - **Noise Removal**: Area filters + std deviation limits  
   - **Material Classification**: Mean + std deviation combinations

6. **Interactive Workflow**:
   - Apply automated filters first to reduce false positives
   - Use interactive clicking to manually remove remaining false positives
   - Toggle between showing/hiding filtered masks for review
   - Use undo functionality for iterative refinement

## 🔍 Troubleshooting

### No Objects Found After Filtering
- Check if filter values are too restrictive
- Use image analysis to see actual pixel value ranges
- Try removing one filter at a time to identify the issue

### Too Many Objects Still Present
- Make filters more restrictive
- Add additional filter types (e.g., combine mean + area)
- Check pixel statistics of unwanted objects to set better thresholds

### Inconsistent Results
- Ensure image quality is consistent
- Consider lighting conditions in your filter strategy
- Use relative thresholds based on image analysis

## 📈 Performance Notes

- Filtering is applied after SAM segmentation
- Analysis endpoint is fast and helps optimize filter selection
- Pixel statistics are cached with each mask for efficient hover preview
- Filter operations are optimized for real-time interaction 