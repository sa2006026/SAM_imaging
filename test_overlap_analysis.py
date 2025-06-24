#!/usr/bin/env python3
"""
Test script to demonstrate the new overlap analysis functionality.
This shows how the enhanced mask_size_grouping.py now analyzes overlaps between 
smaller and larger masks with at least 80% coverage.
"""

import os
import sys

def test_overlap_analysis():
    """Test the overlap analysis functionality."""
    
    print("🔬 OVERLAP ANALYSIS DEMONSTRATION")
    print("="*60)
    print("Enhanced mask_size_grouping.py with overlap detection")
    print("="*60)
    
    # Check if the image exists
    image_path = "image/Test/GY_image.png"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        print("Please make sure you have the test image in the correct location")
        return False
    
    print(f"✅ Found test image: {image_path}")
    
    # Import the updated script
    try:
        import mask_size_grouping
        print("✅ Successfully imported mask_size_grouping.py")
    except ImportError as e:
        print(f"❌ Failed to import mask_size_grouping.py: {e}")
        return False
    
    print("\n🎯 NEW OVERLAP ANALYSIS FEATURES:")
    print("1. Two-group clustering (K-means with 6 features)")
    print("2. Automatic identification of larger vs smaller masks")
    print("3. Overlap detection with 80% minimum coverage")
    print("4. Large masks labeled with number of overlapping small masks")
    print("5. Color-coded visualization by overlap count")
    print("6. Detailed CSV export with overlap information")
    
    print("\n📊 ANALYSIS WORKFLOW:")
    print("Step 1: SAM generates initial masks")
    print("Step 2: Filter by edge proximity, circularity, blob distance")
    print("Step 3: K-means clustering into 2 groups")
    print("Step 4: Identify larger vs smaller mask clusters")
    print("Step 5: ⭐ NEW: Analyze overlaps between groups")
    print("Step 6: Label large masks with overlap counts (0, 1, 2, etc.)")
    print("Step 7: Generate overlap visualization and CSV")
    
    print("\n💾 NEW OUTPUT FILES:")
    print("- overlap_analysis_masks_80pct.png (visual with overlap counts)")
    print("- overlap_analysis_details.csv (detailed overlap data)")
    
    print(f"\n🚀 To run the enhanced analysis:")
    print(f"python3 mask_size_grouping.py")
    
    print(f"\n📈 Expected Output:")
    print("- Original clustering results (2 groups)")
    print("- Overlap analysis summary showing:")
    print("  • How many large masks have 0 overlapping small masks")
    print("  • How many large masks have 1 overlapping small mask") 
    print("  • How many large masks have 2+ overlapping small masks")
    print("- Visual showing large masks color-coded by overlap count")
    print("- CSV with overlap_count and overlapping_small_mask_ids columns")
    
    return True

def show_algorithm_details():
    """Show detailed algorithm explanation."""
    print("\n" + "🧠 OVERLAP ANALYSIS ALGORITHM DETAILS")
    print("─" * 60)
    
    print("\n📏 OVERLAP CALCULATION:")
    print("For each small mask S and large mask L:")
    print("1. Calculate intersection: overlap = S ∩ L")
    print("2. Calculate coverage: coverage% = (overlap_area / S_area) × 100")
    print("3. If coverage% ≥ 80%, count as overlapping")
    
    print("\n🎯 CLUSTER IDENTIFICATION:")
    print("1. Calculate average area for each cluster")
    print("2. Cluster with larger average = 'large masks'")
    print("3. Cluster with smaller average = 'small masks'")
    
    print("\n📊 FINAL OUTPUT:")
    print("Large masks are labeled with counts like:")
    print("- '0' = No overlapping small masks")
    print("- '1' = One overlapping small mask") 
    print("- '2' = Two overlapping small masks")
    print("- etc.")
    
    print("\n🎨 VISUALIZATION:")
    print("- Color map (viridis) based on overlap count")
    print("- Bounding boxes around large masks only")
    print("- Text labels showing mask ID + overlap count")
    print("- Color bar legend for overlap count mapping")

if __name__ == "__main__":
    print("🔬 MASK OVERLAP ANALYSIS - TEST & DEMONSTRATION")
    print("="*70)
    
    success = test_overlap_analysis()
    
    if success:
        show_algorithm_details()
        
        print("\n" + "="*70)
        print("✅ READY TO TEST OVERLAP ANALYSIS!")
        print("="*70)
        print("Run: python3 mask_size_grouping.py")
        print("Look for the 'OVERLAP ANALYSIS' section in the output")
        print("Check for new files: overlap_analysis_masks_80pct.png")
        print("                     overlap_analysis_details.csv")
        print("="*70)
    else:
        print("\n❌ Setup incomplete. Please check requirements.")
        sys.exit(1) 