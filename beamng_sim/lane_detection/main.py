from .thresholding import apply_thresholds
from .perspective import perspective_warp
from .lane_finder import get_histogram, sliding_window_search
from .metrics import calculate_curvature_and_deviation
from .visualization import draw_lane_overlay, add_text_overlay
import numpy as np


def process_frame(frame, debugger=None):
    """
    Main function to process a frame for lane detection.
    
    Args:
        frame: Input image frame
        debugger: Optional debugger instance for detailed analysis
    
    Returns:
        result_image: Image with lane overlay
        metrics: Dictionary containing lane metrics
    """
    try:
        # Step 1: Apply thresholds to create binary image
        binary_image = apply_thresholds(frame, debugger)
        
        # Step 2: Apply perspective transform
        binary_warped, Minv = perspective_warp(binary_image, debugger=debugger)
        
        # Step 3: Find lane lines
        histogram = get_histogram(binary_warped)
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram, debugger)
        
        # Step 4: Calculate metrics
        left_curverad, right_curverad, deviation, lane_center, vehicle_center = \
            calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped, debugger)
        
        # Step 5: Create visualization
        result = draw_lane_overlay(frame, binary_warped, Minv, left_fitx, right_fitx, ploty)
        result = add_text_overlay(result, left_curverad, right_curverad, deviation)
        
        # Return metrics
        metrics = {
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'deviation': deviation,
            'lane_center': lane_center,
            'vehicle_center': vehicle_center
        }
        
        return result, metrics
        
    except Exception as e:
        print(f"Lane detection error: {e}")
        result = frame.copy()
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'error': str(e)
        }
        return result, metrics
