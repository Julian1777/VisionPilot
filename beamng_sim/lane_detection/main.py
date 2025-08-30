import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamng_sim.lane_detection.thresholding import apply_thresholds
from beamng_sim.lane_detection.perspective import get_src_points,perspective_warp
from beamng_sim.lane_detection.lane_finder import get_histogram, sliding_window_search
from beamng_sim.lane_detection.metrics import calculate_curvature_and_deviation
from beamng_sim.lane_detection.visualization import draw_lane_overlay, add_text_overlay
import numpy as np
import cv2


def process_frame(frame, speed=0, debugger=None, debug_display=False):
    try:

        src_points = get_src_points(frame.shape, speed)

        # Step 1: Apply thresholds to create binary image - pass debug flag
        binary_image, avg_brightness = apply_thresholds(frame, src_points=src_points, debugger=debugger, debug_display=debug_display)
        if debug_display:
            cv2.imshow('1. Binary Image', binary_image*255 if binary_image.max()<=1 else binary_image)
            cv2.waitKey(1)

        # AVG Brightness DEBUG WINDOW
        
        # Step 2: Apply perspective transform
        binary_warped, Minv = perspective_warp(binary_image, speed=speed, debugger=debugger)
        
        # Debug display for warped binary
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            cv2.imshow('2. Warped Binary', warped_display)
        
        # Step 3: Find lane lines
        histogram = get_histogram(binary_warped)
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram, debugger)
        
        # Debug display for lane line detection
        if debug_display:
            lane_img = np.zeros_like(frame)
            
            if len(left_fitx) > 0 and len(right_fitx) > 0 and len(ploty) > 0:
                try:
                    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([left_points]), False, (255, 0, 0), 8)
                    
                    right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([right_points]), False, (0, 0, 255), 8)
                    
                except Exception as lane_err:
                    print(f"Debug visualization error: {lane_err}")
            else:
                print("Insufficient lane pixels detected!")
            
            cv2.imshow('3. Lane Lines', lane_img)
        
        # Step 4: Calculate metrics
        metrics_result = calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped, debugger)
        
        if metrics_result == (None, None, None, None, None):
            left_curverad, right_curverad, deviation, lane_center, vehicle_center = None, None, None, None, None
            print("Lane detection metrics calculation returned None values")
        else:
            left_curverad, right_curverad, deviation, lane_center, vehicle_center = metrics_result
        
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
