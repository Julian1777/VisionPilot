import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamng_sim.lane_detection.cv.thresholding import apply_thresholds
from beamng_sim.lane_detection.perspective import get_src_points,perspective_warp
from beamng_sim.lane_detection.cv.lane_finder import get_histogram, sliding_window_search
from beamng_sim.lane_detection.metrics import calculate_curvature_and_deviation, process_deviation
from beamng_sim.lane_detection.visualization import draw_lane_overlay, add_text_overlay, create_mask_overlay
from beamng_sim.lane_detection.unet.postprocess import process_unet_mask, run_unet_on_frame

from beamng_sim.lane_detection.confidence import compute_confidence_cv
from beamng_sim.lane_detection.confidence import compute_confidence_unet


import numpy as np
import cv2


def process_frame_cv(img, speed=0, previous_steering=0, debug_display=False):
    previous_fit = None
    confidence = 0.0
    try:

        src_points = get_src_points(img.shape, speed, previous_steering)

        binary_image, avg_brightness = apply_thresholds(img, src_points=src_points, debug_display=debug_display)
        if debug_display:
            cv2.imshow('1. Binary Image CV', binary_image*255 if binary_image.max()<=1 else binary_image)
            cv2.waitKey(1)
        
        binary_warped, Minv = perspective_warp(binary_image, speed=speed, debug_display=True)
        
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            cv2.imshow('2. Warped Binary CV', warped_display)
        
        histogram = get_histogram(binary_warped)
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram)

        current_fit = (left_fit, right_fit)
        
        if debug_display:
            lane_img = np.zeros_like(img)
            
            if len(left_fitx) > 0 and len(right_fitx) > 0 and len(ploty) > 0:
                try:
                    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([left_points]), False, (255, 0, 0), 8)
                    
                    right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                    cv2.polylines(lane_img, np.int32([right_points]), False, (0, 0, 255), 8)
                    
                except Exception as lane_err:
                    print(f"Debug visualization error CV: {lane_err}")
            else:
                print("Insufficient lane pixels detected CV Method!")

            cv2.imshow('3. Lane Lines CV', lane_img)

        metrics_result = calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped)

        confidence = compute_confidence_cv(left_fitx, right_fitx, ploty, current_fit=current_fit, previous_fit=previous_fit)

        previous_fit = current_fit

        if metrics_result == (None, None, None, None, None, None):
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            smoothed_deviation = 0.0
            effective_deviation = 0.0
            print("Lane detection metrics calculation returned None values")
        else:
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = metrics_result
            smoothed_deviation, effective_deviation = process_deviation(deviation)

        result = draw_lane_overlay(img, binary_warped, Minv, left_fitx, right_fitx, ploty, deviation)
        result = add_text_overlay(result, left_curverad, right_curverad, deviation, avg_brightness, speed)
        
        metrics = {
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'deviation': deviation,
            'smoothed_deviation': smoothed_deviation,
            'effective_deviation': effective_deviation,
            'lane_center': lane_center,
            'vehicle_center': vehicle_center,
            'lane_width': lane_width,
            'confidence': confidence
        }
        
        return result, metrics, confidence
        
    except Exception as e:
        print(f"Lane detection error: {e}")
        result = img.copy()
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'smoothed_deviation': 0,
            'effective_deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'lane_width': 0,
            'confidence': 0,
            'error': str(e)
        }
        return result, metrics, confidence
    
def process_frame_unet(img, model, speed=0, previous_steering=0, debug_display=False):
    previous_fit = None
    confidence = 0.0
    try:
        mask = run_unet_on_frame(img, model)
        
        src_points = get_src_points(img.shape, speed, previous_steering)
        
        mask = process_unet_mask(mask, src_points, min_area=2000)
        
        binary_warped, Minv = perspective_warp(mask, speed=speed, debug_display=False)

        histogram = get_histogram(binary_warped)
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram)

        current_fit = (left_fit, right_fit)

        metrics_result = calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped)
        confidence = compute_confidence_unet(left_fitx, right_fitx, ploty, current_fit=current_fit, previous_fit=previous_fit)

        previous_fit = current_fit

        if metrics_result == (None, None, None, None, None, None):
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            smoothed_deviation = 0.0
            effective_deviation = 0.0
            print("Lane detection metrics calculation returned None values")
        else:
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = metrics_result
            smoothed_deviation, effective_deviation = process_deviation(deviation)

        result = draw_lane_overlay(img, binary_warped, Minv, left_fitx, right_fitx, ploty, deviation)
        result = create_mask_overlay(img, mask, alpha=0.4, color=(0, 255, 0))
        result = add_text_overlay(result, left_curverad, right_curverad, deviation, 0, speed)

        metrics = {
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'deviation': deviation,
            'smoothed_deviation': smoothed_deviation,
            'effective_deviation': effective_deviation,
            'lane_center': lane_center,
            'vehicle_center': vehicle_center,
            'lane_width': lane_width,
            'confidence': confidence
        }

        return result, metrics, confidence

    except Exception as e:
        print(f"Lane detection error: {e}")
        result = img.copy()
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'smoothed_deviation': 0,
            'effective_deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'lane_width': 0,
            'confidence': 0,
            'error': str(e)
        }
        return result, metrics, confidence
