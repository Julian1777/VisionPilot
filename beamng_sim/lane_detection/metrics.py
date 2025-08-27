import numpy as np


def calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped, debugger=None):
    """Calculate lane curvature and vehicle deviation from lane center."""
    ym_per_pix = 30/720  # meters per pixel in y dimension

    # Check for empty arrays first
    if len(left_fitx) == 0 or len(right_fitx) == 0:
        print("Empty lane fit arrays detected")
        return None, None, None, None, None

    leftx = left_fitx[::-1]
    rightx = right_fitx[::-1]

    y_eval = np.max(ploty)

    left_bottom = leftx[-1]
    right_bottom = rightx[-1]
    lane_width_pix = right_bottom - left_bottom

    # Sanity checks for lane width
    if lane_width_pix < 50 or lane_width_pix > 700:
        print(f"Unreasonable lane width: {lane_width_pix:.1f} pixels")
        return None, None, None, None, None

    # Check if lane lines are crossed
    if left_bottom >= right_bottom:
        print(f"Lane lines crossed: left={left_bottom:.1f}, right={right_bottom:.1f}")
        return None, None, None, None, None

    # Standard lane width is 3.55 meters (you changed this)
    xm_per_pix = 3.55 / lane_width_pix

    # Sanity check for meter-per-pixel calculation
    if xm_per_pix <= 0 or xm_per_pix > 0.1:
        print(f"Unreasonable xm_per_pix: {xm_per_pix:.4f}")
        return None, None, None, None, None

    try:
        # Fit polynomial in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                        / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*right_fit_cr[0])

        # Sanity check for curvature
        max_reasonable_curve = 1000  # meters
        if left_curverad > max_reasonable_curve or right_curverad > max_reasonable_curve:
            left_curverad = min(left_curverad, max_reasonable_curve)
            right_curverad = min(right_curverad, max_reasonable_curve)

        # Calculate deviation from lane center
        lane_center = (left_bottom + right_bottom) / 2.0
        vehicle_center = binary_warped.shape[1] / 2.0
        deviation_m = (vehicle_center - lane_center) * xm_per_pix

        # Strict limitation on deviation
        max_reasonable_deviation = 0.75  # meters
        if abs(deviation_m) > max_reasonable_deviation:
            print(f"Unreasonable deviation detected: {deviation_m:.2f}m, clipping to {max_reasonable_deviation:.2f}m")
            deviation_m = np.clip(deviation_m, -max_reasonable_deviation, max_reasonable_deviation)

        # Debug metrics calculation
        if debugger:
            debugger.debug_metrics(left_curverad, right_curverad, deviation_m, lane_center, 
                                 vehicle_center, lane_width_pix, xm_per_pix)

        return left_curverad, right_curverad, deviation_m, lane_center, vehicle_center
        
    except Exception as e:
        print(f"Error in curvature calculation: {e}")
        return None, None, None, None, None

    if debugger:
        debugger.debug_metrics(left_curverad, right_curverad, deviation_m, lane_center, 
                             vehicle_center, lane_width_pix, xm_per_pix)

    return left_curverad, right_curverad, deviation_m, lane_center, vehicle_center
