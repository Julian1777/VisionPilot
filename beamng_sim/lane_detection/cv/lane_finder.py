import numpy as np
import cv2



def get_histogram(binary_warped):
    """
    Compute the histogram of the bottom half of the binary warped image.
    Uses Gaussian smoothing to reduce noise bias.
    Args:
        binary_warped (numpy array): Warped binary image
    Returns:
        numpy array: Smoothed histogram of pixel intensities along the x-axis
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram_blurred = cv2.GaussianBlur(histogram.astype(np.float32).reshape(-1, 1), (11, 1), 1.5)
    histogram = histogram_blurred.flatten()
    return histogram

def detect_lane_type(binary_warped):
    histogram = get_histogram(binary_warped)

    histogram_variance = np.var(histogram)

    is_dashed_lane = histogram_variance > 500

    if is_dashed_lane:
        print(f"Dashed lane detected (variance: {histogram_variance:.1f})")
    else:
        print(f"Solid lane detected (variance: {histogram_variance:.1f})")

    return is_dashed_lane

def fill_dashed_lane_gaps(binary_warped, gap_size=20):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, gap_size))
    filled = cv2.morphologyEx(binary_warped, cv2.MORPH_CLOSE, kernel)
    return filled

def sliding_window_search(binary_warped, histogram):
    # Additional filtering state
    if not hasattr(sliding_window_search, 'last_lane_center'):
        sliding_window_search.last_lane_center = None
    if not hasattr(sliding_window_search, 'last_lane_width'):
        sliding_window_search.last_lane_width = None
    if not hasattr(sliding_window_search, 'last_valid_lanes'):
        sliding_window_search.last_valid_lanes = None
    if not hasattr(sliding_window_search, 'last_left_fitx'):
        sliding_window_search.last_left_fitx = None
    if not hasattr(sliding_window_search, 'last_right_fitx'):
        sliding_window_search.last_right_fitx = None
    if not hasattr(sliding_window_search, 'frame_skip_count'):
        sliding_window_search.frame_skip_count = 0
    
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100  # Use symmetric margin for both lanes

    is_dashed_lane = detect_lane_type(binary_warped)
    minpix = 30 if is_dashed_lane else 50


    left_lane_inds = []
    right_lane_inds = []
    
    # Track pixel counts and positions for quality assessment
    left_pixel_count = 0
    right_pixel_count = 0
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        max_jump = 80  # maximum allowed jump in pixels
        if len(good_left_inds) > minpix:
            new_leftx = int(np.mean(nonzerox[good_left_inds]))
            jump_left = new_leftx - leftx_current
            if abs(jump_left) > max_jump:
                print(f"Left window jump: {jump_left} pixels (threshold: {max_jump}), limiting jump.")
                leftx_current += np.sign(jump_left) * max_jump
            else:
                leftx_current = new_leftx
        if len(good_right_inds) > minpix:        
            new_rightx = int(np.mean(nonzerox[good_right_inds]))
            jump_right = new_rightx - rightx_current
            if abs(jump_right) > max_jump:
                print(f"Right window jump: {jump_right} pixels (threshold: {max_jump}), limiting jump.")
                rightx_current += np.sign(jump_right) * max_jump
            else:
                rightx_current = new_rightx

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print("No lane pixels found in sliding window search")
        print(f"  Total nonzero pixels: {len(nonzerox)}")
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])  # degree 2 coefficients
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])  # degree 2 coefficients
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Quality check: require sufficient pixels for both lanes
    min_pixels_required = 200
    if len(leftx) < min_pixels_required or len(rightx) < min_pixels_required:
        print(f"Insufficient lane pixels: left={len(leftx)}, right={len(rightx)}")
        if sliding_window_search.last_valid_lanes is not None:
            ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search.last_valid_lanes
            print("Using last valid detection due to insufficient pixels")
            return ploty, left_fit, right_fit, left_fitx, right_fitx
    
    if len(leftx) > 0 and len(rightx) > 0:
        left_mean_x = np.mean(leftx)
        right_mean_x = np.mean(rightx)
        detected_lane_width = right_mean_x - left_mean_x

    if not hasattr(sliding_window_search, 'last_valid_lanes'):
        sliding_window_search.last_valid_lanes = None

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fit = None
    right_fit = None
    left_fitx = None
    right_fitx = None
    lane_width_check = None

    try:
        if len(leftx) > 0 and len(rightx) > 0:
            # Adaptive polynomial degree: start with degree 1, increase if curvature is detected
            # First fit degree 1 (linear)
            left_fit_deg1 = np.polyfit(lefty, leftx, 1)
            right_fit_deg1 = np.polyfit(righty, rightx, 1)
            
            # Check if straight fit is good enough (R-squared > 0.98)
            left_residuals_deg1 = leftx - (left_fit_deg1[0]*lefty + left_fit_deg1[1])
            right_residuals_deg1 = rightx - (right_fit_deg1[0]*righty + right_fit_deg1[1])
            
            left_ss_res_deg1 = np.sum(left_residuals_deg1**2)
            left_ss_tot = np.sum((leftx - np.mean(leftx))**2)
            left_r_squared_deg1 = 1 - (left_ss_res_deg1 / left_ss_tot) if left_ss_tot > 0 else 0
            
            right_ss_res_deg1 = np.sum(right_residuals_deg1**2)
            right_ss_tot = np.sum((rightx - np.mean(rightx))**2)
            right_r_squared_deg1 = 1 - (right_ss_res_deg1 / right_ss_tot) if right_ss_tot > 0 else 0
            
            # If straight fit is good for both lanes, use degree 1 (no artificial curvature)
            if left_r_squared_deg1 > 0.98 and right_r_squared_deg1 > 0.98:
                left_fit = np.append(left_fit_deg1, 0)  # [0, slope, intercept]
                right_fit = np.append(right_fit_deg1, 0)
                degree_used = 1
            else:
                # Try degree 2 for moderate curves
                left_fit_deg2 = np.polyfit(lefty, leftx, 2)
                right_fit_deg2 = np.polyfit(righty, rightx, 2)
                
                left_residuals_deg2 = leftx - (left_fit_deg2[0]*lefty**2 + left_fit_deg2[1]*lefty + left_fit_deg2[2])
                right_residuals_deg2 = rightx - (right_fit_deg2[0]*righty**2 + right_fit_deg2[1]*righty + right_fit_deg2[2])
                
                left_ss_res_deg2 = np.sum(left_residuals_deg2**2)
                left_r_squared_deg2 = 1 - (left_ss_res_deg2 / left_ss_tot) if left_ss_tot > 0 else 0
                
                right_ss_res_deg2 = np.sum(right_residuals_deg2**2)
                right_r_squared_deg2 = 1 - (right_ss_res_deg2 / right_ss_tot) if right_ss_tot > 0 else 0
                
                # If degree 2 is significantly better (>0.95 R²) and not too curvy yet, use it
                if (left_r_squared_deg2 > 0.95 and right_r_squared_deg2 > 0.95 and 
                    abs(left_fit_deg2[0]) < 0.0005 and abs(right_fit_deg2[0]) < 0.0005):
                    left_fit = left_fit_deg2
                    right_fit = right_fit_deg2
                    degree_used = 2
                else:
                    # Try degree 3 for very curvy roads
                    left_fit = np.polyfit(lefty, leftx, 3)
                    right_fit = np.polyfit(righty, rightx, 3)
                    degree_used = 3

            # Evaluate polynomial at each point
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)
            lane_width_check = abs(right_fitx[-1] - left_fitx[-1])
            
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        else:
            raise ValueError("Insufficient lane pixels")
            
    except Exception as e:
        print(f"Error in polynomial fitting: {e}")
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])  # degree 2 coefficients
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])  # degree 2 coefficients

    use_history = False
    if sliding_window_search.last_valid_lanes is not None:
        if len(left_fitx) < 50 or len(right_fitx) < 50:
            use_history = True
        elif lane_width_check and (lane_width_check < 100 or lane_width_check > 700):
            use_history = True
            print(f"Lane width unreasonable ({lane_width_check:.1f}), using history")
        elif left_fitx[-1] >= right_fitx[-1]:
            use_history = True
            print(f"Lane crossing detected (left={left_fitx[-1]:.1f}, right={right_fitx[-1]:.1f}), using history")
        else:
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
            if sliding_window_search.last_lane_center is not None:
                # max 80 pixel jump instead of 100
                max_center_shift = 80
                if abs(lane_center - sliding_window_search.last_lane_center) > max_center_shift:
                    use_history = True
                    print(f"Sudden lane center jump ({lane_center:.1f} vs {sliding_window_search.last_lane_center:.1f}), max allowed: {max_center_shift}, using history")

            if (lane_width_check is not None and sliding_window_search.last_lane_width is not None and
                isinstance(lane_width_check, (int, float)) and isinstance(sliding_window_search.last_lane_width, (int, float))):
                if abs(lane_width_check - sliding_window_search.last_lane_width) > 0.3 * sliding_window_search.last_lane_width:
                    use_history = True
                    print(f"Lane width changed too much ({lane_width_check:.1f} vs {sliding_window_search.last_lane_width:.1f}), using history")

            if (np.any(np.isnan(left_fitx)) or np.any(np.isnan(right_fitx)) or
                np.any(np.isinf(left_fitx)) or np.any(np.isinf(right_fitx))):
                use_history = True
                print("NaN or Inf detected in lane fit, using history")

    if use_history:
        ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search.last_valid_lanes
        print("Using last valid lane detection")
    else:
        if len(left_fitx) > 50 and len(right_fitx) > 50 and lane_width_check and 100 < lane_width_check < 700:
            sliding_window_search.last_valid_lanes = (ploty, left_fit, right_fit, left_fitx, right_fitx)
            sliding_window_search.last_lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
            sliding_window_search.last_lane_width = lane_width_check

    return ploty, left_fit, right_fit, left_fitx, right_fitx, is_dashed_lane
