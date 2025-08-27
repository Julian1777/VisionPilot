import numpy as np
import cv2


def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    return histogram


def sliding_window_search(binary_warped, histogram, debugger=None):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    
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
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

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
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate indices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print("No lane pixels found in sliding window search")
        # Return safe default values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])
        
        if debugger:
            debugger.debug_lane_finding(binary_warped, histogram, ([], []), ([], []), 
                                       left_fitx, right_fitx, ploty)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx

    # Extract lane pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Check if we have enough points for polynomial fitting
    if len(leftx) < 50 or len(rightx) < 50:
        print(f"Insufficient lane pixels: left={len(leftx)}, right={len(rightx)}")
        # Return empty/default values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)  # Default left lane position
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)  # Default right lane position
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])
        
        if debugger:
            debugger.debug_lane_finding(binary_warped, histogram, ([], []), ([], []), 
                                       left_fitx, right_fitx, ploty)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx

    try:
        # Fit polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Color the detected pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Debug lane finding step
        if debugger:
            left_points = (lefty, leftx) if len(leftx) > 0 else ([], [])
            right_points = (righty, rightx) if len(rightx) > 0 else ([], [])
            debugger.debug_lane_finding(binary_warped, histogram, left_points, right_points, 
                                       left_fitx, right_fitx, ploty)
            
            # Also create interactive plot for detailed inspection
            debugger.plot_lane_points_interactive(left_points, right_points, left_fitx, right_fitx, 
                                                ploty, binary_warped)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx
        
    except Exception as e:
        print(f"Error in polynomial fitting: {e}")
        # Return safe default values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])
        
        if debugger:
            debugger.debug_lane_finding(binary_warped, histogram, ([], []), ([], []), 
                                       left_fitx, right_fitx, ploty)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx
