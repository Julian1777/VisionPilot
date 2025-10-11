import numpy as np
import cv2


def get_histogram(binary_warped):
    """
    Compute the histogram of the bottom half of the binary warped image.
    Args:
        binary_warped (numpy array): Warped binary image
    Returns:
        numpy array: Histogram of pixel intensities along the x-axis
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    return histogram


def sliding_window_search(binary_warped, histogram, debugger=None):
    """
    Perform sliding window search to find lane lines in a binary warped image.
    Args:
        binary_warped (numpy array): Warped binary image
        histogram (numpy array): Histogram of pixel intensities along the x-axis
        debugger (object, optional): Debugger object for visualization
    Returns:
        tuple: (ploty, left_fit, right_fit, left_fitx, right_fitx
    """
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
        
        max_jump = 80  # maximum allowed jump in pixels
        if len(good_left_inds) > minpix:
            new_leftx = int(np.mean(nonzerox[good_left_inds]))
            if abs(new_leftx - leftx_current) > max_jump:
                print(f"Left window jump too far: {new_leftx-leftx_current} pixels, limiting jump.")
                leftx_current += np.sign(new_leftx - leftx_current) * max_jump
            else:
                leftx_current = new_leftx
        if len(good_right_inds) > minpix:        
            new_rightx = int(np.mean(nonzerox[good_right_inds]))
            if abs(new_rightx - rightx_current) > max_jump:
                print(f"Right window jump too far: {new_rightx-rightx_current} pixels, limiting jump.")
                rightx_current += np.sign(new_rightx - rightx_current) * max_jump
            else:
                rightx_current = new_rightx

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print("No lane pixels found in sliding window search")
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])
        
        if debugger:
            debugger.debug_lane_finding(binary_warped, histogram, ([], []), ([], []), 
                                       left_fitx, right_fitx, ploty)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    if not hasattr(sliding_window_search, 'left_lane_pos_history'):
        sliding_window_search.left_lane_pos_history = []
    if not hasattr(sliding_window_search, 'right_lane_pos_history'):
        sliding_window_search.right_lane_pos_history = []

    if len(leftx) > 0:
        current_left_pos = np.median(leftx)
        sliding_window_search.left_lane_pos_history.append(current_left_pos)
        if len(sliding_window_search.left_lane_pos_history) > 30:
            sliding_window_search.left_lane_pos_history.pop(0)
        avg_left_pos = np.mean(sliding_window_search.left_lane_pos_history)
        if abs(current_left_pos - avg_left_pos) > 120:
            print(f"Left lane jump too far from history: {current_left_pos} vs avg {avg_left_pos:.1f}, ignoring")
            leftx = np.array([])
            lefty = np.array([])

    if len(rightx) > 0:
        current_right_pos = np.median(rightx)
        sliding_window_search.right_lane_pos_history.append(current_right_pos)
        if len(sliding_window_search.right_lane_pos_history) > 30:
            sliding_window_search.right_lane_pos_history.pop(0)
        avg_right_pos = np.mean(sliding_window_search.right_lane_pos_history)
        if abs(current_right_pos - avg_right_pos) > 120:
            print(f"Right lane jump too far from history: {current_right_pos} vs avg {avg_right_pos:.1f}, ignoring")
            rightx = np.array([])
            righty = np.array([])
    if not hasattr(sliding_window_search, 'left_points_history'):
        sliding_window_search.left_points_history = []
        sliding_window_search.right_points_history = []

    sliding_window_search.left_points_history.append(len(leftx))
    sliding_window_search.right_points_history.append(len(rightx))

    if len(sliding_window_search.left_points_history) > 30:
        sliding_window_search.left_points_history.pop(0)
    if len(sliding_window_search.right_points_history) > 30:
        sliding_window_search.right_points_history.pop(0)

    avg_left_points = np.mean(sliding_window_search.left_points_history) if sliding_window_search.left_points_history else 0
    avg_right_points = np.mean(sliding_window_search.right_points_history) if sliding_window_search.right_points_history else 0

    if avg_left_points > 0 and len(leftx) > avg_left_points * 1.7:
        print(f"Ignoring left lane: {len(leftx)} points vs avg {avg_left_points:.1f}")
        leftx = np.array([])
        lefty = np.array([])
    if avg_right_points > 0 and len(rightx) > avg_right_points * 1.7:
        print(f"Ignoring right lane: {len(rightx)} points vs avg {avg_right_points:.1f}")
        rightx = np.array([])
        righty = np.array([])
        
    if not hasattr(sliding_window_search, 'grace_counter'):
        sliding_window_search.grace_counter = 0
        sliding_window_search.last_valid = None

    if len(leftx) < 50 or len(rightx) < 50:
        print(f"Insufficient lane pixels: left={len(leftx)}, right={len(rightx)}")
        if sliding_window_search.last_valid is not None and sliding_window_search.grace_counter < 8:
            print(f"Using last valid lane positions (grace {sliding_window_search.grace_counter}/8)")
            sliding_window_search.grace_counter += 1
            ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search.last_valid
        else:
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
            right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
            left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
            right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])
            sliding_window_search.grace_counter = 0
            sliding_window_search.last_valid = None
        if debugger:
            debugger.debug_lane_finding(binary_warped, histogram, ([], []), ([], []), 
                                       left_fitx, right_fitx, ploty)
        return ploty, left_fit, right_fit, left_fitx, right_fitx

    sliding_window_search.grace_counter = 0
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    sliding_window_search.last_valid = (ploty, left_fit, right_fit, left_fitx, right_fitx)

    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        if debugger:
            left_points = (lefty, leftx) if len(leftx) > 0 else ([], [])
            right_points = (righty, rightx) if len(rightx) > 0 else ([], [])
            debugger.debug_lane_finding(binary_warped, histogram, left_points, right_points, 
                                       left_fitx, right_fitx, ploty)
            
            debugger.plot_lane_points_interactive(left_points, right_points, left_fitx, right_fitx, 
                                                ploty, binary_warped)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx
        
    except Exception as e:
        print(f"Error in polynomial fitting: {e}")
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = np.full_like(ploty, binary_warped.shape[1] // 4)
        right_fitx = np.full_like(ploty, 3 * binary_warped.shape[1] // 4)
        left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
        right_fit = np.array([0, 0, 3 * binary_warped.shape[1] // 4])
        
        if debugger:
            debugger.debug_lane_finding(binary_warped, histogram, ([], []), ([], []), 
                                       left_fitx, right_fitx, ploty)
        
        return ploty, left_fit, right_fit, left_fitx, right_fitx
