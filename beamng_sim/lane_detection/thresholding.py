import numpy as np
import cv2


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dx = 1 if orient == 'x' else 0
    dy = 0 if orient == 'x' else 1
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary


def gradient_thresholds(image, ksize=3, avg_brightness=None):
    x_low, x_high = 40, 120
    y_low, y_high = 40, 120
    mag_low, mag_high = 50, 120
    
    if avg_brightness is not None:
        if avg_brightness < 80:
            x_low = 30
            y_low = 30
            mag_low = 40
        elif avg_brightness > 200:
            x_high = 160
            y_high = 160
            mag_high = 160
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(x_low, x_high))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(y_low, y_high))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(mag_low, mag_high))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined


def color_threshold(image, avg_brightness=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    w_h_min, w_h_max = 0, 180
    w_s_min, w_s_max = 0, 25
    w_v_min, w_v_max = 200, 255

    y_h_min, y_h_max = 10, 45
    y_s_min, y_s_max = 50, 255
    y_v_min, y_v_max = 100, 255

    s_h_min, s_h_max = 0, 180
    s_s_min, s_s_max = 0, 20
    s_v_min, s_v_max = 110, 150


    if not hasattr(color_threshold, "brightness_history"):
        color_threshold.brightness_history = []

    if avg_brightness is not None:
        color_threshold.brightness_history.append(avg_brightness)
        if len(color_threshold.brightness_history) > 5:
            color_threshold.brightness_history.pop(0)
            
        avg_recent = np.mean(color_threshold.brightness_history)
        variance = np.var(color_threshold.brightness_history) if len(color_threshold.brightness_history) > 1 else 0
        
        print(f"Avg brightness: {avg_brightness:.1f}, Recent avg: {avg_recent:.1f}, Variance: {variance:.1f}")
        print(f"White HSV: H({w_h_min}-{w_h_max}) S({w_s_min}-{w_s_max}) V({w_v_min}-{w_v_max})")
        
        is_stable_lighting = variance < 100
        
        if avg_recent > 200:  # Very bright conditions (direct sunlight)
            w_s_max = min(w_s_max, 15)
            w_v_min = 220
            y_s_min = 100
            
        elif avg_recent > 170:
            w_v_min = 200
            w_s_max = 20
            
        elif 100 < avg_recent < 170:
            w_v_min = 190
            w_s_max = 22

        elif 70 < avg_recent <= 100:
            w_v_min = 170
            w_s_max = 25
            s_v_max = 160

        elif avg_recent <= 70:
            w_v_min = 140
            w_s_max = 30
            y_v_min = 90
            y_s_min = 40
            s_v_max = 150

    # Apply white mask
    white_lower = np.array([w_h_min, w_s_min, w_v_min])
    white_upper = np.array([w_h_max, w_s_max, w_v_max])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Apply yellow mask
    yellow_lower = np.array([y_h_min, y_s_min, y_v_min])
    yellow_upper = np.array([y_h_max, y_s_max, y_v_max])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    white_pixels = np.sum(white_mask)
    yellow_pixels = np.sum(yellow_mask)
    print(f"White mask pixels: {white_pixels}, Yellow mask pixels: {yellow_pixels}")
    
    shadow_mask = np.zeros_like(white_mask)
    if avg_brightness is not None and 50 < avg_brightness < 150:
        shadow_lower = np.array([s_h_min, s_s_min, s_v_min])
        shadow_upper = np.array([s_h_max, s_s_max, s_v_max])
        shadow_mask = cv2.inRange(hsv, shadow_lower, shadow_upper)
    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    if avg_brightness is not None and 60 < avg_brightness < 120:
        kernel = np.ones((3,3), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)
    
    combined_pixels = np.sum(combined_mask)
    print(f"Combined color mask pixels: {combined_pixels}")
    
    binary = np.zeros_like(hsv[:,:,0])
    binary[combined_mask > 0] = 1
    
    return binary


def combine_thresholds(color_binary, gradient_binary, avg_brightness=None):
    combined_binary = np.zeros_like(color_binary)
    
    color_coverage = np.sum(color_binary)
    gradient_coverage = np.sum(gradient_binary)
    total_pixels = color_binary.size
    
    gradient_weight = 1.0
    color_weight = 1.0
    
    if avg_brightness is not None:
        if avg_brightness < 100:
            gradient_weight = 1.5
            color_weight = 0.7
        elif avg_brightness > 200:
            gradient_weight = 0.9
            color_weight = 1.3
        else:
            gradient_weight = 1.0
            color_weight = 1.1
    
    color_confidence = color_coverage / total_pixels
    gradient_confidence = gradient_coverage / total_pixels
    
    combined_binary[color_binary == 1] = 1
    
    color_pixels = np.sum(color_binary)
    print(f"Color binary pixels going into combine: {color_pixels}")
    print(f"Combined binary pixels after adding color: {np.sum(combined_binary)}")
    
    if avg_brightness is not None:
        if avg_brightness < 120:  # In darker conditions, rely more on gradients
            combined_binary[gradient_binary == 1] = 1
        elif avg_brightness < 180:  # Medium lighting
            dilated_color = cv2.dilate(color_binary.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
            combined_binary[(gradient_binary == 1) & (dilated_color > 0)] = 1
        else:  # Very bright conditions
            dilated_color = cv2.dilate(color_binary.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
            combined_binary[(gradient_binary == 1) & (dilated_color > 0)] = 1
    else:
        dilated_color = cv2.dilate(color_binary.astype(np.uint8), np.ones((7,7), np.uint8), iterations=1)
        combined_binary[(gradient_binary == 1) & (dilated_color > 0)] = 1
    
    if np.sum(combined_binary) > 0:
        combined_binary = combined_binary.astype(np.uint8)
        
        small_kernel = np.ones((2,2), np.uint8)
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, small_kernel)
        
        combined_binary = combined_binary.astype(bool)
    
    final_pixels = np.sum(combined_binary)
    print(f"Final combined binary pixels: {final_pixels}")
    
    return combined_binary


def apply_thresholds(image, src_points=None, debugger=None, debug_display=False):

    if src_points is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray[mask == 1])
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)


    grad_binary = gradient_thresholds(image, avg_brightness=avg_brightness)
    color_binary = color_threshold(image, avg_brightness=avg_brightness)
    combined_binary = combine_thresholds(color_binary, grad_binary, avg_brightness=avg_brightness)
    
    if debug_display:
        debug_display = np.dstack((np.zeros_like(combined_binary), 
                                  np.zeros_like(combined_binary),
                                  np.zeros_like(combined_binary)))
        
        debug_display[color_binary == 1] = [0, 0, 255]
        
        debug_display[(color_binary == 0) & (grad_binary == 1)] = [0, 255, 0]
        
        debug_display[(color_binary == 1) & (grad_binary == 1)] = [0, 255, 255]
        
        final_vis = np.zeros_like(debug_display)
        final_vis[combined_binary == 1] = [255, 255, 255]
        
        grad_display = np.dstack((grad_binary, grad_binary, grad_binary)) * 255
        cv2.imshow('1a. Gradient Threshold', grad_display)
        
        color_display = np.dstack((color_binary, color_binary, color_binary)) * 255
        cv2.imshow('1b. Color Threshold', color_display)
        
        combined_display_array = combined_binary.astype(np.uint8)
        combined_display = np.dstack((combined_display_array, combined_display_array, combined_display_array)) * 255
        cv2.imshow('1. Lane Detection Combined', combined_display)
        
        cv2.imshow('1c. Detection Method Contributions', debug_display)
        
        cv2.imshow('1d. Final Output Pixels', final_vis)

    if debugger:
        debugger.debug_thresholding(image, grad_binary, color_binary, combined_binary)

    return combined_binary, avg_brightness
