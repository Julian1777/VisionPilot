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
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def color_threshold(image, avg_brightness=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # White lane lines
    white_lower = np.array([0, 0, 170])
    white_upper = np.array([80, 80, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Yellow lane lines
    yellow_lower = np.array([15, 80, 180])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Shadow areas - handle darker lane markings
    shadow_lower = np.array([90, 15, 150])
    shadow_upper = np.array([150, 80, 255])
    shadow_mask = cv2.inRange(hsv, shadow_lower, shadow_upper)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)
    
    # Create binary image
    binary = np.zeros_like(hsv[:,:,0])
    binary[combined_mask > 0] = 1
    return binary


def combine_thresholds(color_binary, gradient_binary):
    combined_binary = np.zeros_like(color_binary)
    combined_binary[color_binary == 1] = 1
    combined_binary[(color_binary == 1) & (gradient_binary == 1)] = 1
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
    combined_binary = combine_thresholds(color_binary, grad_binary)
    
    if debug_display:
        grad_display = np.dstack((grad_binary, grad_binary, grad_binary)) * 255
        cv2.imshow('1a. Gradient Threshold', grad_display)
        
        color_display = np.dstack((color_binary, color_binary, color_binary)) * 255
        cv2.imshow('1b. Color Threshold', color_display)
        
        union_binary = combine_thresholds(color_binary, grad_binary)
        union_display = np.dstack((union_binary, union_binary, union_binary)) * 255
        cv2.imshow('1c. Combined (Union)', union_display)
        
        intersection_display = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
        cv2.imshow('1d. Combined (Intersection)', intersection_display)

    if debugger:
        debugger.debug_thresholding(image, grad_binary, color_binary, combined_binary)

    return combined_binary, avg_brightness
