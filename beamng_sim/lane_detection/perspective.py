import numpy as np
import cv2
import pickle
import os


def undistort_image(img, calibration_data):
    """
    Undistort an image using camera calibration parameters.
    
    Args:
        img (numpy array): Distorted image
        calibration_data (dict): Calibration data containing 'mtx' and 'dist'
    
    Returns:
        numpy array: Undistorted image, or original image if calibration_data is None
    """
    if calibration_data is None:
        return img
    
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    return cv2.undistort(img, mtx, dist, None, mtx)


def load_calibration(calibration_file):
    """
    Load camera calibration parameters from file.
    
    Args:
        calibration_file (str): Path to pickled calibration file
    
    Returns:
        dict: Calibration data containing mtx, dist, rvecs, tvecs, img_shape, rms_error
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)
    
    print(f"Loaded calibration from: {calibration_file}")
    print(f"RMS error: {calibration_data['rms_error']:.4f}")
    
    return calibration_data

def get_src_points(image_shape, speed=0, previous_steering=0):
    """
    Generate source points for perspective transform based on the passed image shape, speed of the vehicle, and steering angle

    Args:
        image_shape (tuple): Shape of the input image (height, width, channels)
        speed (float): Speed of the vehicle in km/h
        previous_steering (float): Previous steering angle in degrees
    
    Returns:
        Numpy Array: Array of source points for perspective transform left_bottom, right_bottom, top_right, top_left

    """
    h, w = image_shape[:2]

    left_bottom  = [6, 300]
    right_bottom = [634, 300]
    top_right    = [385, 201]
    top_left     = [255, 201]

    speed_norm = min(speed / 120.0, 1.0)
    top_shift = -30 * speed_norm
    side_shift = 50 * speed_norm

    src = np.float32([
        [left_bottom[0] + side_shift, left_bottom[1]],           # Bottom-left
        [right_bottom[0] - side_shift, right_bottom[1]],         # Bottom-right
        [top_right[0] - side_shift, top_right[1] + top_shift],   # Top-right
        [top_left[0] + side_shift, top_left[1] + top_shift]      # Top-left
    ])
    
    return src

def perspective_warp(img, speed=0, debug_display=False, calibration_data=None):
    """
    Applies perspective transform to the passed image using the source points generated.
    Optionally applies camera undistortion first if calibration data is provided.

    Args:
        img (numpy array): Input image to be warped
        speed (float): Speed of the vehicle in km/h
        calibration_data (dict): Optional camera calibration data for undistortion
    Returns:
        tuple: (warped image, inverse perspective transform matrix)
    """
    
    # Apply camera undistortion if calibration data is available
    if calibration_data is not None:
        img = undistort_image(img, calibration_data)
        
    img_size = (img.shape[1], img.shape[0])
    w, h = img_size

    src = get_src_points(img.shape, speed)

    dst = np.float32([
        [w*0.2, h],        # Bottom-left in warped space
        [w*0.8, h],        # Bottom-right in warped space
        [w*0.8, 0],        # Top-right in warped space
        [w*0.2, 0]         # Top-left in warped space
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return binary_warped, Minv


def debug_perspective_live(img, speed_kph, previous_steering=0):
    """
    Visualize the perspective transform source points on the original image for debugging.
    """
    debug_img = img.copy()
    
    src_points = get_src_points(img.shape, speed_kph, previous_steering)
    
    src_int = src_points.astype(np.int32)
    
    cv2.polylines(debug_img, [src_int], isClosed=True, color=(0, 255, 0), thickness=2)
    
    labels = ['Bottom Left', 'Bottom Right', 'Top Right', 'Top Left']
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    
    for i, (point, label, color) in enumerate(zip(src_int, labels, colors)):
        cv2.circle(debug_img, tuple(point), 5, color, -1)
        cv2.putText(debug_img, label, (point[0] + 10, point[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.putText(debug_img, f"Speed: {speed_kph:.1f} km/h", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Steering: {previous_steering:.1f} deg", (10, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, "Perspective Transform Region", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_img, "Green: Transform Area", (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Perspective Transform Debug', debug_img)
    cv2.waitKey(1)
