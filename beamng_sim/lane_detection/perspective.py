import numpy as np
import cv2

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
    ref_w, ref_h = 1278, 720
    scale_w = w / ref_w
    scale_h = h / ref_h

    left_bottom  = [80, 590]
    right_bottom = [1115, 590]
    top_right    = [790, 408]
    top_left     = [500, 408]

    speed_norm = min(speed / 120.0, 1.0)
    top_shift = -40 * speed_norm
    side_shift = 100 * speed_norm

    max_steer_deg = 30.0
    max_shift_px = 200.0
    steer_norm = max(min(previous_steering / max_steer_deg, 1.0), -1.0)
    steer_shift = steer_norm * max_shift_px

    src = np.float32([
        [left_bottom[0] * scale_w + steer_shift, left_bottom[1] * scale_h],
        [right_bottom[0] * scale_w + steer_shift, right_bottom[1] * scale_h],
        [(top_right[0] - side_shift) * scale_w + steer_shift, (top_right[1] + top_shift) * scale_h],
        [(top_left[0] + side_shift) * scale_w + steer_shift,  (top_left[1]  + top_shift) * scale_h]
    ])
    return src

def perspective_warp(img, speed=0, debug_display=False):
    """
    Applies perspective transform to the passed image using the source points generated

    Args:
        img (numpy array): Input image to be warped
        speed (float): Speed of the vehicle in km/h
    Returns:
        tuple: (warped image, inverse perspective transform matrix)
    """
        
    img_size = (img.shape[1], img.shape[0])
    w, h = img_size

    src = get_src_points(img.shape, speed)

    dst = np.float32([
        [w*0.2, h],
        [w*0.8, h],
        [w*0.8, 0],
        [w*0.2, 0]
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
