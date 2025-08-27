import numpy as np
import cv2

def perspective_warp(img, speed=0, debugger=None):
    img_size = (img.shape[1], img.shape[0])
    w, h = img_size

    # Baseline city src points
    left_bottom  = [118, 608]
    right_bottom = [1077, 596]
    top_right    = [730, 406]
    top_left     = [519, 408]

    # Apply high speed logic to push top points further up
    speed_norm = min(speed / 120.0, 1.0)  # normalize speed (0-120 km/h)
    top_shift = -80 * speed_norm  # move up for higher speed (adjust as needed)

    src = np.float32([
        left_bottom,
        right_bottom,
        [top_right[0], top_right[1] + top_shift],
        [top_left[0],  top_left[1]  + top_shift]
    ])

    dst = np.float32([
        [w*0.2, h],
        [w*0.8, h],
        [w*0.8, 0],
        [w*0.2, 0]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    # Debug perspective transform
    if debugger:
        debugger.debug_perspective_transform(img, binary_warped, src, dst)
    
    return binary_warped, Minv
