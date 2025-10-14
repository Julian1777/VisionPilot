import numpy as np
import cv2


def draw_lane_overlay(original_image, warped_image, Minv, left_fitx, right_fitx, ploty, deviation):
    """
    Draw the detected lane area and lane lines on the original image.
    Lane lines are colored based on vehicle deviation from lane center.
    
    Args:
        Original_image: The original undistorted image
        warped_image: The warped binary image
        Minv: Inverse perspective transform matrix
        left_fitx: X coordinates for left lane line
        right_fitx: X coordinates for right lane line
        ploty: Y coordinates for lane fitting
        deviation: Vehicle deviation from lane center in meters
    Returns:
        Resulting image with lane overlay
    """
    
    if len(left_fitx) == 0 or len(right_fitx) == 0 or len(ploty) == 0:
        return original_image
    

    try:
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        if deviation is not None and deviation > 0 and deviation > 0.1 and deviation < 0.3:
            left_color = (0, 255, 255)
            right_color = (0, 0, 255)
        elif deviation is not None and deviation < 0 and deviation < -0.1 and deviation > -0.3:
            left_color = (0, 0, 255)
            right_color = (0, 255, 255)
        else:
            left_color = (0, 255, 255)
            right_color = (0, 255, 255)

        cv2.polylines(color_warp, [pts_left.astype(np.int32)], isClosed=False, color=left_color, thickness=3)

        cv2.polylines(color_warp, [pts_right.astype(np.int32)], isClosed=False, color=right_color, thickness=3)

        center_fitx = (left_fitx + right_fitx) / 2
        center_pts = np.array([np.transpose(np.vstack([center_fitx, ploty]))]).astype(np.int32)
        cv2.polylines(color_warp, center_pts, isClosed=False, color=(255, 0, 0), thickness=5)

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0])) 
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
        
        return result
    except Exception as e:
        print(f"Error in draw_lane_overlay: {e}")
        return original_image


def add_text_overlay(image, left_curverad, right_curverad, deviation, avg_brightness, speed):
    """
    Add text overlay with lane curvature, deviation, average brightness, and speed.
    Args:
        image: Image to add text overlay on
        left_curverad: Left lane line curvature in meters
        right_curverad: Right lane line curvature in meters
        deviation: Vehicle deviation from lane center in meters
        avg_brightness: Average brightness of the image
        speed: Vehicle speed in km/h
    Returns:
        Image with text overlay
    """

    fontType = cv2.FONT_HERSHEY_SIMPLEX
    
    if left_curverad is None or right_curverad is None:
        curvature_text = "Curvature: N/A"
    else:
        curvature_text = f"Curvature: L={left_curverad:.1f}m, R={right_curverad:.1f}m"
    
    if deviation is None:
        deviation_text = "Deviation: N/A"
    else:
        direction = '+' if deviation > 0 else '-'
        deviation_text = f"Deviation: {direction}{abs(deviation):.2f}m"
    
    cv2.putText(image, curvature_text, (30, 60), fontType, 1.2, (255, 255, 255), 1)
    cv2.putText(image, deviation_text, (30, 110), fontType, 1.2, (255, 255, 255), 1)

    cv2.putText(image, f"Avg Brightness: {avg_brightness:.1f}", (30, 160), fontType, 1.2, (255, 255, 255), 1)

    cv2.putText(image, f"Speed: {speed:.1f} kph", (30, 210), fontType, 1.2, (255, 255, 255), 1)
    
    return image

def create_mask_overlay(img, mask, alpha=0.4, color=(0, 255, 0)):
    """
    Create an overlay of a binary mask on the original image.
    
    Args:
        img: The original image (BGR)
        mask: Binary mask (0s and 1s)
        alpha: Transparency of the overlay (0.0 to 1.0)
        color: Color of the mask overlay (BGR tuple)
    
    Returns:
        Image with mask overlay
    """
    try:
        # Ensure mask is the right size
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        overlay = img.copy()
        mask_bool = mask > 0
        
        for c in range(3):
            overlay[..., c] = np.where(
                mask_bool,
                (1 - alpha) * overlay[..., c] + alpha * color[c],
                overlay[..., c]
            )
        
        return overlay.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in create_mask_overlay: {e}")
        return img
