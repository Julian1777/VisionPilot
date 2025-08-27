import numpy as np
import cv2


def draw_lane_overlay(original_image, warped_image, Minv, left_fitx, right_fitx, ploty):
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    center_fitx = (left_fitx + right_fitx) / 2
    center_pts = np.array([np.transpose(np.vstack([center_fitx, ploty]))]).astype(np.int32)
    cv2.polylines(color_warp, center_pts, isClosed=False, color=(255, 0, 0), thickness=5)

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0])) 
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    
    return result


def add_text_overlay(image, left_curverad, right_curverad, deviation):
    """Add text information to image."""
    direction = '+' if deviation > 0 else '-'
    curvature_text = f"Curvature: L={left_curverad:.1f}m, R={right_curverad:.1f}m"
    deviation_text = f"Deviation: {direction}{abs(deviation):.2f}m"
    
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, curvature_text, (30, 60), fontType, 1.2, (255, 255, 255), 2)
    cv2.putText(image, deviation_text, (30, 110), fontType, 1.2, (255, 255, 255), 2)
    
    return image
