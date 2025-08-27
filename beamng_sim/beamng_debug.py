import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lane_detection.main import process_frame
from lane_detection.thresholding import apply_thresholds_debug
from lane_detection.perspective import perspective_warp
from lane_detection.lane_finder import get_histogram, sliding_window_search
from pid_controller import PIDController
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from config.config import BEAMNG_HOME
import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os


def yaw_to_quat(yaw_deg):
    """Convert yaw angle in degrees to quaternion."""
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)


def save_debug_images(frame_count, img_bgr, result, grad_binary, color_binary, combined_binary, 
                     binary_warped, left_points, right_points, left_fitx, right_fitx, ploty, 
                     deviation, steering, src_points=None):
    """Save debug images every 30 frames."""
    if frame_count % 30 != 0:
        return
    
    debug_dir = "C:\Users\user\Documents\github\self-driving-car-simulation\beamng_sim\debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create a combined debug image
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image with source points
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Draw source points on the image
    if src_points is not None:
        # Create a copy for visualization
        src_img = rgb_img.copy()
        
        # Draw the quadrilateral
        src_points = src_points.astype(np.int32)
        pts = src_points.reshape((-1, 1, 2))
        cv2.polylines(src_img, [pts], True, (255, 0, 0), 2)
        
        # Add dots at the corners for clarity
        for pt in src_points:
            cv2.circle(src_img, (pt[0], pt[1]), 5, (0, 255, 0), -1)
        
        # Show the image with source points
        axes[0, 0].imshow(src_img)
    else:
        axes[0, 0].imshow(rgb_img)
    
    axes[0, 0].set_title('Original with Source Points')
    axes[0, 0].axis('off')
    
    # Gradient binary
    axes[0, 1].imshow(grad_binary, cmap='gray')
    axes[0, 1].set_title('Gradient Binary')
    axes[0, 1].axis('off')
    
    # Color binary
    axes[0, 2].imshow(color_binary, cmap='gray')
    axes[0, 2].set_title('Color Binary')
    axes[0, 2].axis('off')
    
    # Combined binary
    axes[1, 0].imshow(combined_binary, cmap='gray')
    axes[1, 0].set_title('Combined Binary')
    axes[1, 0].axis('off')
    
    # Warped with lane points
    # Ensure binary_warped is not empty and has proper dimensions
    if binary_warped is not None and binary_warped.size > 0:
        warped_colored = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    else:
        # Create a blank image if binary_warped is empty
        warped_colored = np.zeros((360, 640, 3), dtype=np.uint8)
        print("Warning: binary_warped is empty or invalid")
    
    # Print number of lane points found for debugging
    left_count = len(left_points[0]) if left_points is not None and len(left_points) > 0 else 0
    right_count = len(right_points[0]) if right_points is not None and len(right_points) > 0 else 0
    print(f"Lane points: left={left_count}, right={right_count}")
    
    if left_count > 0:
        # Red for left lane points
        warped_colored[left_points[0], left_points[1]] = [255, 0, 0]
    
    if right_count > 0:
        # Blue for right lane points
        warped_colored[right_points[0], right_points[1]] = [0, 0, 255]
    
    axes[1, 1].imshow(warped_colored.astype(np.uint8))
    
    # If no points found, add an explanatory text
    if left_count == 0 and right_count == 0:
        # Add text to explain the issue
        text_x = binary_warped.shape[1] // 2
        text_y = binary_warped.shape[0] // 2
        axes[1, 1].text(text_x, text_y, "No lane pixels detected", 
                    horizontalalignment='center', 
                    color='red', 
                    fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.6))
    
    axes[1, 1].set_title(f'Warped + Lane Points ({left_count}/{right_count})')
    axes[1, 1].axis('off')
    
    # Lane fits with lane area visualization
    # Create a lane image even if we don't have sufficient lane points
    lane_img = np.zeros_like(binary_warped)
    lane_img = np.dstack((lane_img, lane_img, lane_img))
    
    # Add a gray background for better visibility
    lane_img[:,:] = (40, 40, 40)  # Dark gray background
    
    # Only draw lane area if we have sufficient lane points
    if len(left_fitx) > 0 and len(right_fitx) > 0 and len(ploty) > 0 and len(left_fitx) == len(ploty) and len(right_fitx) == len(ploty):
        try:
            # Fill the lane area
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(lane_img, np.int_([pts]), (0, 128, 0))  # Green lane area
        except Exception as e:
            print(f"Error drawing lane area: {e}")
        
        try:
            # Draw lane lines
            cv2.polylines(lane_img, np.int32([pts_left]), False, (255, 0, 0), 8)  # Red left line
            cv2.polylines(lane_img, np.int32([pts_right]), False, (0, 0, 255), 8)  # Blue right line
            
            # Add lane image as background
            axes[1, 2].imshow(lane_img)
            
            # Plot fit lines on top
            axes[1, 2].plot(left_fitx, ploty, 'r-', linewidth=2, label='Left Lane')
            axes[1, 2].plot(right_fitx, ploty, 'b-', linewidth=2, label='Right Lane')
        except Exception as e:
            print(f"Error drawing lane lines: {e}")
            # Add lane image as background even if lines fail
            axes[1, 2].imshow(lane_img)
        
        # Add lane center and vehicle center lines
        try:
            if len(left_fitx) > 0 and len(right_fitx) > 0:
                lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
                vehicle_center = binary_warped.shape[1] / 2
                axes[1, 2].axvline(x=lane_center, color='g', linestyle='-', linewidth=2, label='Lane Center')
                axes[1, 2].axvline(x=vehicle_center, color='y', linestyle='--', linewidth=2, label='Vehicle Center')
            else:
                # Only draw vehicle center if lane center can't be calculated
                vehicle_center = binary_warped.shape[1] / 2
                axes[1, 2].axvline(x=vehicle_center, color='y', linestyle='--', linewidth=2, label='Vehicle Center')
        except Exception as e:
            print(f"Error drawing center lines: {e}")
            # Fallback to only vehicle center
            vehicle_center = binary_warped.shape[1] / 2
            axes[1, 2].axvline(x=vehicle_center, color='y', linestyle='--', linewidth=2, label='Vehicle Center')
        
        title = f'Lane Fits\nDev: {deviation:.3f}m, Steer: {steering:.3f}'
    else:
        # Display a message when insufficient lane points are detected
        axes[1, 2].imshow(lane_img)
        vehicle_center = binary_warped.shape[1] / 2
        axes[1, 2].axvline(x=vehicle_center, color='y', linestyle='--', linewidth=2, label='Vehicle Center')
        
        # Add text to explain the issue
        text_x = binary_warped.shape[1] // 2
        text_y = binary_warped.shape[0] // 2
        axes[1, 2].text(text_x, text_y, "Insufficient lane pixels", 
                    horizontalalignment='center', 
                    color='white', 
                    fontsize=12)
        
        # Set title with warning
        title = f'Lane Fits\nNo lanes detected'
    
    axes[1, 2].set_xlim(0, binary_warped.shape[1])
    axes[1, 2].set_ylim(binary_warped.shape[0], 0)
    axes[1, 2].set_title(title)
    axes[1, 2].legend(loc='upper right', fontsize=8)
    axes[1, 2].grid(False)
    
    plt.tight_layout()
    plt.savefig(f"{debug_dir}/debug_frame_{frame_count:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug image saved: debug_frame_{frame_count:04d}.png")


def get_lane_points_direct(binary_warped):
    """Extract lane points directly from the binary image for visualization."""
    # Check if binary_warped has any nonzero pixels
    if np.sum(binary_warped) == 0:
        # Return empty points if no lane pixels detected
        return (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    # Get all nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Create histogram of bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0]//2)
    
    # Check if we have any nonzero pixels in each half
    left_hist_max = np.max(histogram[:midpoint]) if midpoint > 0 else 0
    right_hist_max = np.max(histogram[midpoint:]) if midpoint < len(histogram) else 0
    
    # If no significant peaks found, try a broader search
    min_peak_value = 5  # Minimum value to consider a valid peak
    
    if left_hist_max < min_peak_value or right_hist_max < min_peak_value:
        # If no significant peaks, use full width search
        left_points = (nonzeroy[nonzerox < midpoint], nonzerox[nonzerox < midpoint])
        right_points = (nonzeroy[nonzerox >= midpoint], nonzerox[nonzerox >= midpoint])
        return left_points, right_points
    
    # Find peaks for lane bases
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Define width of search window +/- margin
    margin = int(binary_warped.shape[1] * 0.15)  # Increased to 15% of image width
    
    # Create more accurate left/right separation
    left_lane_pixels = (nonzerox > max(0, leftx_base - margin)) & (nonzerox < leftx_base + margin)
    right_lane_pixels = (nonzerox > rightx_base - margin) & (nonzerox < min(binary_warped.shape[1]-1, rightx_base + margin))
    
    left_points = (nonzeroy[left_lane_pixels], nonzerox[left_lane_pixels])
    right_points = (nonzeroy[right_lane_pixels], nonzerox[right_lane_pixels])
    
    return left_points, right_points


def get_perspective_points(img, speed=0):
    """Get source and destination points for perspective transform directly from perspective module logic."""
    img_size = (img.shape[1], img.shape[0])
    w, h = img_size
    
    # Baseline city src points - same as in perspective.py
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
    
    # Scale points for actual frame size if different from 1280x720
    scale_x = w / 1280  # Assuming original is 1280 width
    scale_y = h / 720   # Assuming original is 720 height
    
    src_scaled = np.copy(src)
    src_scaled[:, 0] = src[:, 0] * scale_x
    src_scaled[:, 1] = src[:, 1] * scale_y
    
    return src_scaled, dst

def process_frame_with_debug(frame, frame_count):
    """Process frame and extract debug data."""
    # Step 1: Apply thresholds
    grad_binary, color_binary, combined_binary = apply_thresholds_debug(frame)
    
    # Get source points properly
    src_scaled, dst = get_perspective_points(frame)
    
    # Step 2: Apply perspective transform
    binary_warped, Minv = perspective_warp(combined_binary, debugger=None)
    
    # Step 3: Find lane lines
    histogram = get_histogram(binary_warped)
    ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(binary_warped, histogram, None)
    
    # Extract ALL lane points directly from the binary warped image for better visualization
    left_points, right_points = get_lane_points_direct(binary_warped)
    
    # Check if we have enough lane points to proceed
    left_count = len(left_points[0]) if left_points is not None and len(left_points) > 0 else 0
    right_count = len(right_points[0]) if right_points is not None and len(right_points) > 0 else 0
    
    # If we have insufficient lane points, print a warning
    if left_count < 10 or right_count < 10:
        print(f"Insufficient lane pixels: left={left_count}, right={right_count}")
    
    # Use the main process_frame for final result and metrics
    result, metrics = process_frame(frame, None)
    
    # Return source points for visualization
    return result, metrics, grad_binary, color_binary, combined_binary, binary_warped, left_points, right_points, left_fitx, right_fitx, ploty, src_scaled


def main():
    # Initialize BeamNG
    beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
    beamng.open()

    # Create scenario and vehicle
    scenario = Scenario('west_coast_usa', 'lane_detection')
    vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')

    # Spawn positions
    rot_city = yaw_to_quat(-133.506 + 180)
    rot_highway = yaw_to_quat(-135.678)

    # Choose spawn location
    # Street Spawn
    scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)
    
    # Highway Spawn (uncomment to use)
    # scenario.add_vehicle(vehicle, pos=(-287.210, 73.609, 112.363), rot_quat=rot_highway)

    scenario.make(beamng)
    beamng.scenario.load(scenario)
    beamng.scenario.start()

    # Setup camera
    camera = Camera(
        'front_cam',
        beamng,
        vehicle,
        requested_update_time=0.01,
        is_using_shared_memory=True,
        pos=(0, -1.3, 1.4),
        dir=(0, -1, 0),
        field_of_view_y=90,
        near_far_planes=(0.1, 1000),
        resolution=(640, 360),
        is_streaming=True,
        is_render_colours=True,
    )

    # Simple PID controller for steering
    pid = PIDController(Kp=0.8, Ki=0.02, Kd=0.15)

    # Control parameters
    base_throttle = 0.15
    steering_bias = 0.02
    max_steering_change = 0.1
    
    # State variables
    previous_steering = 0.0
    last_time = time.time()
    frame_count = 0
    smooth_deviation = 0.0
    alpha = 0.1

    try:
        for step_i in range(1000):
            # Timing
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Step simulation
            beamng.control.step(10)
            
            # Get camera image
            images = camera.stream()
            img = np.array(images['colour'])
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # We'll use the get_perspective_points function instead of duplicating code
            # This will happen in process_frame_with_debug
            
            # Process frame with debug data extraction
            result, metrics, grad_binary, color_binary, combined_binary, binary_warped, left_points, right_points, left_fitx, right_fitx, ploty, src_scaled = process_frame_with_debug(img_bgr, frame_count)

            # Get deviation safely
            if metrics and 'deviation' in metrics and metrics['deviation'] is not None:
                raw_deviation = metrics['deviation']
            else:
                raw_deviation = 0.0

            # Sanity check and smoothing
            if abs(raw_deviation) > 1.0:
                raw_deviation = np.clip(raw_deviation, -1.0, 1.0)

            smooth_deviation = alpha * raw_deviation + (1.0 - alpha) * smooth_deviation

            # PID control
            steering = pid.update(-smooth_deviation, dt)
            steering += steering_bias
            steering = np.clip(steering, -1.0, 1.0)
            
            # Limit rate of change
            steering_change = steering - previous_steering
            if abs(steering_change) > max_steering_change:
                steering = previous_steering + np.sign(steering_change) * max_steering_change
            
            previous_steering = steering

            # Throttle control
            throttle = base_throttle * (1.0 - 0.3 * abs(steering))
            throttle = np.clip(throttle, 0.05, 0.3)

            # Save debug images every 30 frames
            save_debug_images(frame_count, img_bgr, result, grad_binary, color_binary, combined_binary,
                            binary_warped, left_points, right_points, left_fitx, right_fitx, ploty,
                            raw_deviation, steering, src_scaled)

            # Apply controls
            vehicle.control(steering=steering, throttle=throttle, brake=0.0)

            # Display image
            cv2.imshow('Lane Detection', result)

            # Print status every 30 frames
            if step_i % 30 == 0:
                print(f"[{step_i}] Dev: {raw_deviation:.3f}m | Smooth: {smooth_deviation:.3f}m | Steering: {steering:.3f}")

            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        vehicle.control(throttle=0, steering=0, brake=1.0)
        cv2.destroyAllWindows()
        beamng.close()
        print(f"Debug images saved to: debug_output/")


if __name__ == "__main__":
    main()
