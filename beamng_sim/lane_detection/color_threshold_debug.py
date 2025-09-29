import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import math
from beamng_sim.lane_detection.thresholding import gradient_thresholds, color_threshold, combine_thresholds


def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)


def compute_avg_brightness(frame, src_points=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if src_points is not None:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        avg_brightness = np.mean(gray[mask == 1])
    else:
        avg_brightness = np.mean(gray)
    return avg_brightness


def get_src_points(image_shape, speed=0):
    h, w = image_shape[:2]
    ref_w, ref_h = 1278, 720
    scale_w = w / ref_w
    scale_h = h / ref_h
    left_bottom  = [118, 590]
    right_bottom = [1077, 590]
    top_right    = [730, 408]
    top_left     = [519, 408]
    speed_norm = min(speed / 120.0, 1.0)
    top_shift = -40 * speed_norm
    side_shift = 100 * speed_norm
    src = np.float32([
        [left_bottom[0] * scale_w, left_bottom[1] * scale_h],
        [right_bottom[0] * scale_w, right_bottom[1] * scale_h],
        [(top_right[0] - side_shift) * scale_w, (top_right[1] + top_shift) * scale_h],
        [(top_left[0] + side_shift) * scale_w,  (top_left[1]  + top_shift) * scale_h]
    ])
    return src


def process_frame(frame, src_points):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_brightness = compute_avg_brightness(frame, src_points)
    
    grad_binary = gradient_thresholds(frame, avg_brightness=avg_brightness)
    color_bin = color_threshold(frame, avg_brightness=avg_brightness)
    combined_binary = combine_thresholds(color_bin, grad_binary, avg_brightness=avg_brightness)
    
    w_h_min, w_h_max = 0, 180
    w_s_min, w_s_max = 0, 25  # Reduced to avoid off-white road surfaces
    w_v_min, w_v_max = 200, 255  # Higher minimum to focus on brighter white markings
    
    y_h_min, y_h_max = 10, 45  # Wider hue range for yellow
    y_s_min, y_s_max = 50, 255  # Lower saturation threshold to catch faded yellow
    y_v_min, y_v_max = 100, 255
    
    s_h_min, s_h_max = 0, 180
    s_s_min, s_s_max = 0, 20
    s_v_min, s_v_max = 110, 150
    
    if avg_brightness > 200:  # Very bright conditions (direct sunlight)
        w_s_max = min(w_s_max, 15)  # More restrictive on saturation
        w_v_min = 220  # Higher minimum value to only catch bright white lines
        y_s_min = 100  # Keep yellow detection robust
        
    elif avg_brightness > 170:
        w_v_min = 200  # Higher to focus on true white markings
        w_s_max = 20  # More restrictive
        
    elif 100 < avg_brightness < 170:
        w_v_min = 190  # Higher to avoid road noise
        w_s_max = 22  # More restrictive

    elif 70 < avg_brightness <= 100:
        w_v_min = 170  # Higher for better noise reduction
        w_s_max = 25  # Still restrictive
        s_v_max = 160

    elif avg_brightness <= 70:  # Low light conditions
        w_v_min = 140  # Lower but still filtering noise
        w_s_max = 30  # Allow for some off-white in dark conditions
        y_v_min = 90   # More permissive for yellow in low light
        y_s_min = 40   # More permissive for yellow in low light
        s_v_max = 150
    
    white_lower = np.array([w_h_min, w_s_min, w_v_min])
    white_upper = np.array([w_h_max, w_s_max, w_v_max])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    yellow_lower = np.array([y_h_min, y_s_min, y_v_min])
    yellow_upper = np.array([y_h_max, y_s_max, y_v_max])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    yellow_kernel = np.ones((5,3), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, yellow_kernel, iterations=1)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    shadow_mask = np.zeros_like(white_mask)
    if 60 < avg_brightness < 120:
        shadow_lower = np.array([s_h_min, s_s_min, s_v_min])
        shadow_upper = np.array([s_h_max, s_s_max, s_v_max])
        shadow_mask = cv2.inRange(hsv, shadow_lower, shadow_upper)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    
    white_display = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    yellow_display = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
    shadow_display = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
    
    grad_display = np.dstack((grad_binary, grad_binary, grad_binary)) * 255
    color_display = np.dstack((color_bin, color_bin, color_bin)) * 255
    
    combined_display_array = combined_binary.astype(np.uint8)
    combined_display = np.dstack((combined_display_array, combined_display_array, combined_display_array)) * 255
    
    method_vis = np.zeros_like(frame)
    method_vis[color_bin == 1] = [255, 0, 0]  # Red for color detection
    method_vis[(color_bin == 0) & (grad_binary == 1)] = [0, 255, 0]  # Green for gradient detection
    method_vis[(color_bin == 1) & (grad_binary == 1)] = [0, 255, 255]  # Cyan for both methods
    
    final_vis = np.zeros_like(frame)
    final_vis[combined_binary.astype(bool) == 1] = [255, 255, 255]  # White for final output
    
    return {
        'frame': frame,
        'white_mask': white_display,
        'yellow_mask': yellow_display,
        'shadow_mask': shadow_display,
        'grad_binary': grad_display,
        'color_binary': color_display,
        'combined': combined_display,
        'method_vis': method_vis,
        'final_vis': final_vis,
        'avg_brightness': avg_brightness
    }


try:
    print("Initializing BeamNG...")
    beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
    print("Opening BeamNG connection...")
    beamng.open()
    print("Connection opened successfully")

    scenario = Scenario('west_coast_usa', 'lane_detection_city')
    print("Creating scenario...")
    vehicle = Vehicle('ego_vehicle', model='etk800', licence='JULIAN')

    rot_city = yaw_to_quat(-133.506 + 180)
    rot_highway = yaw_to_quat(-135.678)

    print("Adding vehicle to scenario...")
    scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)

    print("Making scenario...")
    scenario.make(beamng)
    print("Loading scenario...")
    beamng.scenario.load(scenario)
    print("Starting scenario...")
    beamng.scenario.start()
    print("Scenario started!")
except Exception as e:
    print(f"ERROR during BeamNG setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Setting up camera...")
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
    print("Camera setup complete")
except Exception as e:
    print(f"ERROR during camera setup: {e}")
    import traceback
    traceback.print_exc()
    beamng.close()
    sys.exit(1)

print("Waiting for camera to provide frames...")
frame = None
max_attempts = 50
attempts = 0

while frame is None and attempts < max_attempts:
    try:
        print(f"Polling camera (attempt {attempts+1}/{max_attempts})...")
        images = camera.poll()
        if images and 'colour' in images:
            frame = np.array(images['colour'])
            print("Successfully received camera frame!")
        else:
            print("No colour image data received")
    except Exception as e:
        print(f"Error polling camera: {e}")
        frame = None
    
    attempts += 1
    cv2.waitKey(100)

if frame is None:
    print("ERROR: Could not get camera frames after multiple attempts")
    beamng.close()
    sys.exit(1)

cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Shadow Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Gradient Binary", cv2.WINDOW_NORMAL)
cv2.namedWindow("Color Binary", cv2.WINDOW_NORMAL)
cv2.namedWindow("Combined Binary", cv2.WINDOW_NORMAL)
cv2.namedWindow("Method Contributions", cv2.WINDOW_NORMAL)
cv2.namedWindow("Final Output", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Original Frame", 640, 360)
cv2.resizeWindow("White Mask", 640, 360)
cv2.resizeWindow("Yellow Mask", 640, 360)
cv2.resizeWindow("Shadow Mask", 640, 360)
cv2.resizeWindow("Gradient Binary", 640, 360)
cv2.resizeWindow("Color Binary", 640, 360)
cv2.resizeWindow("Combined Binary", 640, 360)
cv2.resizeWindow("Method Contributions", 640, 360)
cv2.resizeWindow("Final Output", 640, 360)

print("Starting main loop...")
frame_counter = 0
tod_values = [0.2, 0.5, 0.7, 0.85, 0.0]
tod_names = ["Morning", "Midday", "Afternoon", "Evening", "Night"]
tod_index = 0
frames_per_tod = 200

while True:
    try:
        beamng.control.step(10)
        images = camera.poll()
        if not images or 'colour' not in images:
            print("Warning: No valid images received from camera")
            cv2.waitKey(100)
            continue

        frame = np.array(images['colour'])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src_points = get_src_points(frame.shape, speed=0)

        if frame_counter % frames_per_tod == 0:
            tod_index = (frame_counter // frames_per_tod) % len(tod_values)
            beamng.set_tod(tod=tod_values[tod_index], play=False)
            print(f"\n--- Time of Day changed to {tod_names[tod_index]} (tod={tod_values[tod_index]}) ---\n")

        frame_counter += 1
        
        results = process_frame(frame_rgb, src_points)
        
        cv2.putText(frame, f"Avg Brightness: {results['avg_brightness']:.1f}", 
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Original Frame", frame)
        cv2.imshow("White Mask", results['white_mask'])
        cv2.imshow("Yellow Mask", results['yellow_mask'])
        cv2.imshow("Shadow Mask", results['shadow_mask'])
        cv2.imshow("Gradient Binary", results['grad_binary'])
        cv2.imshow("Color Binary", results['color_binary'])
        cv2.imshow("Combined Binary", results['combined'])
        cv2.imshow("Method Contributions", results['method_vis'])
        cv2.imshow("Final Output", results['final_vis'])

        if frame_counter % 30 == 0:
            print(f"Brightness: {results['avg_brightness']:.1f}")
        
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        cv2.waitKey(100)
        continue

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
beamng.close()