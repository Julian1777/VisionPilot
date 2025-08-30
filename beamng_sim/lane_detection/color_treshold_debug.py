import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import math


def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)

USE_AUTO_BRIGHTNESS = False  # Set to False for manual/static thresholding

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

def nothing(x):
    pass

# Create a window
cv2.namedWindow("Trackbars")

# White mask trackbars
cv2.createTrackbar("White H min", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("White H max", "Trackbars", 80, 180, nothing)
cv2.createTrackbar("White S min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("White S max", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("White V min", "Trackbars", 170, 255, nothing)
cv2.createTrackbar("White V max", "Trackbars", 255, 255, nothing)

# Yellow mask trackbars
cv2.createTrackbar("Yellow H min", "Trackbars", 15, 180, nothing)
cv2.createTrackbar("Yellow H max", "Trackbars", 35, 180, nothing)
cv2.createTrackbar("Yellow S min", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("Yellow S max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Yellow V min", "Trackbars", 180, 255, nothing)
cv2.createTrackbar("Yellow V max", "Trackbars", 255, 255, nothing)

# Shadow mask trackbars
cv2.createTrackbar("Shadow H min", "Trackbars", 90, 180, nothing)
cv2.createTrackbar("Shadow H max", "Trackbars", 150, 180, nothing)
cv2.createTrackbar("Shadow S min", "Trackbars", 15, 255, nothing)
cv2.createTrackbar("Shadow S max", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("Shadow V min", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("Shadow V max", "Trackbars", 255, 255, nothing)


beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
beamng.open()

#scenario = Scenario('west_coast_usa', 'lane_detection_city')
scenario = Scenario('west_coast_usa', 'lane_detection_highway')
vehicle = Vehicle('ego_vehicle', model='etk800', licence='JULIAN')
#vehicle = Vehicle('Q8', model='adroniskq8', licence='JULIAN')

# Spawn positions rotation conversion
rot_city = yaw_to_quat(-133.506 + 180)
rot_highway = yaw_to_quat(-135.678)

# Street Spawn
#scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)

# Highway Spawn
scenario.add_vehicle(vehicle, pos=(-287.210, 73.609, 112.363), rot_quat=rot_highway)

scenario.make(beamng)
beamng.scenario.load(scenario)
beamng.scenario.start()

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

while True:
    images = camera.poll()
    frame = np.array(images['colour'])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    src_points = get_src_points(frame.shape, speed=0)  # speed=0 for static test

    if USE_AUTO_BRIGHTNESS:
        avg_brightness = compute_avg_brightness(frame, src_points=src_points)
        # Auto-adaptive thresholds
        w_h_min, w_h_max = 0, 80
        w_s_min, w_s_max = 0, 80
        if avg_brightness > 180:
            w_v_min = 200
        elif avg_brightness < 100:
            w_v_min = 160
        else:
            w_v_min = 170
        w_v_max = 255

        y_h_min, y_h_max = 15, 35
        y_s_min, y_s_max = 80, 255
        y_v_min, y_v_max = 180, 255

        s_h_min, s_h_max = 90, 150
        s_s_min, s_s_max = 15, 80
        s_v_min, s_v_max = 150, 255
    else:
        # Manual mode: get values from trackbars
        w_h_min = cv2.getTrackbarPos("White H min", "Trackbars")
        w_h_max = cv2.getTrackbarPos("White H max", "Trackbars")
        w_s_min = cv2.getTrackbarPos("White S min", "Trackbars")
        w_s_max = cv2.getTrackbarPos("White S max", "Trackbars")
        w_v_min = cv2.getTrackbarPos("White V min", "Trackbars")
        w_v_max = cv2.getTrackbarPos("White V max", "Trackbars")

        y_h_min = cv2.getTrackbarPos("Yellow H min", "Trackbars")
        y_h_max = cv2.getTrackbarPos("Yellow H max", "Trackbars")
        y_s_min = cv2.getTrackbarPos("Yellow S min", "Trackbars")
        y_s_max = cv2.getTrackbarPos("Yellow S max", "Trackbars")
        y_v_min = cv2.getTrackbarPos("Yellow V min", "Trackbars")
        y_v_max = cv2.getTrackbarPos("Yellow V max", "Trackbars")

        s_h_min = cv2.getTrackbarPos("Shadow H min", "Trackbars")
        s_h_max = cv2.getTrackbarPos("Shadow H max", "Trackbars")
        s_s_min = cv2.getTrackbarPos("Shadow S min", "Trackbars")
        s_s_max = cv2.getTrackbarPos("Shadow S max", "Trackbars")
        s_v_min = cv2.getTrackbarPos("Shadow V min", "Trackbars")
        s_v_max = cv2.getTrackbarPos("Shadow V max", "Trackbars")

    # Masks
    white_mask = cv2.inRange(hsv, np.array([w_h_min, w_s_min, w_v_min]), np.array([w_h_max, w_s_max, w_v_max]))
    yellow_mask = cv2.inRange(hsv, np.array([y_h_min, y_s_min, y_v_min]), np.array([y_h_max, y_s_max, y_v_max]))
    shadow_mask = cv2.inRange(hsv, np.array([s_h_min, s_s_min, s_v_min]), np.array([s_h_max, s_s_max, s_v_max]))

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)

    if USE_AUTO_BRIGHTNESS:
        cv2.putText(frame, f"Avg Brightness: {avg_brightness:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Mask", combined_mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
beamng.close()
