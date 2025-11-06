import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beamng_sim.utils.pid_controller import PIDController

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Lidar, Radar

from beamng_sim.sign.detect_classify import random_brightness

from config.config import SIGN_DETECTION_MODEL, SIGN_CLASSIFICATION_MODEL, VEHICLE_PEDESTRIAN_MODEL, UNET_LANE_DETECTION_MODEL, SCNN_LANE_DETECTION_MODEL, CAMERA_CALIBRATION

from ultralytics import YOLO
import tensorflow as tf
import torch
import numpy as np
import time
import math
import cv2
import csv
import datetime

from beamng_sim.lane_detection.main import process_frame_cv as lane_detection_cv_process_frame
from beamng_sim.lane_detection.main import process_frame_scnn as lane_detection_scnn_process_frame
from beamng_sim.sign.main import process_frame as sign_process_frame
from beamng_sim.vehicle_obstacle.main import process_frame as vehicle_obstacle_process_frame
from beamng_sim.lidar.main import process_frame as lidar_process_frame
from beamng_sim.radar.main import process_frame as radar_process_frame

from beamng_sim.lane_detection.fusion import fuse_lane_metrics
from beamng_sim.lane_detection.perspective import load_calibration

from beamng_sim.foxglove_integration.bridge_instance import bridge

MODELS = {}

def yaw_to_quat(yaw_deg):
    """
    Convert yaw angle in degrees to a quaternion representation for vehicle orientation.
    Args:
        yaw_deg (float): Yaw angle in degrees
    Returns:
        tuple: Quaternion (x, y, z, w)
    """
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)

def load_models():
    """
    Load all the models into a global dictionary for use in detection or classification.
    This way models are only loaded once.
    """
    global CAMERA_CALIBRATION
    
    print("Loading models")

    # Sign detection model
    MODELS['sign_detect'] = YOLO(str(SIGN_DETECTION_MODEL))
    print("Sign detection model loaded")

    # Sign classification model with custom objects used during training
    MODELS['sign_classify'] = tf.keras.models.load_model(
        str(SIGN_CLASSIFICATION_MODEL), 
        custom_objects={"random_brightness": random_brightness}
    )
    print("Sign classification model loaded")
    
    # Vehicle detection model 
    MODELS['vehicle'] = YOLO(str(VEHICLE_PEDESTRIAN_MODEL))
    print("Vehicle detection model loaded")

    # Lane detection UNET model
    MODELS['lane_unet'] = tf.keras.models.load_model(str(UNET_LANE_DETECTION_MODEL))
    print("Lane detection UNET model loaded")
    
    # Lane detection SCNN model
    from beamng_sim.lane_detection.scnn.scnn_model import SCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"SCNN will run on: {device}")
    
    scnn_model = SCNN(input_size=(800, 288), pretrained=False)
    checkpoint = torch.load(str(SCNN_LANE_DETECTION_MODEL), map_location=device)
    scnn_model.load_state_dict(checkpoint['net'])
    scnn_model = scnn_model.to(device)
    scnn_model.eval()
    
    MODELS['lane_scnn'] = scnn_model
    MODELS['scnn_device'] = device
    print(f"Lane detection SCNN model loaded (device: {device})")
    
    # Load camera calibration
    try:
        CAMERA_CALIBRATION = load_calibration(str(CAMERA_CALIBRATION))
        print("Camera calibration loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load camera calibration from {str(CAMERA_CALIBRATION)}: {e}")
        CAMERA_CALIBRATION = None
    print("All models loaded successfully!")


def sim_setup():
    """
    Setup BeamNG simulation, scenario, vehicle, spawn point and sensors.
    """

    beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
    beamng.open()

    #scenario = Scenario('west_coast_usa', 'lane_detection_city')
    scenario = Scenario('west_coast_usa', 'lane_detection_highway')

    vehicle = Vehicle('ego_vehicle', model='etk800', licence='JULIAN')
    #vehicle = Vehicle('Q8', model='rsq8_600_tfsi', licence='JULIAN')

    # Spawn positions rotation conversion
    rot_city = yaw_to_quat(-133.506 + 180)
    rot_highway = yaw_to_quat(-135.678)

    # Street Spawn
    #scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)
    
    # Highway Spawn
    scenario.add_vehicle(vehicle, pos=(-287.210, 73.609, 112.363), rot_quat=rot_highway)

    scenario.make(beamng)

    beamng.settings.set_deterministic(60)

    beamng.scenario.load(scenario)
    beamng.scenario.start()
    

    try:
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
        print("Camera initialized")
    except Exception as e:
        print(f"Camera initialization error: {e}")
        camera = None
    try:
        lidar = Lidar(
            "lidar1",
            beamng,
            vehicle,
            requested_update_time=0.01,
            is_using_shared_memory=True,
            is_rotate_mode=False,
            horizontal_angle=170,  # Horizontal field of view
            vertical_angle=30,  # Vertical field of view
            vertical_resolution=64,  # Number of lasers/channels
            density=7,
            frequency=15,
            max_distance=100,
            pos=(0, -0.35, 1.425),
            is_visualised=False,
        )
        print("LiDAR initialized")
    except Exception as e:
        print(f"LiDAR initialization error: {e}")
        lidar = None
    
    try:
        print("Attempting Radar initialization...")
        radar = Radar(
            "radar1",
            beamng,
            vehicle,
            requested_update_time=0.01,
            pos=(0, -2.5, 0.5),
            dir=(0, -1, 0),
            up=(0, 0, 1),
            size=(200, 200),
            near_far_planes=(2, 120),
            field_of_view_y=70,
        )
        print("Radar initialized")
    except Exception as e:
        print(f"  Radar initialization error: {e}")
        radar = None

    return beamng, scenario, vehicle, camera, lidar, radar

def get_vehicle_speed(vehicle):
    """
    Get the vehicle speed in m/s and kph, and also return position.
    Args:
        vehicle (Vehicle): BeamNG vehicle object
    Returns:
        tuple: (speed_mps, speed_kph, position)
    """

    vehicle.poll_sensors()
    if 'vel' in vehicle.state:
        speed_mps = vehicle.state['vel'][0]
        speed_kph = speed_mps * 3.6
    else:
        speed_mps = 0.0
        speed_kph = 0.0

    if 'pos' in vehicle.state:
        position = vehicle.state['pos']
        print(f"Vehicle position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
    else:
        print("Vehicle position not available")
        position = None

    if 'dir' in vehicle.state:
        direction = vehicle.state['dir']
        print(f"Vehicle direction: x={direction[0]:.2f}, y={direction[1]:.2f}, z={direction[2]:.2f}")

    return speed_mps, speed_kph, position, direction


def lane_detection_fused(img, speed_kph, pid, previous_steering, base_throttle, max_steering_change, step_i):

    # Use static variables to store last SCNN result/metrics/confidence
    if not hasattr(lane_detection_fused, "scnn_cache"):
        lane_detection_fused.scnn_cache = {
            'result': None,
            'metrics': None,
            'conf': 0.0,
            'last_frame': -5
        }

    cv_result, cv_metrics, cv_conf = lane_detection_cv_process_frame(
        img, speed=speed_kph, previous_steering=previous_steering, 
        debug_display=False, perspective_debug_display=False,
        calibration_data=CAMERA_CALIBRATION
    )

    # Run SCNN every 5 frames (same as UNet was doing)
    if lane_detection_fused.scnn_cache['last_frame'] is None or step_i - lane_detection_fused.scnn_cache['last_frame'] >= 5:
        scnn_result, scnn_metrics, scnn_conf = lane_detection_scnn_process_frame(
            img, model=MODELS['lane_scnn'], device=MODELS['scnn_device'], 
            speed=speed_kph, previous_steering=previous_steering, 
            debug_display=False,
            calibration_data=CAMERA_CALIBRATION
        )
        lane_detection_fused.scnn_cache = {
            'result': scnn_result,
            'metrics': scnn_metrics,
            'conf': scnn_conf,
            'last_frame': step_i
        }
    else:
        scnn_result = lane_detection_fused.scnn_cache['result']
        scnn_metrics = lane_detection_fused.scnn_cache['metrics']
        scnn_conf = lane_detection_fused.scnn_cache['conf']
    
    print(f"CV Conf: {cv_conf:.3f}, SCNN Conf: {scnn_conf:.3f}")

    fused_metrics = fuse_lane_metrics(cv_metrics, cv_conf, scnn_metrics, scnn_conf, method_name="SCNN")

    cv2.imshow('Lane Detection CV', cv_result)
    cv2.imshow('Lane Detection SCNN', scnn_result)


    deviation = fused_metrics.get('deviation', 0.0)
    smoothed_deviation = fused_metrics.get('smoothed_deviation', 0.0)
    effective_deviation = fused_metrics.get('effective_deviation', 0.0)
    lane_center = fused_metrics.get('lane_center', 0.0)
    vehicle_center = fused_metrics.get('vehicle_center', 0.0)

    steering = pid.update(-effective_deviation, 0.01)
    steering = np.clip(steering, -1.0, 1.0)
    steering_change = steering - previous_steering
    if abs(steering_change) > max_steering_change:
        steering = previous_steering + np.sign(steering_change) * max_steering_change

    throttle = base_throttle * (1.0 - 0.3 * abs(steering))
    throttle = np.clip(throttle, 0.05, 0.3)

    result = cv_result if cv_conf > scnn_conf else scnn_result

    return result, steering, throttle, smoothed_deviation, lane_center, vehicle_center, fused_metrics

def sign_detection_classification(img):
    """
    Process sign detection and classification on the input image.
    """
    sign_detections, sign_img = sign_process_frame(img, draw_detections=True)
    return sign_detections, sign_img

def vehicle_obstacle_detection(img):
    """
    Process vehicle and pedestrian detection on the input image.
    """
    vehicle_obstacle_detections, vehicle_img = vehicle_obstacle_process_frame(img, draw_detections=True)
    return vehicle_obstacle_detections, vehicle_img

# def lidar_object_detections(lidar_data, camera_detections=vehicle_obstacle_detection):
#     """
#     Process LiDAR data for object detection.
#     """
#     lidar_detections, lidar_obj_img = lidar_process_object_frame(lidar_data)
#     return lidar_detections, lidar_obj_img


def cruise_control(target_speed_kph, current_speed_kph, speed_pid, dt):
    """
    Simple cruise control to maintain target speed using PID controller.
    Args:
        target_speed_kph (float): Desired speed in kph
        current_speed_kph (float): Current speed in kph
        speed_pid (PIDController): PID controller instance for speed
        dt (float): Time delta in seconds
    Returns:
        float: Throttle value between 0.0 and 1.0
    """
    speed_error = target_speed_kph - current_speed_kph
    throttle = speed_pid.update(speed_error, dt)
    throttle = np.clip(throttle, 0.0, 1.0)
    return throttle

def main():
    """
    Main function to run the simulation.
    """

    try:
        load_models()
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    beamng, scenario, vehicle, camera, lidar, radar = sim_setup()
    print("Simulation setup complete")

    print("Wait for sensors to initialize")
    time.sleep(3)
    
    try:
        print("Testing camera...")
        camera_test = camera.poll()
        print(f"Camera working: {type(camera_test)}")
    except Exception as e:
        print(f"Camera error: {e}")
        
    try:
        print("Testing lidar...")
        lidar_test = lidar.poll()
        print(f"LiDAR working: {type(lidar_test)}")
    except Exception as e:
        print(f"LiDAR error: {e}")

    try:
        print("Testing radar...")
        radar_test = radar.poll()
        print(f"Radar working: {type(radar_test)}")
    except Exception as e:
        print(f"Radar error: {e}")

    steering_pid = PIDController(Kp=0.017, Ki=0.0, Kd=0.004, derivative_filter_alpha=0.2)
    max_steering_change = 0.22
    previous_steering = 0.0


    base_throttle = 0.12
    target_speed_kph = 40
    speed_pid = PIDController(Kp=0.02, Ki=0.0, Kd=0.01)

    frame_count = 0

    log_dir = "./beamng_sim/drive_log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"drive_log_{timestamp}.csv"
    log_path = os.path.join(log_dir, log_filename)
    log_fields = ["frame", "deviation_m", "lane_center", "vehicle_center", "steering", "throttle", "speed_kph"]
    log_file = open(log_path, mode="w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    last_time = time.time()
    try:
        step_i = 0
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            try:
                beamng.control.step(10)
            except Exception as e:
                print(f"Simulation step error: {e}")

            images = camera.stream()
            img = np.array(images['colour'])

            # Speed
            try:
                speed_mps, speed_kph, car_pos, direction = get_vehicle_speed(vehicle)
                speed_mps = abs(speed_mps)
                speed_kph = abs(speed_kph)
            except Exception as e:
                print(f"Speed retrieval error: {e}")
                break

            # Lane Detection
            result, steering, throttle, deviation, lane_center, vehicle_center, fused_metrics = lane_detection_fused(
                img, speed_kph, steering_pid, previous_steering, base_throttle, max_steering_change, step_i=step_i
            )

            # Log to CSV
            fused_confidence = fused_metrics.get('confidence', 0.0)
            log_writer.writerow({
                "frame": step_i,
                "deviation_m": round(deviation, 3) if deviation is not None else 0.0,
                "lane_center": round(lane_center, 3) if lane_center is not None else 0.0,
                "vehicle_center": round(vehicle_center, 3) if vehicle_center is not None else 0.0,
                "steering": round(steering, 3),
                "throttle": round(throttle, 3),
                "speed_kph": round(speed_kph, 3)
            })


            if step_i % 80 == 0: # Lower later
                # Sign Detection
                sign_detections, sign_img = sign_detection_classification(img)
                cv2.imshow('Sign Detection', sign_img)


                # Vehicle & Obstacle Detection
                vehicle_detections, vehicle_img = vehicle_obstacle_detection(img)
                cv2.imshow('Vehicle and Pedestrian Detection', vehicle_img)

            # radar_detections = radar_process_frame(radar_sensor=radar, camera_detections=vehicle_detections, speed=speed_kph)

            # Lidar Road Boundaries
            try:
                lidar_lane_boundaries, filtered_points = lidar_process_frame(lidar, beamng=beamng, speed=speed_kph, debug_window=None, vehicle=vehicle, car_position=car_pos, car_direction=direction)
            except Exception as lidar_e:
                print(f"Lidar process error: {lidar_e}")
                lidar_lane_boundaries = None
                filtered_points = None


            # Lidar Object Detection
            # lidar_detections, lidar_obj_img = lidar_object_detections(lidar, camera_detections=vehicle_detections)

            throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt)
            
            # Steering, throttle, brake inputs
            previous_steering = steering
            vehicle.control(steering=steering, throttle=throttle, brake=0.0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            step_i += 1

            try:
                bridge.send_lane_detection(
                    lane_center=lane_center,
                    vehicle_center=vehicle_center,
                    deviation=deviation,
                    confidence=fused_confidence,
                    left_lane_points=lidar_lane_boundaries.get('left_lane_points') if lidar_lane_boundaries else None,
                    right_lane_points=lidar_lane_boundaries.get('right_lane_points') if lidar_lane_boundaries else None
                )
                print("Lane detection sent to Foxglove")
            except Exception as lane_det_send_e:
                print(f"Error sending lane detection to Foxglove: {lane_det_send_e}")

            try:
                bridge.send_vehicle_state(
                    speed_kph=speed_kph,
                    steering=steering,
                    throttle=throttle,
                    x=car_pos[0],
                    y=car_pos[1],
                    z=car_pos[2]
                )
                print("Vehicle state sent to Foxglove")
            except Exception as vehicle_state_send_e:
                print(f"Error sending vehicle state to Foxglove: {vehicle_state_send_e}")

            try:
                if filtered_points is not None and len(filtered_points) > 0:
                    bridge.send_lidar(filtered_points)
                    print("LiDAR data sent to Foxglove")
            except Exception as lidar_send_e:
                print(f"Error sending LiDAR to Foxglove: {lidar_send_e}")

            if step_i % 80 == 0 and vehicle_detections:
                for detection in vehicle_detections:
                    bbox = detection['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    bridge.send_vehicle_detection(
                        detection_type=detection['class'],
                        x=x_center,
                        y=y_center,
                        width=width,
                        height=height,
                        confidence=detection['confidence']
                    )
                for sign_det in sign_detections:
                    bbox = sign_det['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    
                    bridge.send_sign_detection(
                        sign_type=sign_det.get('classification', 'Unknown'),
                        x=x_center,
                        y=y_center,
                        confidence=sign_det.get('classification_confidence', 0.0)
                    )

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        beamng.close()

if __name__ == "__main__":
    main()