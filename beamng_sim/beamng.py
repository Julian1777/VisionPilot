import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beamng_sim.utils.pid_controller import PIDController

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Lidar, Radar

from beamng_sim.sign.detect_classify import random_brightness
from config.config import SIGN_DETECTION_MODEL, SIGN_CLASSIFICATION_MODEL, VEHICLE_PEDESTRIAN_MODEL, UNET_LANE_DETECTION_MODEL

from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import time
import math
import cv2
import csv
import datetime

from beamng_sim.lane_detection.main import process_frame_cv as lane_detection_cv_process_frame
from beamng_sim.lane_detection.main import process_frame_unet as lane_detection_unet_process_frame
from beamng_sim.sign.main import process_frame as sign_process_frame
from beamng_sim.vehicle_obstacle.main import process_frame as vehicle_obstacle_process_frame
from beamng_sim.lidar.main import process_frame as lidar_process_frame
from beamng_sim.radar.main import process_frame as radar_process_frame

from beamng_sim.lane_detection.fusion import fuse_lane_metrics

from beamng_sim.lidar.lidar_lane_debug import LiveLidarDebugWindow

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
    
    print("All models loaded!")


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
            is_360_mode=False,
            horizontal_angle=120,  # Horizontal field of view
            vertical_angle=30,  # Vertical field of view
            vertical_resolution=50,  # Number of lasers/channels
            density=10,
            frequency=20,
            max_distance=100,
            pos=(0, -0.35, 1.425),
            is_visualised=False,
        )
        print("LiDAR initialized")
    except Exception as e:
        print(f"LiDAR initialization error: {e}")
        lidar = None
    
    # try:
    #     radar = Radar(
    #         "radar1",
    #         beamng,
    #         vehicle,
    #         requested_update_time=0.01,
    #         pos=(0, -2.5, 0.5),
    #         dir=(0, -1, 0),
    #         range_min=2,
    #         range_max=120,
    #         vel_min=-40,
    #         vel_max=40,
    #         field_of_view_y=70,
    #     )
    #     print("Radar initialized")
    # except Exception as e:
    #     print(f"Radar initialization error: {e}")
    #     radar = None

    return beamng, scenario, vehicle, camera, lidar, # radar

def get_vehicle_speed(vehicle):
    """
    Get the vehicle speed in m/s and kph.
    Args:
        vehicle (Vehicle): BeamNG vehicle object
    Returns:
        tuple: (speed_mps, speed_kph)
    """

    vehicle.poll_sensors()
    if 'vel' in vehicle.state:
        speed_mps = vehicle.state['vel'][0]
        speed_kph = speed_mps * 3.6
    else:
        speed_mps = 0.0
        speed_kph = 0.0

    return speed_mps, speed_kph


def lane_detection_fused(img, speed_kph, pid, previous_steering, base_throttle, steering_bias, max_steering_change, step_i):

    # Use static variables to store last UNet result/metrics/confidence
    if not hasattr(lane_detection_fused, "unet_cache"):
        lane_detection_fused.unet_cache = {
            'result': None,
            'metrics': None,
            'conf': 0.0,
            'last_frame': -5
        }

    cv_result, cv_metrics, cv_conf = lane_detection_cv_process_frame(img, speed=speed_kph, previous_steering=previous_steering, debug_display=False, perspective_debug_display=True)

    if lane_detection_fused.unet_cache['last_frame'] is None or step_i - lane_detection_fused.unet_cache['last_frame'] >= 5:
        unet_result, unet_metrics, unet_conf = lane_detection_unet_process_frame(img, model=MODELS['lane_unet'], speed=speed_kph, previous_steering=previous_steering, debug_display=True)
        lane_detection_fused.unet_cache = {
            'result': unet_result,
            'metrics': unet_metrics,
            'conf': unet_conf,
            'last_frame': step_i
        }
    else:
        unet_result = lane_detection_fused.unet_cache['result']
        unet_metrics = lane_detection_fused.unet_cache['metrics']
        unet_conf = lane_detection_fused.unet_cache['conf']
    
    print(f"CV Conf: {cv_conf:.3f}, UNet Conf: {unet_conf:.3f}")

    fused_metrics = fuse_lane_metrics(cv_metrics, cv_conf, unet_metrics, unet_conf)

    fused_result = cv_result if cv_conf > unet_conf else unet_result # CHANGE

    cv2.imshow('Lane Detection CV', cv_result)
    cv2.imshow('Lane Detection UNet', unet_result)
    #cv2.imshow('Lane Detection Fused', fused_result)

    deviation = fused_metrics.get('deviation', 0.0)
    smoothed_deviation = fused_metrics.get('smoothed_deviation', 0.0)
    effective_deviation = fused_metrics.get('effective_deviation', 0.0)
    lane_center = fused_metrics.get('lane_center', 0.0)
    vehicle_center = fused_metrics.get('vehicle_center', 0.0)

    steering = pid.update(-effective_deviation, 0.01)
    steering += steering_bias
    steering = np.clip(steering, -1.0, 1.0)
    steering_change = steering - previous_steering
    if abs(steering_change) > max_steering_change:
        steering = previous_steering + np.sign(steering_change) * max_steering_change

    throttle = base_throttle * (1.0 - 0.3 * abs(steering))
    throttle = np.clip(throttle, 0.05, 0.3)

    result = cv_result if cv_conf > unet_conf else unet_result # CHANGE

    return result, steering, throttle, smoothed_deviation, lane_center, vehicle_center

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

def main():
    """
    Main function to run the simulation.
    """
    try:
        load_models()
    except Exception as e:
        print(f"Model loading error: {e}")
        return
    
    beamng, scenario, vehicle, camera, lidar = sim_setup()
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

    # try:
    #     print("Testing radar...")
    #     radar_test = radar.poll()
    #     print(f"Radar working: {type(radar_test)}")
    # except Exception as e:
    #     print(f"Radar error: {e}")

    debug_window = LiveLidarDebugWindow()

    pid = PIDController(Kp=0.14, Ki=0.0, Kd=0.17)

    base_throttle = 0.02

    steering_bias = 0
    max_steering_change = 0.15
    previous_steering = 0.0

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
                speed_mps, speed_kph = get_vehicle_speed(vehicle)
                speed_mps = abs(speed_mps)
                speed_kph = abs(speed_kph)
            except Exception as e:
                print(f"Speed retrieval error: {e}")
                break

            # Lane Detection
            result, steering, throttle, deviation, lane_center, vehicle_center = lane_detection_fused(
                img, speed_kph, pid, previous_steering, base_throttle, steering_bias, max_steering_change, step_i=step_i
            )

            # Log to CSV
            log_writer.writerow({
                "frame": step_i,
                "deviation_m": round(deviation, 3) if deviation is not None else 0.0,
                "lane_center": round(lane_center, 3) if lane_center is not None else 0.0,
                "vehicle_center": round(vehicle_center, 3) if vehicle_center is not None else 0.0,
                "steering": round(steering, 3),
                "throttle": round(throttle, 3),
                "speed_kph": round(speed_kph, 3)
            })


            if step_i % 20 == 0:
                # Sign Detection
                sign_detections, sign_img = sign_detection_classification(img)
                cv2.imshow('Sign Detection', sign_img)

                # Vehicle & Obstacle Detection
                vehicle_detections, vehicle_img = vehicle_obstacle_detection(img)
                cv2.imshow('Vehicle and Pedestrian Detection', vehicle_img)

            # radar_detections = radar_process_frame(radar_sensor=radar, camera_detections=vehicle_detections, speed=speed_kph)

            # Lidar Road Boundaries
            lidar_lane_boundaries = lidar_process_frame(lidar, camera_detections=vehicle_detections, beamng=beamng, speed=speed_kph, debug_window=debug_window)

            # Lidar Object Detection
            # lidar_detections, lidar_obj_img = lidar_object_detections(lidar, camera_detections=vehicle_detections)

            # Steering, throttle, brake inputs
            previous_steering = steering
            vehicle.control(steering=steering, throttle=throttle, brake=0.0)

            if step_i % 5 == 0:
                print(f"[{step_i}] Deviation: {deviation:.3f}m | Steering: {steering:.3f} | Throttle: {throttle:.3f}")
                print(f"[{step_i}] lane_center={lane_center}, vehicle_center={vehicle_center}, speed={speed_kph:.2f} kph")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            step_i += 1

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        beamng.close()
        debug_window.close()

if __name__ == "__main__":
    main()