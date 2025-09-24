import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beamng_sim.utils.pid_controller import PIDController

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Lidar, Radar

from ultralytics import YOLO
import tensorflow as tf

from beamng_sim.sign.detect_classify import random_brightness
from config.config import SIGN_DETECTION_MODEL, SIGN_CLASSIFICATION_MODEL


import numpy as np
import time
import math
import cv2


from beamng_sim.lane_detection.main import process_frame as lane_detection_process_frame
from beamng_sim.sign.main import process_frame as sign_process_frame
from beamng_sim.vehicle_obstacle.main import process_frame as vehicle_obstacle_process_frame
from beamng_sim.lidar.main import process_frame as lidar_process_frame
from beamng_sim.radar.main import process_frame as radar_process_frame

from beamng_sim.lidar.lidar_lane_debug import LiveLidarDebugWindow


MODELS = {}

def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)

def load_models():    
    print("Loading models")
    
    # Load sign detection model
    MODELS['sign_detect'] = YOLO(str(SIGN_DETECTION_MODEL))
    print("Sign detection model loaded")
    
    # Load sign classification model with custom objects used during training
    MODELS['sign_classify'] = tf.keras.models.load_model(
        str(SIGN_CLASSIFICATION_MODEL), 
        custom_objects={"random_brightness": random_brightness}
    )
    print("Sign classification model loaded")
    
    print("All models loaded!")


def sim_setup():
    beamng = BeamNGpy('localhost', 64256, home=r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0')
    beamng.open()

    scenario = Scenario('west_coast_usa', 'lane_detection_city')
    #scenario = Scenario('west_coast_usa', 'lane_detection_highway')

    vehicle = Vehicle('ego_vehicle', model='etk800', licence='JULIAN')
    #vehicle = Vehicle('Q8', model='rsq8_600_tfsi', licence='JULIAN')

    # Spawn positions rotation conversion
    rot_city = yaw_to_quat(-133.506 + 180)
    rot_highway = yaw_to_quat(-135.678)

    # Street Spawn
    scenario.add_vehicle(vehicle, pos=(-730.212, 94.630, 118.517), rot_quat=rot_city)
    
    # Highway Spawn
    #scenario.add_vehicle(vehicle, pos=(-287.210, 73.609, 112.363), rot_quat=rot_highway)

    scenario.make(beamng)

    beamng.settings.set_deterministic(60)

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

    lidar = Lidar(
        "lidar1",
        beamng,
        vehicle,
        requested_update_time=0.01,
        is_using_shared_memory=True,
        is_360_mode=False,  # Might set to true for mapping later
        vertical_angle=30,  # Vertical field of view
        vertical_resolution=64,  # Number of lasers/channels
        density=10,
        frequency=20,
        max_distance=100,
        pos=(0, 0.2, 1.8),
    )

    radar = Radar(
        "radar1",
        beamng,
        vehicle,
        requested_update_time=0.1,
        pos=(0, -2.5, 0.5),
        dir=(0, -1, 0),
        range_min=2,
        range_max=120,
        vel_min=-40,
        vel_max=40,
        half_angle_deg=15,
        field_of_view_y=70,
    )

    return beamng, scenario,vehicle, camera, lidar, radar

def get_vehicle_speed(vehicle):

    vehicle.poll_sensors()
    if 'vel' in vehicle.state:
        speed_mps = vehicle.state['vel'][0]
        speed_kph = speed_mps * 3.6
    else:
        speed_mps = 0.0
        speed_kph = 0.0

    return speed_mps, speed_kph

def lane_detection(img, speed_kph, pid, previous_steering, base_throttle, steering_bias, max_steering_change):
    result, metrics = lane_detection_process_frame(img, speed=speed_kph, debug_display=True)

    deviation = metrics.get('deviation', 0.0)
    lane_center = metrics.get('lane_center', 0.0)
    vehicle_center = metrics.get('vehicle_center', 0.0)

    if deviation is None or lane_center is None or vehicle_center is None:
        deviation, lane_center, vehicle_center = 0.0, 0.0, 0.0
    
    if abs(deviation) > 2.5:
        print(f"Large deviation detected: {deviation:.2f}m - attempting correction")
        deviation = np.clip(deviation, -2.5, 2.5)

    steering = pid.update(-deviation, 0.01)
    steering += steering_bias
    steering = np.clip(steering, -1.0, 1.0)
    steering_change = steering - previous_steering

    if abs(steering_change) > max_steering_change:
        steering = previous_steering + np.sign(steering_change) * max_steering_change

    throttle = base_throttle * (1.0 - 0.3 * abs(steering))
    throttle = np.clip(throttle, 0.05, 0.3)

    return result, steering, throttle, deviation, lane_center, vehicle_center

def sign_detection_classification(img):
    sign_detections, sign_img = sign_process_frame(img, draw_detections=True)
    return sign_detections, sign_img

def vehicle_obstacle_detection(img):
    vehicle_obstacle_detections, vehicle_img = vehicle_obstacle_process_frame(img, draw_detections=True)
    return vehicle_obstacle_detections, vehicle_img

# def lidar_object_detections(lidar_data, camera_detections=vehicle_obstacle_detection):
#     lidar_detections, lidar_obj_img = lidar_process_object_frame(lidar_data)
#     return lidar_detections, lidar_obj_img

def main():
    load_models()
    
    beamng, scenario, vehicle, camera, lidar, radar = sim_setup()

    debug_window = LiveLidarDebugWindow()

    pid = PIDController(Kp=0.15, Ki=0.005, Kd=0.04)

    base_throttle = 0.05
    steering_bias = 0
    max_steering_change = 0.1
    previous_steering = 0.0

    frame_count = 0

    last_time = time.time()
    try:
        step_i = 0
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            beamng.control.step(10)
            images = camera.stream()
            img = np.array(images['colour'])

            # Speed
            speed_mps, speed_kph = get_vehicle_speed(vehicle)

            # Lane Detection
            result, steering, throttle, deviation, lane_center, vehicle_center = lane_detection(
                img, speed_kph, pid, previous_steering, base_throttle, steering_bias, max_steering_change
            )
            cv2.imshow('Lane Detection', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

            # Sign Detection
            sign_detections, sign_img = sign_detection_classification(img)
            cv2.imshow('Sign Detection', sign_img)

            # Vehicle & Obstacle Detection
            vehicle_detections, vehicle_img = vehicle_obstacle_detection(img)
            cv2.imshow('Vehicle and Pedestrian Detection', vehicle_img)

            radar_detections = radar_process_frame(radar_sensor=radar, camera_detections=vehicle_detections, speed=speed_kph)

            # Lidar Road Boundaries
            lidar_boundaries = lidar_process_frame(lidar, camera_detections=vehicle_detections, beamng=beamng, speed=speed_kph, debug_window=None)

            # Lidar Object Detection
            # lidar_detections, lidar_obj_img = lidar_object_detections(lidar, camera_detections=vehicle_detections)

            # Steering, throttle, brake inputs
            previous_steering = steering
            vehicle.control(steering=steering, throttle=throttle, brake=0.0)

            if step_i % 20 == 0:
                print(f"[{step_i}] Deviation: {deviation:.3f}m | Steering: {steering:.3f} | Throttle: {throttle:.3f}")
                print(f"Frame {step_i}: lane_center={lane_center}, vehicle_center={vehicle_center}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            step_i += 1

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        beamng.close()
        debug_window.close()

if __name__ == "__main__":
    main()