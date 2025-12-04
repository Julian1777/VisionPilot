import sys
import os
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beamng_sim.utils.pid_controller import PIDController

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Lidar, Radar, GPS, AdvancedIMU
from foxglove.schemas import Color

from beamng_sim.sign.detect_classify import random_brightness

from config.config import SIGN_DETECTION_MODEL, SIGN_CLASSIFICATION_MODEL, VEHICLE_PEDESTRIAN_MODEL, SCNN_LANE_DETECTION_MODEL, LIGHT_DETECTION_CLASSIFICATION_MODEL

from ultralytics import YOLO
import tensorflow as tf
import torch
import numpy as np
import time
import math
import cv2
from scipy.spatial.transform import Rotation as R


from beamng_sim.lane_detection.main import process_frame_cv as lane_detection_cv_process_frame
from beamng_sim.lane_detection.main import process_frame_scnn as lane_detection_scnn_process_frame
from beamng_sim.sign.main import process_frame as sign_process_frame
from beamng_sim.traffic_light.detect_classify import detect_traffic_lights
from beamng_sim.vehicle_obstacle.main import process_frame as vehicle_obstacle_process_frame
from beamng_sim.lidar.main import process_frame as lidar_process_frame
from beamng_sim.radar.main import process_frame as radar_process_frame

from beamng_sim.lane_detection.fusion import fuse_lane_metrics

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

def yaw_rad_to_quaternion(yaw_rad):
    """
    Convert yaw angle in radians to a quaternion representation for vehicle orientation.
    Args:
        yaw_rad (float): Yaw angle in radians
    Returns:
        tuple: Quaternion (x, y, z, w)
    """
    w = math.cos(yaw_rad / 2)
    z = math.sin(yaw_rad / 2)
    return (0.0, 0.0, z, w)

def get_timestamp_ns():
    """
    Get current timestamp in nanoseconds since epoch.
    Returns:
        int: Timestamp in nanoseconds
    """
    return int(time.time_ns())

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

    # Traffic Light detection model
    MODELS['traffic_light'] = YOLO(str(LIGHT_DETECTION_CLASSIFICATION_MODEL))
    print("Traffic Light detection model loaded")
    
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
    print("All models loaded successfully!")


def load_config():
    """Load all configuration files."""
    config_path = os.path.join(os.path.dirname(__file__), 'config')
    
    with open(os.path.join(config_path, 'beamng_sim.yaml'), 'r') as f:
        beamng_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'scenarios.yaml'), 'r') as f:
        scenarios_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'sensors.yaml'), 'r') as f:
        sensors_config = yaml.safe_load(f)
        
    return beamng_config, scenarios_config, sensors_config


def sim_setup(scenario_name='highway', vehicle_name=None):
    """
    Setup BeamNG simulation, scenario, vehicle, spawn point and sensors.
    Args:
        scenario_name (str): Name of the scenario to load ('highway' or 'city')
        vehicle_name (str): Name of the vehicle config to use (e.g., 'etk800_highway', 'q8_default_highway'). 
                           If None, uses the vehicle from the scenario config.
    """
    # Load configurations
    beamng_config, scenarios_config, sensors_config = load_config()
    
    # Setup BeamNG
    sim_cfg = beamng_config['simulation']
    beamng = BeamNGpy(sim_cfg['host'], sim_cfg['port'], home=sim_cfg['home'])
    beamng.open()

    # Setup scenario
    if scenario_name not in scenarios_config:
        raise ValueError(f"Scenario '{scenario_name}' not found in config")
    
    scenario_cfg = scenarios_config[scenario_name]
    scenario = Scenario(scenario_cfg['map'], scenario_cfg['scene'])

    if vehicle_name is None:
        vehicle_name = scenario_cfg['vehicle']
    
    if vehicle_name not in beamng_config['vehicles']:
        raise ValueError(f"Vehicle '{vehicle_name}' not found in config")
    
    vehicle_cfg = beamng_config['vehicles'][vehicle_name]
    
    vehicle = Vehicle(
        vehicle_cfg['name'],
        model=vehicle_cfg['model'],
        licence=vehicle_cfg['license'],
        part_config=vehicle_cfg['part_config']
    )

    # Spawn vehicle
    rot = yaw_to_quat(vehicle_cfg['spawn_yaw'])
    scenario.add_vehicle(vehicle, pos=tuple(vehicle_cfg['spawn_pos']), rot_quat=rot)

    scenario.make(beamng)
    beamng.settings.set_deterministic(60)
    beamng.scenario.load(scenario)
    beamng.scenario.start()

    # Setup sensors - select config based on vehicle model
    vehicle_model = vehicle_cfg['model']
    if vehicle_model not in sensors_config:
        raise ValueError(f"Sensor configuration for vehicle model '{vehicle_model}' not found in config")
    
    sensors = sensors_config[vehicle_model]
    cameras = {}
    lidar = None
    radars = {}
    gps_sensors = {}
    imus = {}

    # Initialize cameras - support multiple cameras (camera_front, camera_left, camera_right, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('camera_') and sensor_cfg.get('enabled', False):
            try:
                camera = Camera(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    requested_update_time=sensor_cfg['requested_update_time'],
                    is_using_shared_memory=sensor_cfg.get('is_using_shared_memory', False),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg['dir']),
                    field_of_view_y=sensor_cfg['field_of_view_y'],
                    near_far_planes=tuple(sensor_cfg['near_far_planes']),
                    resolution=tuple(sensor_cfg['resolution']),
                    is_streaming=sensor_cfg.get('is_streaming', False),
                    is_render_colours=sensor_cfg.get('is_render_colours', True),
                )
                cameras[sensor_key] = camera
                print(f"Camera '{sensor_key}' initialized")
            except Exception as e:
                print(f"Camera '{sensor_key}' initialization error: {e}")
                cameras[sensor_key] = None

    # Initialize LiDAR - support multiple LiDARs (lidar_top, lidar_rear, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('lidar_') and sensor_cfg.get('enabled', False):
            try:
                lidar = Lidar(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    requested_update_time=sensor_cfg['requested_update_time'],
                    is_using_shared_memory=sensor_cfg.get('is_using_shared_memory', False),
                    is_rotate_mode=sensor_cfg.get('is_rotate_mode', False),
                    horizontal_angle=sensor_cfg.get('horizontal_angle', 360),
                    vertical_angle=sensor_cfg.get('vertical_angle', 26.9),
                    vertical_resolution=sensor_cfg.get('vertical_resolution', 64),
                    density=sensor_cfg.get('density', 1),
                    frequency=sensor_cfg.get('frequency', 20),
                    max_distance=sensor_cfg.get('max_distance', 120),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg.get('dir', [0, -1, 0])),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                )
                print(f"LiDAR '{sensor_key}' initialized")
                break  # Use first enabled LiDAR as primary
            except Exception as e:
                print(f"LiDAR '{sensor_key}' initialization error: {e}")

    # Initialize Radars - support multiple radars (radar_front, radar_rear_left, radar_rear_right, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('radar_') and sensor_cfg.get('enabled', False):
            try:
                print(f"Attempting {sensor_key} initialization...")
                radar = Radar(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    requested_update_time=sensor_cfg.get('requested_update_time', 0.05),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg.get('dir', [0, -1, 0])),
                    up=tuple(sensor_cfg.get('up', [0, 0, 1])),
                    size=tuple(sensor_cfg.get('size', [200, 200])),
                    near_far_planes=tuple(sensor_cfg.get('near_far_planes', [0.1, 200])),
                    field_of_view_y=sensor_cfg.get('field_of_view_y', 18),
                    range_min=sensor_cfg.get('range_min', 0.5),
                    range_max=sensor_cfg.get('range_max', 150.0),
                    vel_min=sensor_cfg.get('vel_min', -40),
                    vel_max=sensor_cfg.get('vel_max', 40),
                    range_bins=sensor_cfg.get('range_bins', 128),
                    azimuth_bins=sensor_cfg.get('azimuth_bins', 64),
                    vel_bins=sensor_cfg.get('vel_bins', 32),
                    half_angle_deg=sensor_cfg.get('half_angle_deg', 9),
                )
                radars[sensor_key] = radar
                print(f"{sensor_key.replace('_', ' ').title()} initialized")
            except Exception as e:
                print(f"{sensor_key} initialization error: {e}")
                radars[sensor_key] = None

    # Initialize GPS sensors - support multiple GPS (gps_front, gps_rear, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('gps_') and sensor_cfg.get('enabled', False):
            try:
                print(f"Attempting {sensor_key} initialization...")
                gps = GPS(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    gfx_update_time=sensor_cfg.get('gfx_update_time', 0.0),
                    physics_update_time=sensor_cfg.get('physics_update_time', 0.05),
                    pos=tuple(sensor_cfg['pos']),
                    ref_lon=sensor_cfg.get('ref_lon', 13.1856),
                    ref_lat=sensor_cfg.get('ref_lat', 51.5074),
                    is_send_immediately=sensor_cfg.get('is_send_immediately', False),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                    is_snapping_desired=sensor_cfg.get('is_snapping_desired', False),
                    is_force_inside_triangle=sensor_cfg.get('is_force_inside_triangle', False),
                    is_dir_world_space=sensor_cfg.get('is_dir_world_space', False),
                )
                gps_sensors[sensor_key] = gps
                print(f"{sensor_key.replace('_', ' ').title()} initialized")
            except Exception as e:
                print(f"{sensor_key} initialization error: {e}")
                gps_sensors[sensor_key] = None

    # Initialize IMU sensors - support multiple IMUs (imu_1, imu_2, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('imu_') and sensor_cfg.get('enabled', False):
            try:
                print(f"Attempting {sensor_key} initialization...")
                imu = AdvancedIMU(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    gfx_update_time=sensor_cfg.get('gfx_update_time', 0.0),
                    physics_update_time=sensor_cfg.get('physics_update_time', 0.01),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg.get('dir', [0, -1, 0])),
                    up=tuple(sensor_cfg.get('up', [0, 0, 1])),
                    smoother_strength=sensor_cfg.get('smoother_strength', 1.0),
                    is_send_immediately=sensor_cfg.get('is_send_immediately', False),
                    is_using_gravity=sensor_cfg.get('is_using_gravity', False),
                    is_allow_wheel_nodes=sensor_cfg.get('is_allow_wheel_nodes', False),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                    is_snapping_desired=sensor_cfg.get('is_snapping_desired', False),
                    is_force_inside_triangle=sensor_cfg.get('is_force_inside_triangle', False),
                    is_dir_world_space=sensor_cfg.get('is_dir_world_space', False),
                )
                imus[sensor_key] = imu
                print(f"{sensor_key.replace('_', ' ').title()} initialized")
            except Exception as e:
                print(f"{sensor_key} initialization error: {e}")
                imus[sensor_key] = None

    # Return primary camera (camera_front) and primary GPS for backwards compatibility
    primary_camera = cameras.get('camera_front', next(iter(cameras.values())) if cameras else None)
    primary_gps = gps_sensors.get('gps_front', next(iter(gps_sensors.values())) if gps_sensors else None)
    primary_imu = imus.get('imu_1', next(iter(imus.values())) if imus else None)

    return beamng, scenario, vehicle, primary_camera, lidar, radars, primary_gps, primary_imu, vehicle_model

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
    else:
        print("Vehicle position not available")
        position = None

    if 'dir' in vehicle.state:
        direction = vehicle.state['dir']
    else:
        print("Vehicle direction not available")
        direction = None

    return speed_mps, speed_kph, position, direction


def lane_detection_fused(img, speed_kph, previous_steering, step_i, vehicle_model='q8_andronisk'):

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
        calibration_data=None, vehicle_model=vehicle_model
    )

    # Run SCNN every 5 frames (same as UNet was doing)
    if lane_detection_fused.scnn_cache['last_frame'] is None or step_i - lane_detection_fused.scnn_cache['last_frame'] >= 5:
        scnn_result, scnn_metrics, scnn_conf = lane_detection_scnn_process_frame(
            img, model=MODELS['lane_scnn'], device=MODELS['scnn_device'], 
            speed=speed_kph, previous_steering=previous_steering, 
            debug_display=False,
            calibration_data=None, vehicle_model=vehicle_model
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

    cv_result_rgb = cv2.cvtColor(cv_result, cv2.COLOR_BGR2RGB)
    scnn_result_rgb = cv2.cvtColor(scnn_result, cv2.COLOR_BGR2RGB)
    display_scale = 0.5
    cv_small = cv2.resize(cv_result_rgb, (0, 0), fx=display_scale, fy=display_scale)
    scnn_small = cv2.resize(scnn_result_rgb, (0, 0), fx=display_scale, fy=display_scale)
    cv2.imshow('Lane Detection CV', cv_small)
    cv2.imshow('Lane Detection SCNN', scnn_small)

    result = cv_result if cv_conf > scnn_conf else scnn_result

    return result, fused_metrics

def sign_detection_classification(img, draw=True):
    """
    Process sign detection and classification on the input image.
    """
    sign_detections, sign_img = sign_process_frame(img, draw_detections=draw)
    return sign_detections, sign_img

def vehicle_obstacle_detection(img, draw=True):
    """
    Process vehicle and pedestrian detection on the input image.
    """
    vehicle_obstacle_detections, vehicle_img = vehicle_obstacle_process_frame(img, draw_detections=draw)
    return vehicle_obstacle_detections, vehicle_img

def traffic_light_detection(img):
    """
    Process traffic light detection on the input image.
    """
    detections = detect_traffic_lights(img)
    return detections

def draw_combined_detections(img, sign_detections, vehicle_detections, tl_detections):
    result_img = img.copy()
    
    # Draw Signs (Blue)
    for det in sign_detections:
        x1, y1, x2, y2 = det['bbox']
        classification = det.get('classification', 'Sign')
        conf = det.get('classification_confidence', 0.0)
        label = f"{classification} {conf:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw Vehicles (Green)
    for det in vehicle_detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Traffic Lights (Orange)
    for det in tl_detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2) 
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
    return result_img

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

    print("Starting Foxglove WebSocket server...")
    bridge.start_server()
    print("Initializing Foxglove channels...")
    bridge.initialize_channels()
    print("Foxglove ready - connect to ws://localhost:8765")

    try:
        load_models()
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    # Load configurations
    beamng_config, scenarios_config, sensors_config = load_config()
    
    # Load control config
    with open(os.path.join(os.path.dirname(__file__), 'config/control.yaml'), 'r') as f:
        control_config = yaml.safe_load(f)

    # Change scenario name between 'highway' and 'city' here
    # You can also specify a vehicle: 'etk800_highway', 'q8_default_highway', 'etk800_city', 'q8_default_city'
    # If vehicle_name is None, it will use the default vehicle from the scenario config
    beamng, scenario, vehicle, camera, lidar, radars, gps, imu, vehicle_model = sim_setup(scenario_name='highway', vehicle_name='q8_default_highway')
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

    # Test all enabled radars
    for radar_name, radar in radars.items():
        if radar is not None:
            try:
                print(f"Testing {radar_name}...")
                radar_test = radar.poll()
                print(f"{radar_name.replace('_', ' ').title()} working: {type(radar_test)}")
            except Exception as e:
                print(f"{radar_name} error: {e}")

    try:
        print("Testing GPS...")
        gps_test = gps.poll()
        print(f"GPS working: {type(gps_test)}")
    except Exception as e:
        print(f"GPS error: {e}")

    try:
        print("Testing IMU...")
        imu_test = imu.poll()
        print(f"IMU working: {type(imu_test)}")
    except Exception as e:
        print(f"IMU error: {e}")

    print("Setting up traffic")
    try:
        beamng.traffic.spawn(max_amount=3, police_ratio=0.0, extra_amount=0, parked_amount=0)
        print("Traffic spawned: 3 vehicles")
    except Exception as e:
        print(f"Traffic setup error: {e}")

    # Load control parameters from config
    control_cfg = control_config['control']
    steering_pid = PIDController(**control_cfg['steering_pid'])
    max_steering_change = control_cfg['max_steering_change']
    previous_steering = 0.0

    min_gap = control_cfg['min_gap']
    target_speed_kph = control_cfg['target_speed_kph']
    speed_pid = PIDController(
        Kp=control_cfg['speed_pid']['Kp'],
        Ki=control_cfg['speed_pid']['Ki'],
        Kd=control_cfg['speed_pid']['Kd']
    )

    frame_count = 0

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

            # Send camera image to Foxglove
            try:
                timestamp_ns = get_timestamp_ns()
                bridge.send_camera_image(img, timestamp_ns, frame_id="camera")
            except Exception as camera_send_e:
                print(f"Error sending camera image to Foxglove: {camera_send_e}")

            # Speed
            try:
                speed_mps, speed_kph, car_pos, direction = get_vehicle_speed(vehicle)
                speed_mps = abs(speed_mps)
                speed_kph = abs(speed_kph)
            except Exception as e:
                print(f"Speed retrieval error: {e}")
                continue

            # Lane Detection
            try:
                result, fused_metrics = lane_detection_fused(
                    img, speed_kph, previous_steering, step_i=step_i, vehicle_model=vehicle_model
                )
            except Exception as lane_e:
                print(f"Lane detection error: {lane_e}")
                continue

            deviation = fused_metrics.get('deviation', 0.0)
            smoothed_deviation = fused_metrics.get('smoothed_deviation', 0.0)
            effective_deviation = fused_metrics.get('effective_deviation', 0.0)
            lane_center = fused_metrics.get('lane_center', 0.0)
            vehicle_center = fused_metrics.get('vehicle_center', 0.0)

            steering = steering_pid.update(-effective_deviation, dt)
            steering = np.clip(steering, -1.0, 1.0)
            steering_change = steering - previous_steering
            if abs(steering_change) > max_steering_change:
                steering = previous_steering + np.sign(steering_change) * max_steering_change

            throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt)
            throttle = throttle * (1.0 - 0.3 * abs(steering))
            throttle = np.clip(throttle, 0.05, 0.3)

            fused_confidence = fused_metrics.get('confidence', 0.0)
            
            # Calculate vehicle yaw from direction
            car_yaw = np.arctan2(-direction[1], -direction[0])
            
            # LiDAR pose (offset from base_link + vehicle rotation/position)
            lidar_offset = np.array([0.0, -0.35, 1.425])
            car_quat = yaw_rad_to_quaternion(car_yaw)
            rotation = R.from_quat([car_quat[0], car_quat[1], car_quat[2], car_quat[3]])
            lidar_pos_in_map = rotation.apply(lidar_offset) + car_pos
            lidar_yaw = car_yaw  # LiDAR has same yaw as vehicle


            if step_i % 80 == 0: # Lower later
                try:
                    # Sign Detection
                    sign_detections, _ = sign_detection_classification(img, draw=False)
                except Exception as sign_e:
                    print(f"Sign detection error: {sign_e}")
                    sign_detections = []
                
                try:
                    # Vehicle & Obstacle Detection
                    vehicle_detections, _ = vehicle_obstacle_detection(img, draw=False)
                except Exception as vehicle_e:
                    print(f"Vehicle detection error: {vehicle_e}")
                    vehicle_detections = []

                try:
                    # Traffic Light Detection
                    traffic_light_detections = traffic_light_detection(img)
                except Exception as tl_e:
                    print(f"Traffic light detection error: {tl_e}")
                    traffic_light_detections = []
                
                # Combine and display all detections
                try:
                    combined_img = draw_combined_detections(img, sign_detections, vehicle_detections, traffic_light_detections)
                    cv2.imshow('All Detections', combined_img)
                except Exception as draw_e:
                    print(f"Error drawing detections: {draw_e}")

            # radar_detections, radar_fig = radar_process_frame(
            #     radar_sensor=radar, 
            #     speed=speed_kph,
            #     debug_window=True  # Enable PPI visualization
            # )

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
                timestamp_ns = get_timestamp_ns()
                lane_message = {
                    "timestamp": timestamp_ns,
                    "lane_center": float(lane_center) if lane_center is not None else 0.0,
                    "vehicle_center": float(vehicle_center) if vehicle_center is not None else 0.0,
                    "deviation": float(deviation) if deviation is not None else 0.0,
                    "confidence": float(fused_confidence)
                }
                if lidar_lane_boundaries and 'left_lane_points' in lidar_lane_boundaries:
                    lane_message["left_lane_points"] = [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                        for p in lidar_lane_boundaries['left_lane_points']
                    ]
                if lidar_lane_boundaries and 'right_lane_points' in lidar_lane_boundaries:
                    lane_message["right_lane_points"] = [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                        for p in lidar_lane_boundaries['right_lane_points']
                    ]
                bridge.lane_channel.log(lane_message)
                
                # Send lane paths
                try:
                    timestamp_ns = get_timestamp_ns()
                    if lidar_lane_boundaries and 'left_lane_points' in lidar_lane_boundaries:
                        left_points = np.array(lidar_lane_boundaries['left_lane_points'])
                        if len(left_points) > 0:
                            bridge.send_lane_path(
                                left_points,
                                timestamp_ns=timestamp_ns,
                                lane_id="left_lane",
                                color=Color(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
                                thickness=0.2
                            )
                    
                    if lidar_lane_boundaries and 'right_lane_points' in lidar_lane_boundaries:
                        right_points = np.array(lidar_lane_boundaries['right_lane_points'])
                        if len(right_points) > 0:
                            bridge.send_lane_path(
                                right_points,
                                timestamp_ns=timestamp_ns,
                                lane_id="right_lane",
                                color=Color(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
                                thickness=0.2
                            )
                except Exception as lane_visual_e:
                    print(f"Error sending lane visualization: {lane_visual_e}")
            except Exception as lane_det_send_e:
                print(f"Error sending lane detection to Foxglove: {lane_det_send_e}")

            try:
                # Send vehicle control state (steering, throttle, brake)
                timestamp_ns = get_timestamp_ns()
                bridge.send_vehicle_control(
                    timestamp_ns=timestamp_ns,
                    speed_kph=speed_kph,
                    steering=steering,
                    throttle=throttle,
                    brake=0.0
                )
            except Exception as control_send_e:
                print(f"Error sending vehicle control to Foxglove: {control_send_e}")

            try:
                # Send vehicle pose (PosesInFrame)
                car_yaw = np.arctan2(-direction[1], -direction[0])
                quat_x, quat_y, quat_z, quat_w = yaw_rad_to_quaternion(car_yaw)
                timestamp_ns = get_timestamp_ns()
                bridge.send_vehicle_pose(
                    timestamp_ns=timestamp_ns,
                    x=car_pos[0],
                    y=car_pos[1],
                    z=car_pos[2],
                    quat_x=quat_x,
                    quat_y=quat_y,
                    quat_z=quat_z,
                    quat_w=quat_w,
                    frame_id="map"
                )
            except Exception as pose_send_e:
                print(f"Error sending vehicle pose to Foxglove: {pose_send_e}")

            try:
                # Publish complete TF tree (map - base_link - lidar_top)
                car_yaw = np.arctan2(-direction[1], -direction[0])
                quat_x, quat_y, quat_z, quat_w = yaw_rad_to_quaternion(car_yaw)
                timestamp_ns = get_timestamp_ns()
                bridge.send_tf_tree(
                    timestamp_ns=timestamp_ns,
                    x=car_pos[0],
                    y=car_pos[1],
                    z=car_pos[2],
                    quat_x=quat_x,
                    quat_y=quat_y,
                    quat_z=quat_z,
                    quat_w=quat_w
                )
            except Exception as tf_send_e:
                print(f"Error publishing TF tree to Foxglove: {tf_send_e}")

            try:
                car_yaw = np.arctan2(-direction[1], -direction[0])
                quat_x, quat_y, quat_z, quat_w = yaw_rad_to_quaternion(car_yaw)
                timestamp_ns = get_timestamp_ns()
                bridge.send_vehicle_3d(
                    timestamp_ns=timestamp_ns,
                    x=car_pos[0],
                    y=car_pos[1],
                    z=car_pos[2],
                    quat_x=quat_x,
                    quat_y=quat_y,
                    quat_z=quat_z,
                    quat_w=quat_w,
                    frame_id="map"
                )
            except Exception as vehicle_3d_send_e:
                print(f"Error sending vehicle 3D model to Foxglove: {vehicle_3d_send_e}")

            try:
                # Send LiDAR point cloud
                if filtered_points is not None and len(filtered_points) > 0:
                    timestamp_ns = get_timestamp_ns()
                    
                    bridge.send_lidar(
                        filtered_points,
                        timestamp_ns=timestamp_ns,
                        frame_id="map"
                    )
            except Exception as lidar_send_e:
                print(f"Error sending LiDAR to Foxglove: {lidar_send_e}")

            if step_i % 80 == 0:
                timestamp_ns = get_timestamp_ns()
                # Send vehicle detections
                if vehicle_detections:
                    for detection in vehicle_detections:
                        try:
                            bbox = detection['bbox']
                            vehicle_det_message = {
                                "timestamp": timestamp_ns,
                                "type": detection['class'],
                                "x": float((bbox[0] + bbox[2]) / 2),
                                "y": float((bbox[1] + bbox[3]) / 2),
                                "width": float(bbox[2] - bbox[0]),
                                "height": float(bbox[3] - bbox[1]),
                                "confidence": float(detection['confidence'])
                            }
                            bridge.vehicle_channel.log(vehicle_det_message)
                            print(f"Vehicle detection sent: {detection['class']}")
                        except Exception as e:
                            print(f"Error sending vehicle detection: {e}")
                
                # Send sign detections
                if sign_detections:
                    for sign_det in sign_detections:
                        try:
                            bbox = sign_det['bbox']
                            sign_message = {
                                "timestamp": timestamp_ns,
                                "type": sign_det.get('classification', 'Unknown'),
                                "x": float((bbox[0] + bbox[2]) / 2),
                                "y": float((bbox[1] + bbox[3]) / 2),
                                "width": float(bbox[2] - bbox[0]),
                                "height": float(bbox[3] - bbox[1]),
                                "confidence": float(sign_det.get('classification_confidence', 0.0))
                            }
                            bridge.sign_channel.log(sign_message)
                            print(f"Sign detection sent: {sign_det.get('classification', 'Unknown')}")
                        except Exception as e:
                            print(f"Error sending sign detection: {e}")

                # Send traffic light detections
                if traffic_light_detections:
                    for tl_det in traffic_light_detections:
                        try:
                            bbox = tl_det['bbox']
                            tl_message = {
                                "timestamp": timestamp_ns,
                                "type": tl_det.get('class', 'Unknown'),
                                "x": float((bbox[0] + bbox[2]) / 2),
                                "y": float((bbox[1] + bbox[3]) / 2),
                                "width": float(bbox[2] - bbox[0]),
                                "height": float(bbox[3] - bbox[1]),
                                "confidence": float(tl_det.get('confidence', 0.0))
                            }
                            bridge.traffic_light_channel.log(tl_message)
                            print(f"Traffic light detection sent: {tl_det.get('class', 'Unknown')}")
                        except Exception as e:
                            print(f"Error sending traffic light detection: {e}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        beamng.close()

if __name__ == "__main__":
    main()