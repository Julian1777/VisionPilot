import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beamng_sim.utils.pid_controller import PIDController
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import time
import math

from beamng_sim.lane_detection.main import lane_detection_process_frame
from beamng_sim.sign.main import process_frame as sign_process_frame

from beamng_sim.vehicle_obstacle.main import process_frame as vehicle_obstacle_process_frame

def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)


def sim_setup():
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

    return beamng, vehicle, camera

def get_vehicle_speed(vehicle):

    vehicle.poll_sensors()  # Update the state
    if 'vel' in vehicle.state:
        speed_mps = vehicle.state['vel'][0]
        speed_kph = speed_mps * 3.6
    else:
        speed_mps = 0.0
        speed_kph = 0.0

    return speed_kph

def lane_detection(img, speed_kph, pid, previous_steering, base_throttle, steering_bias, max_steering_change):
    result, metrics = lane_detection_process_frame(img, speed=speed_kph, debug_display=True)

    deviation = metrics.get('deviation', 0.0)
    lane_center = metrics.get('lane_center', 0.0)
    vehicle_center = metrics.get('vehicle_center', 0.0)

    if deviation is None or lane_center is None or vehicle_center is None:
        deviation, lane_center, vehicle_center = 0.0, 0.0, 0.0
    if abs(deviation) > 1.0:
        deviation = 0.0

    steering = pid.update(-deviation, 0.01)  # dt can be passed in
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

def main():
    beamng, scenario, vehicle, camera = sim_setup()

    pid = PIDController(Kp=0.15, Ki=0.005, Kd=0.04)

    base_throttle = 0.08
    steering_bias = 0
    max_steering_change = 0.08
    previous_steering = 0.0

    frame_count = 0

    last_time = time.time()
    try:
        for step_i in range(1000):
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
            _, sign_img = sign_detection_classification(img)
            cv2.imshow('Sign Detection', sign_img)

            # Vehicle & Obstacle Detection
            _, vehicle_img = vehicle_obstacle_detection(img)
            cv2.imshow('Vehicle and Pedestrian Detection', vehicle_img)

            # Steering, throttle, brake inputs
            previous_steering = steering
            vehicle.control(steering=steering, throttle=throttle, brake=0.0)

            # Debug every 20 frames
            if step_i % 20 == 0:
                print(f"[{step_i}] Deviation: {deviation:.3f}m | Steering: {steering:.3f} | Throttle: {throttle:.3f}")
                print(f"Frame {step_i}: lane_center={lane_center}, vehicle_center={vehicle_center}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vehicle.control(throttle=0, steering=0, brake=1.0)
        cv2.destroyAllWindows()
        beamng.close()

if __name__ == "__main__":
    main()