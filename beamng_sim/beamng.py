from beamng_sim.lane_detection import process_frame
from beamng_sim.pid_controller import PIDController
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import time
import math


def yaw_to_quat(yaw_deg):
    """Convert yaw angle in degrees to quaternion."""
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)


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
    steering_bias = 0.02  # Small bias to counter drift
    max_steering_change = 0.1  # Maximum steering change per frame
    
    # State variables
    previous_steering = 0.0
    last_time = time.time()
    frame_count = 0

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

            # Process frame for lane detection
            result, metrics = process_frame(img_bgr)

            # Get deviation from lane center
            deviation = metrics.get('deviation', 0.0)
            
            # Handle invalid deviations
            if deviation is None or abs(deviation) > 1.0:
                deviation = 0.0

            # Simple PID control
            steering = pid.update(-deviation, dt)  # Negative because we want opposite correction
            
            # Add bias and limit steering
            steering += steering_bias
            steering = np.clip(steering, -1.0, 1.0)
            
            # Limit rate of change
            steering_change = steering - previous_steering
            if abs(steering_change) > max_steering_change:
                steering = previous_steering + np.sign(steering_change) * max_steering_change
            
            previous_steering = steering

            # Simple throttle control
            throttle = base_throttle * (1.0 - 0.3 * abs(steering))
            throttle = np.clip(throttle, 0.05, 0.3)

            # Apply controls
            vehicle.control(steering=steering, throttle=throttle, brake=0.0)

            # Display image
            cv2.imshow('Lane Detection', result)

            # Print status every 30 frames
            if step_i % 30 == 0:
                print(f"[{step_i}] Deviation: {deviation:.3f}m | Steering: {steering:.3f} | Throttle: {throttle:.3f}")

            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        vehicle.control(throttle=0, steering=0, brake=1.0)
        cv2.destroyAllWindows()
        beamng.close()


if __name__ == "__main__":
    main()
