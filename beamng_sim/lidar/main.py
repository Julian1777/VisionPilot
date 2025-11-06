from beamng_sim.lidar.lidar import collect_lidar_data
import numpy as np

from .lane_boundry import detect_lane_boundaries
from .lidar_preprocessing import LidarPreprocessor

_preprocessor = LidarPreprocessor()

def process_frame(lidar_sensor, beamng, speed, debug_window=None, vehicle=None, car_position=None, car_direction=None):
    try:
        lidar_data = lidar_sensor.poll()
        if lidar_data is None:
            print("Warning: LiDAR sensor returned None")
            return {}, []

        point_cloud = collect_lidar_data(beamng, lidar_data)
        if len(point_cloud) == 0:
            print("Warning: Empty LiDAR point cloud")
            return {}, []

        car_yaw = None
        if car_direction is not None:
            car_yaw = np.arctan2(car_direction[1], car_direction[0])
            print(f"Car direction: {car_direction}, Car yaw: {np.degrees(car_yaw):.1f} degrees")

        # Passthrough filtering temporarily disabled
        # ...existing code...
        filtered_points = point_cloud
        print(f"Bypassing passthrough: {len(filtered_points)} points (from {len(point_cloud)})")

        if len(filtered_points) == 0:
            print("Warning: No points after filtering")
            print(f"  Point cloud range: X=[{point_cloud[:, 0].min():.1f}, {point_cloud[:, 0].max():.1f}]")
            print(f"  Point cloud range: Y=[{point_cloud[:, 1].min():.1f}, {point_cloud[:, 1].max():.1f}]")
            print(f"  Point cloud range: Z=[{point_cloud[:, 2].min():.1f}, {point_cloud[:, 2].max():.1f}]")
            return {}, []

        boundaries = detect_lane_boundaries(filtered_points)
        return boundaries, filtered_points
    except Exception as e:
        print(f"LiDAR processing error: {e}")
        return {}, []