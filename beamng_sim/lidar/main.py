from beamng_sim.lidar.lidar import collect_lidar_data
import numpy as np

from .lane_boundry import detect_lane_boundaries

bin_size = 1.0
y_min, y_max = 0, 30

def process_frame(lidar_sensor, beamng, speed, debug_window=None):
    try:
        lidar_data = lidar_sensor.poll()
        if lidar_data is None:
            print("Warning: LiDAR sensor returned None")
            return []
            
        point_cloud = collect_lidar_data(beamng, lidar_data)

        if not point_cloud or len(point_cloud) == 0:
            print("Warning: Empty LiDAR point cloud")
            if debug_window is not None:
                debug_window.update([], [])
            return []

        filtered_points = []
        for point in point_cloud:
            x, y, z = point[:3]
            distance = (x**2 + y**2 + z**2) ** 0.5
            if speed <= 70 and distance > 60:
                continue
            else:
                filtered_points.append((x, y, z))
                
        if len(filtered_points) == 0:
            print("Warning: No valid LiDAR points after filtering")
            if debug_window is not None:
                debug_window.update([], [])
            return []
                
        boundaries = detect_lane_boundaries(filtered_points)

        if debug_window is not None:
            try:
                debug_window.update(filtered_points, boundaries)
            except Exception as debug_e:
                print(f"Debug window update failed: {debug_e}")

        if boundaries:
            for b in boundaries:
                print(
                    f"Y bin: {b['y_bin']}, "
                    f"Left: {b['left']}, "
                    f"Right: {b['right']}, "
                    f"Mean road height: {b['mean_road_height']:.2f}"
                )
        return boundaries
    except Exception as e:
        print(f"LiDAR processing error: {e}")
        return []