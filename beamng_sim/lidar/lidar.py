import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def collect_lidar_data(beamng, lidar_data):
    
    if lidar_data is None:
        print("LiDAR data is None")
        return []
    
    print(f"LiDAR data type: {type(lidar_data)}")
    print(f"LiDAR data: {lidar_data}")
        
    readings_data = lidar_data
    point_cloud = readings_data.get("pointCloud", [])
    
    print(f"Point cloud length: {len(point_cloud)}")
    if len(point_cloud) > 0:
        print(f"First few points: {point_cloud[:5]}")
        if len(point_cloud) > 10:
            points_array = np.array(point_cloud)
            print(f"X range: {points_array[:, 0].min():.2f} to {points_array[:, 0].max():.2f}")
            print(f"Y range: {points_array[:, 1].min():.2f} to {points_array[:, 1].max():.2f}")  
            print(f"Z range: {points_array[:, 2].min():.2f} to {points_array[:, 2].max():.2f}")
    
    return point_cloud