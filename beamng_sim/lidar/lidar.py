import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def collect_lidar_data(beamng, lidar_data):
    # Remove duplicate beamng.control.step call to prevent double-stepping
    # beamng.control.step(10)  # This is already called in main loop
    
    if lidar_data is None:
        return []
        
    readings_data = lidar_data
    point_cloud = readings_data.get("pointCloud", [])
    return point_cloud