import pickle
import numpy as np
import open3d as o3d

file_path = "/Users/jstamm2024/Documents/GitHub/self-driving-car-simulation/beamng_sim/lidar/lidar_data_checkpoint_250.pkl"
with open(file_path, 'rb') as f:
    point_clouds = pickle.load(f)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(point_clouds[-1]))
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="LiDAR Point Cloud", width=1024, height=768, point_show_normal=False)