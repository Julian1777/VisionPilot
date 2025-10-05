import open3d as o3d
import numpy as np

class LiveLidarDebugWindow:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="LiDAR Road Boundary Debug", width=1024, height=768)
        self.pcd = o3d.geometry.PointCloud()
        self.line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.line_set)
        
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.7, 0.7, 0.7])  # Light gray
        opt.point_size = 5.0
        
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 1, -0.5])
        ctr.set_lookat([0, 10, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.3)

        print("LiDAR debug window initialized with camera setup")

    def update(self, points, boundaries):
        try:
            print(f"Debug window update: {len(points) if points else 0} points, {len(boundaries) if boundaries else 0} boundaries")
            points = np.array(points) if points is not None else np.empty((0, 3))
            if len(points) == 0:
                print("No points - showing dummy point")
                points = np.array([[0, 5, 0]])
                colors = np.array([[1.0, 1.0, 0.0]])
            else:
                print(f"Point cloud range: X[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}], Y[{points[:, 1].min():.1f}, {points[:, 1].max():.1f}], Z[{points[:, 2].min():.1f}, {points[:, 2].max():.1f}")
                print("First 5 points:", points[:5])
                colors = np.tile([1.0, 1.0, 0.0], (len(points), 1))
                points = np.vstack([points, [0, 10, 0]])
                colors = np.vstack([colors, [1.0, 0.0, 1.0]])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Build boundary lines
            line_points = []
            lines = []
            line_colors = []
            idx = 0
            print(f"Processing {len(boundaries) if boundaries else 0} boundaries")
            for i, b in enumerate(boundaries):
                if not b:
                    continue
                y_start, y_end = b['y_bin']
                left = b['left']
                right = b['right']
                mean_z = b['mean_road_height']
                print(f"Boundary {i}: y[{y_start:.1f}-{y_end:.1f}], left={left}, right={right}, z={mean_z:.2f}")
                # Left boundary (green)
                if left is not None:
                    line_points.append([left, y_start, mean_z])
                    line_points.append([left, y_end, mean_z])
                    lines.append([idx, idx+1])
                    line_colors.append([0,1,0])
                    idx += 2
                # Right boundary (red)
                if right is not None:
                    line_points.append([right, y_start, mean_z])
                    line_points.append([right, y_end, mean_z])
                    lines.append([idx, idx+1])
                    line_colors.append([1,0,0])
                    idx += 2
            line_set = o3d.geometry.LineSet()
            if line_points:
                line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
            else:
                line_set.points = o3d.utility.Vector3dVector()
                line_set.lines = o3d.utility.Vector2iVector()
                line_set.colors = o3d.utility.Vector3dVector()

            if len(points) > 0:
                center = points.mean(axis=0)
                extent = points.max(axis=0) - points.min(axis=0)
                zoom = 0.5 * min(1.0, 50.0 / max(extent.max(), 1.0))  # heuristic zoom
                def custom_view(vis):
                    ctr = vis.get_view_control()
                    ctr.set_lookat(center.tolist())
                    ctr.set_front([0, 1, 0])
                    ctr.set_up([0, 0, 1])
                    ctr.set_zoom(zoom)
                o3d.visualization.draw_geometries([pcd, line_set], window_name="LiDAR Debug Frame", width=1024, height=768, render_option=None, view_control=custom_view)
            else:
                o3d.visualization.draw_geometries([pcd, line_set], window_name="LiDAR Debug Frame", width=1024, height=768)
        except Exception as e:
            print(f"LiDAR debug window update error: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception as e:
            print(f"Error closing LiDAR debug window: {e}")