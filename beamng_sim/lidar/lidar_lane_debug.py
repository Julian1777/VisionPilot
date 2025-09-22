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
        opt.background_color = np.array([0, 0, 0])

    def update(self, points, boundaries):
        try:
            points = np.array(points) if points else np.empty((0, 3))
            
            if len(points) == 0:
                dummy_point = np.array([[0, 0, 0]])
                self.pcd.points = o3d.utility.Vector3dVector(dummy_point)
                self.pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
                
                self.line_set.points = o3d.utility.Vector3dVector()
                self.line_set.lines = o3d.utility.Vector2iVector()
                self.line_set.colors = o3d.utility.Vector3dVector()
                
                self.vis.update_geometry(self.pcd)
                self.vis.update_geometry(self.line_set)
                self.vis.poll_events()
                self.vis.update_renderer()
                return
                
            # Color points by y (height)
            y_vals = points[:, 1]
            if len(y_vals) > 1:
                y_min, y_max = y_vals.min(), y_vals.max()
                if y_max > y_min:
                    y_norm = (y_vals - y_min) / (y_max - y_min)
                else:
                    y_norm = np.zeros_like(y_vals)
            else:
                y_norm = np.zeros_like(y_vals)
                
            colors = np.stack([y_norm, np.zeros_like(y_norm), 1 - y_norm], axis=1)  # Red to Blue

            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            # Build boundary lines
            line_points = []
            lines = []
            line_colors = []
            idx = 0
            for b in boundaries:
                y_start, y_end = b['y_bin']
                left = b['left']
                right = b['right']
                mean_z = b['mean_road_height']
                # Left boundary (green)
                line_points.append([left, y_start, mean_z])
                line_points.append([left, y_end, mean_z])
                lines.append([idx, idx+1])
                line_colors.append([0,1,0])
                idx += 2
                # Right boundary (red)
                line_points.append([right, y_start, mean_z])
                line_points.append([right, y_end, mean_z])
                lines.append([idx, idx+1])
                line_colors.append([1,0,0])
                idx += 2

            if line_points:
                self.line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
                self.line_set.lines = o3d.utility.Vector2iVector(lines)
                self.line_set.colors = o3d.utility.Vector3dVector(line_colors)
            else:
                self.line_set.points = o3d.utility.Vector3dVector()
                self.line_set.lines = o3d.utility.Vector2iVector()
                self.line_set.colors = o3d.utility.Vector3dVector()

            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.line_set)
            self.vis.poll_events()
            self.vis.update_renderer()
            
        except Exception as e:
            print(f"LiDAR debug window update error: {e}")
            # Continue simulation even if debug window fails

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception as e:
            print(f"Error closing LiDAR debug window: {e}")