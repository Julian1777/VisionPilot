import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class LiveLidarDebugWindow:
    """
    Non-blocking live LiDAR visualization using matplotlib.
    Much better for debugging than open3d which blocks on draw_geometries.
    """
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.suptitle("LiDAR Road Boundary Debug (Live)")
        
        # Set initial view
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.view_init(elev=20, azim=45)
        
        # Scatter plot for points
        self.scatter = self.ax.scatter([], [], [], c='yellow', marker='o', s=5, label='LiDAR Points')
        
        # Line collections for boundaries
        self.lines = []
        
        print("✓ LiDAR live debug window initialized (matplotlib)")
        plt.ion()  # Interactive mode - non-blocking

    def update(self, points, boundaries=None):
        """
        Update visualization with new points and boundaries.
        Non-blocking - returns immediately.
        """
        try:
            self.ax.clear()
            
            # Convert points to numpy array
            points = np.array(points) if points is not None else np.empty((0, 3))
            
            if len(points) == 0:
                print("⚠️  No LiDAR points received")
                # Show origin marker for reference
                self.ax.scatter([0], [0], [0], c='red', marker='x', s=100, label='Origin')
            else:
                print(f"✓ Displaying {len(points)} points")
                print(f"  Point range: X[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}], " +
                      f"Y[{points[:, 1].min():.1f}, {points[:, 1].max():.1f}], " +
                      f"Z[{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]")
                
                # Plot points colored by height (Z)
                scatter = self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                         c=points[:, 2], cmap='viridis', marker='o', s=10, alpha=0.6)
                plt.colorbar(scatter, ax=self.ax, label='Height (Z)')
            
            # Plot boundaries if provided
            if boundaries:
                print(f"✓ Drawing {len(boundaries)} boundaries")
                for i, boundary in enumerate(boundaries):
                    if not boundary:
                        continue
                    
                    y_start, y_end = boundary.get('y_bin', (0, 0))
                    left = boundary.get('left')
                    right = boundary.get('right')
                    mean_z = boundary.get('mean_road_height', 0)
                    
                    # Left boundary (green line)
                    if left is not None:
                        self.ax.plot([left, left], [y_start, y_end], [mean_z, mean_z], 
                                    'g-', linewidth=2, label=f'Left {i}' if i == 0 else '')
                    
                    # Right boundary (red line)
                    if right is not None:
                        self.ax.plot([right, right], [y_start, y_end], [mean_z, mean_z], 
                                    'r-', linewidth=2, label=f'Right {i}' if i == 0 else '')
            
            # Set axis limits with some padding
            if len(points) > 0:
                x_min, x_max = points[:, 0].min() - 2, points[:, 0].max() + 2
                y_min, y_max = points[:, 1].min() - 2, points[:, 1].max() + 2
                z_min, z_max = points[:, 2].min() - 0.5, points[:, 2].max() + 0.5
            else:
                x_min, x_max = -5, 5
                y_min, y_max = 0, 20
                z_min, z_max = -1, 2
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
            
            self.ax.set_xlabel('X (Left/Right)')
            self.ax.set_ylabel('Y (Forward)')
            self.ax.set_zlabel('Z (Up)')
            self.ax.legend()
            
            plt.pause(0.001)  # Small pause to allow rendering
            
        except Exception as e:
            print(f"❌ LiDAR debug update error: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """Close the visualization window."""
        plt.close('all')
        print("LiDAR debug window closed")


# Alternative: Simple 2D top-down view (faster, easier to debug)
class LiveLidarDebugWindow2D:
    """
    Simple 2D top-down view of LiDAR points and road boundaries.
    Very fast and good for debugging lane detection.
    """
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.fig.suptitle("LiDAR Top-Down View (2D Debug)")
        self.ax.set_xlabel('X (Left/Right) [m]')
        self.ax.set_ylabel('Y (Forward) [m]')
        self.ax.set_aspect('equal')
        plt.ion()  # Non-blocking
        print("✓ LiDAR 2D debug window initialized")

    def update(self, points, boundaries=None):
        """Update 2D view."""
        try:
            self.ax.clear()
            
            points = np.array(points) if points is not None else np.empty((0, 3))
            
            if len(points) == 0:
                print("⚠️  No points to display")
                self.ax.scatter([0], [0], c='red', marker='x', s=200, label='Origin')
            else:
                print(f"✓ Top-down: {len(points)} points")
                # Plot points, color by height
                scatter = self.ax.scatter(points[:, 0], points[:, 1], 
                                         c=points[:, 2], cmap='viridis', s=20, alpha=0.7)
                plt.colorbar(scatter, ax=self.ax, label='Height')
            
            # Draw boundaries
            if boundaries:
                for boundary in boundaries:
                    if boundary:
                        y_start, y_end = boundary.get('y_bin', (0, 0))
                        left = boundary.get('left')
                        right = boundary.get('right')
                        
                        if left is not None:
                            self.ax.axvline(left, color='g', linewidth=2, alpha=0.7, label='Left')
                        if right is not None:
                            self.ax.axvline(right, color='r', linewidth=2, alpha=0.7, label='Right')
            
            if len(points) > 0:
                self.ax.set_xlim(points[:, 0].min() - 2, points[:, 0].max() + 2)
                self.ax.set_ylim(points[:, 1].min() - 2, points[:, 1].max() + 2)
            else:
                self.ax.set_xlim(-10, 10)
                self.ax.set_ylim(0, 30)
            
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            plt.pause(0.001)
            
        except Exception as e:
            print(f"❌ 2D debug error: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        plt.close('all')
