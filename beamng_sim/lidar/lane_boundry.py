import numpy as np

def detect_lane_boundaries(filtered_points, bin_size=1.0, y_min=0, y_max=30, x_tolerance=0.05, threshold=0.15):
    results = []
    for y_bin_start in np.arange(y_min, y_max, bin_size):
        y_bin_end = y_bin_start + bin_size
        bin_points = [p for p in filtered_points if y_bin_start <= p[1] < y_bin_end]
        if not bin_points:
            continue
        xs = [point[0] for point in bin_points]
        unique_xs = np.unique(xs)
        ground_points = []
        for x in unique_xs:
            # Convert numpy scalar to Python float to avoid numpy array comparison issues
            x = float(x)
            zs_at_x = [z for (x2, y2, z) in bin_points if abs(x2 - x) < x_tolerance]
            if zs_at_x:
                zs_at_x.sort()
                n = max(1, int(0.1 * len(zs_at_x)))
                avg_ground_z = sum(zs_at_x[:n]) / n
                ground_points.append((x, avg_ground_z))
        if not ground_points:
            continue
        mean_road_height = sum(z for x, z in ground_points) / len(ground_points)
        left_boundary = None
        right_boundary = None
        for x, z in ground_points:
            if float(z) > float(mean_road_height) + threshold:
                if left_boundary is None:
                    left_boundary = x
                else:
                    right_boundary = x
                    break
        results.append({
            'y_bin': (y_bin_start, y_bin_end),
            'left': left_boundary,
            'right': right_boundary,
            'mean_road_height': mean_road_height
        })
    return results