import cv2 as cv
import numpy as np
import os

# Function to preprocess including color space conversion and masking the frames for lane detection 

def preprocess_image(image):
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([10, 60, 60])
    upper_yellow = np.array([40, 210, 255])
    yellow_mask = cv.inRange(hls, lower_yellow, upper_yellow)

    combined_mask = cv.bitwise_or(white_mask, yellow_mask)
    masked_image = cv.bitwise_and(image, image, mask=combined_mask)
    return masked_image

# Function to apply perspective transform to the image to get a bird's eye view

def perspective_transform(image):
    height, width = image.shape[:2]
    src = np.float32([
        [width * 0.4, height * 0.65],  # Top left
        [width * 0.6, height * 0.65],  # Top right
        [width * 0.2, height * 0.9],   # Bottom left
        [width * 0.8, height * 0.9]    # Bottom right
    ])
    
    dst = np.float32([
        [width * 0.25, 0],             # Top left
        [width * 0.75, 0],             # Top right
        [width * 0.25, height],        # Bottom left
        [width * 0.75, height]         # Bottom right
    ])

    matrix = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(image, matrix, (width, height))
    return warped, matrix

# Function to define the region of interest (ROI) for lane detection

def roi(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0.1 * width, 0.78 * height),      # Bottom left
        (0.3 * width, 0.65 * height),       # Mid left
        (0.4 * width, 0.55 * height),       # Top left
        (0.55 * width, 0.55 * height),      # Top right
        (0.65 * width, 0.7 * height),       # Mid right
        (0.7 * width, 0.78 * height)        # Bottom right
    ]], dtype=np.float32).astype(np.int32)

    cv.fillPoly(mask, polygon, 255)
    masked_img = cv.bitwise_and(image, mask)
    return masked_img, polygon

# Function to filter detected lines into left and right lanes based on their slopes and positions

def filter_lanes(lines, image_width):
    if lines is None:
        return None, None
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        mid_x = (x1 + x2) / 2

        if abs(slope) > 0.1 and abs(slope) < 2.0 and length > 30:
            if slope < 0 and mid_x < image_width * 0.5:
                left_lines.append(line)
            elif slope > 0 and mid_x > image_width * 0.5:
                right_lines.append(line)

    return left_lines, right_lines

# Function to convert line parameters (slope and intercept) into coordinates for drawing lines on the image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.65)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

prev_right_fit_average = np.array([-0.5, 300])
prev_left_fit_average = np.array([0.5, -50])

prev_left_curve = None
prev_left_degree = 1
prev_right_curve = None
prev_right_degree = 1

prev_left_base = None
prev_right_base = None
lane_history_size = 10
left_base_history = []
right_base_history = []

# Function to fit a polynomial curve to the lane points and determine its degree

def fit_lane_curve(xs, ys):
    if len(xs) < 2:
        return None, 1
    linear = np.polyfit(ys, xs, 1)

    if len(xs) >= 3:
        quad = np.polyfit(ys, xs, 2)

        y_vals = np.linspace(min(ys), max(ys), num=10)
        x_linear = np.polyval(linear, y_vals)
        x_quad = np.polyval(quad, y_vals)

        deviation = np.mean(np.abs(x_linear - x_quad))

        if deviation > 10:
            return quad, 2
        
    return linear, 1

# Function to fit lanes based on detected lines and previous lane curves

def fit_lanes(image, lines):
    global prev_left_curve, prev_left_degree, prev_right_curve, prev_right_degree
    height, width = image.shape[:2]

    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if abs(x2 - x1) < 1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            mid_x = (x1 + x2) / 2

            if slope < -0.3 and mid_x < width * 0.5:
                left_points_x.extend([x1, x2])
                left_points_y.extend([y1, y2])
            elif slope > 0.2 and mid_x > width * 0.6:
                right_points_x.extend([x1, x2])
                right_points_y.extend([y1, y2])

    left_curve, left_degree = None, 1
    right_curve, right_degree = None, 1

    try:
        if len(left_points_x) >= 2:
            left_curve, left_degree = fit_lane_curve(left_points_x, left_points_y)
            if left_curve is not None:
                prev_left_curve, prev_left_degree = left_curve, left_degree
        elif prev_left_curve is not None:
            left_curve, left_degree = prev_left_curve, prev_left_degree
    except Exception as e:
        print(f"Failed to fit left lane curve: {str(e)}")
        if prev_left_curve is not None:
            left_curve, left_degree = prev_left_curve, prev_left_degree

    try:
        if len(right_points_x) >= 2:
            right_curve, right_degree = fit_lane_curve(right_points_x, right_points_y)
            if right_curve is not None:
                prev_right_curve, prev_right_degree = right_curve, right_degree
        elif prev_right_curve is not None:
            right_curve, right_degree = prev_right_curve, prev_right_degree
    except Exception as e:
        print(f"Failed to fit right lane curve: {str(e)}")
        if prev_right_curve is not None:
            right_curve, right_degree = prev_right_curve, prev_right_degree

    left_lane_points = []
    right_lane_points = []
    
    y_start = int(height * 0.65)
    y_end = height
    num_points = 25
    
    y_coords = np.linspace(y_start, y_end, num_points)
    
    if left_curve is not None:
        for y in y_coords:
            if left_degree == 2:
                x = left_curve[0] * y**2 + left_curve[1] * y + left_curve[2]
            else:
                x = left_curve[0] * y + left_curve[1]
            if 0 <= x < width:
                left_lane_points.append((int(x), int(y)))
    
    if right_curve is not None:
        for y in y_coords:
            if right_degree == 2:
                x = right_curve[0] * y**2 + right_curve[1] * y + right_curve[2]
            else:
                x = right_curve[0] * y + right_curve[1]
            if 0 <= x < width:
                right_lane_points.append((int(x), int(y)))
    
    return left_lane_points, right_lane_points

# Function to find the base points of the lanes using histogram analysis

def find_lane_base_points(warped_image):
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    return leftx_base, rightx_base, histogram

# Function to enhance lane detection from color channels using HLS color space

def enhance_lane_from_color(warped_color):
    hls = cv.cvtColor(warped_color, cv.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    l_mean = np.mean(l_channel)
    s_mean = np.mean(s_channel)

    l_threshold = max(80, min(130, l_mean * 0.9))
    s_threshold = max(50, min(100, s_mean * 0.9))
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold) & (s_channel <= 255)] = 255
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_threshold) & (l_channel <= 255)] = 255
    
    combined_binary = cv.bitwise_or(s_binary, l_binary)
    
    kernel = np.ones((3,3), np.uint8)
    binary = cv.morphologyEx(combined_binary, cv.MORPH_CLOSE, kernel)
    
    return binary

# Function to transform points back to the original image coordinates after perspective transformation

def transform_points_back(points, matrix):
    if not points:
        return []
    
    pts = np.array(points, dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    
    transformed_pts = cv.perspectiveTransform(pts, matrix)
    
    return [tuple(map(int, pt[0])) for pt in transformed_pts]

# Function to find the base points of the lanes using histogram analysis with previous lane positions as fallback

def find_lane_base_points(warped_image, previous_left=None, previous_right=None):
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    
    car_position = warped_image.shape[1] // 2
    
    search_window = int(warped_image.shape[1] * 0.15)
    
    left_search_start = max(0, car_position - search_window * 2)
    left_search_end = car_position - int(search_window * 0.5)
    
    right_search_start = car_position + int(search_window * 0.5)
    right_search_end = min(warped_image.shape[1], car_position + search_window * 2)
    
    left_histogram = histogram[left_search_start:left_search_end]
    leftx_base = left_search_start + np.argmax(left_histogram) if np.max(left_histogram) > 0 else previous_left
    
    right_histogram = histogram[right_search_start:right_search_end]
    rightx_base = right_search_start + np.argmax(right_histogram) if np.max(right_histogram) > 0 else previous_right
    
    return leftx_base, rightx_base, histogram

# Function to track lanes using sliding windows approach

def track_lanes_with_windows(img, start_x, previous_points_x=None, previous_points_y=None):
    height = img.shape[0]
    width = img.shape[1]

    if previous_points_x and len(previous_points_x) > 10:
        if max(previous_points_x) >= width or min(previous_points_x) < 0:
            previous_points_x = None
            previous_points_y = None
        else:
            x_range = max(previous_points_x) - min(previous_points_x)
            if x_range < 20:
                previous_points_x = None
                previous_points_y = None

    if start_x < 0 or start_x >= width:
        if previous_points_x and len(previous_points_x) > 10:
            return previous_points_x, previous_points_y
        return [], []

    num_windows = 10
    window_height = height // num_windows
    window_half_width = 40
    min_pixels = 25

    lane_points_x = []
    lane_points_y = []
    current_x = start_x
    
    bottom_y = height - 1
    lane_points_x.extend([current_x] * 3)
    lane_points_y.extend([bottom_y] * 3)

    for window in range(num_windows):
        y_top = height - (window + 1) * window_height
        y_bottom = height - window * window_height

        window_left = max(0, current_x - window_half_width)
        window_right = min(img.shape[1], current_x + window_half_width)

        window_region = img[y_top:y_bottom, window_left:window_right]

        nonzero_indices = np.where(window_region > 0)
        if len(nonzero_indices[0]) > min_pixels:
            y_indices = nonzero_indices[0] + y_top
            x_indices = nonzero_indices[1] + window_left

            current_x = int(np.mean(x_indices))

            lane_points_x.extend(x_indices)
            lane_points_y.extend(y_indices)

    if len(lane_points_x) <= 3 and previous_points_x and len(previous_points_x) > 10:
        print("No points detected, using previous frame")
        return previous_points_x, previous_points_y

    return lane_points_x, lane_points_y

# Function to calculate the confidence of lane detection based on the number of points and their distribution

def calculate_lane_confidence(left_points_x, left_points_y, right_points_x, right_points_y, image_width):
    expected_width = image_width * 0.4
    
    left_confidence = 0.0
    min_points_required = 20
    left_points_ratio = min(1.0, len(left_points_x) / min_points_required)
    left_confidence += 0.6 * left_points_ratio
    
    if len(left_points_x) > 10:
        left_y_sorted = sorted(zip(left_points_x, left_points_y), key=lambda p: p[1])
        left_x_values = [x for x, _ in left_y_sorted]
        
        if len(left_x_values) > 1:
            left_std = np.std(left_x_values)
            std_score = max(0, 1.0 - (left_std / 50.0))
            left_confidence += 0.4 * std_score
    
    right_confidence = 0.0
    right_points_ratio = min(1.0, len(right_points_x) / min_points_required)
    right_confidence += 0.6 * right_points_ratio
    
    if len(right_points_x) > 10:
        right_y_sorted = sorted(zip(right_points_x, right_points_y), key=lambda p: p[1])
        right_x_values = [x for x, _ in right_y_sorted]
        
        if len(right_x_values) > 1:
            right_std = np.std(right_x_values)
            std_score = max(0, 1.0 - (right_std / 50.0))
            right_confidence += 0.4 * std_score
    
    width_confidence = 0.0
    if len(left_points_x) > 10 and len(right_points_x) > 10:
        widths = []
        left_y_sorted = sorted(zip(left_points_x, left_points_y), key=lambda p: p[1])
        right_y_sorted = sorted(zip(right_points_x, right_points_y), key=lambda p: p[1])
        
        sample_points = min(5, len(left_y_sorted), len(right_y_sorted))
        step = max(1, len(left_y_sorted) // sample_points)
        
        for i in range(0, len(left_y_sorted), step):
            if i >= len(left_y_sorted) or i >= len(right_y_sorted):
                break
            left_x = left_y_sorted[i][0]
            right_x = right_y_sorted[i][0]
            width = right_x - left_x
            if width > 0:
                widths.append(width)
        
        if widths:
            width_std = np.std(widths) if len(widths) > 1 else 0
            width_mean = np.mean(widths)
            
            std_score = max(0, 1.0 - (width_std / 50.0))
            width_confidence += 0.5 * std_score
            
            width_error = abs(width_mean - expected_width) / expected_width
            width_score = max(0, 1.0 - min(1.0, width_error))
            width_confidence += 0.5 * width_score
    
    combined_confidence = (left_confidence * 0.4 + right_confidence * 0.4 + width_confidence * 0.2)
    
    return combined_confidence, left_confidence, right_confidence

# Function to create a lane path image based on detected lane points

def create_lane_path(image, left_points, right_points):
    lane_path = np.zeros_like(image)
    
    if not left_points or not right_points:
        return lane_path
    
    if left_points and right_points and len(left_points) > 1 and len(right_points) > 1:
        lane_full_points = np.vstack((left_points, right_points[::-1]))
        cv.fillPoly(lane_path, [np.array(lane_full_points, dtype=np.int32)], (0, 0, 255))  # Blue
    
    return lane_path

# Function to display lane lines on the original image

def display_lane_lines(image, left_points, right_points):
    line_image = np.zeros_like(image)
    
    if left_points and len(left_points) > 1:
        for i in range(len(left_points) - 1):
            pt1 = left_points[i]
            pt2 = left_points[i + 1]
            cv.line(line_image, pt1, pt2, (0, 255, 0), 8)  # Green
    
    if right_points and len(right_points) > 1:
        for i in range(len(right_points) - 1):
            pt1 = right_points[i]
            pt2 = right_points[i + 1]
            cv.line(line_image, pt1, pt2, (0, 0, 255), 8)  # Red
            
    return line_image

# Function to detect lanes in a single frame of video

def lane_detection(frame):
    global prev_left_curve, prev_left_degree, prev_right_curve, prev_right_degree
    global prev_left_base, prev_right_base, left_base_history, right_base_history
    global prev_left_points_x, prev_left_points_y, prev_right_points_x, prev_right_points_y

    if 'prev_left_points_x' not in globals():
        global prev_left_points_x, prev_left_points_y, prev_right_points_x, prev_right_points_y
        prev_left_points_x, prev_left_points_y = [], []
        prev_right_points_x, prev_right_points_y = [], []
        
    if 'prev_confidence' not in globals():
        global prev_confidence, confidence_history
        prev_confidence = 0
        confidence_history = []


    height, width = frame.shape[:2]
    
    processed_image = preprocess_image(frame)
    roi_image, roi_polygon = roi(processed_image)
    warped_image, transform_matrix = perspective_transform(roi_image)
    binary = enhance_lane_from_color(warped_image)

    left_base_x, right_base_x, histogram = find_lane_base_points(binary, prev_left_base, prev_right_base)

    if left_base_x is None:
        left_base_x = width // 4
    if right_base_x is None:
        right_base_x = width * 3 // 4

    if prev_left_base is not None and abs(left_base_x - prev_left_base) > binary.shape[1] * 0.05:
        left_base_x = prev_left_base
    
    if prev_right_base is not None and abs(right_base_x - prev_right_base) > binary.shape[1] * 0.05:
        right_base_x = prev_right_base
    
    left_base_history.append(left_base_x)
    right_base_history.append(right_base_x)
    
    if len(left_base_history) > lane_history_size:
        left_base_history.pop(0)
    if len(right_base_history) > lane_history_size:
        right_base_history.pop(0)
    
    left_base_x = int(np.mean(left_base_history))
    right_base_x = int(np.mean(right_base_history))
    
    prev_left_base = left_base_x
    prev_right_base = right_base_x

    left_points_x, left_points_y = track_lanes_with_windows(
        binary, left_base_x, prev_left_points_x, prev_left_points_y)
    
    right_points_x, right_points_y = track_lanes_with_windows(
        binary, right_base_x, prev_right_points_x, prev_right_points_y)
    
    confidence, left_confidence, right_confidence = calculate_lane_confidence(
        left_points_x, left_points_y, right_points_x, right_points_y, binary.shape[1])
    
    confidence_history.append(confidence)
    if len(confidence_history) > 10:
        confidence_history.pop(0)
    avg_confidence = np.mean(confidence_history)
    
    print(f"Lane confidence: {confidence:.2f}, Left: {left_confidence:.2f}, Right: {right_confidence:.2f}, Avg: {avg_confidence:.2f}")
    
    confidence_threshold = 0.7
    if avg_confidence < confidence_threshold and prev_confidence >= confidence_threshold:
        print("Low confidence detection - using previous frame")
        left_points_x, left_points_y = prev_left_points_x, prev_left_points_y
        right_points_x, right_points_y = prev_right_points_x, prev_right_points_y

    prev_left_points_x, prev_left_points_y = left_points_x.copy(), left_points_y.copy()
    prev_right_points_x, prev_right_points_y = right_points_x.copy(), right_points_y.copy()
    prev_confidence = confidence
    
    left_curve, left_degree = fit_lane_curve(left_points_x, left_points_y) if len(left_points_x) >= 2 else (prev_left_curve, prev_left_degree)
    right_curve, right_degree = fit_lane_curve(right_points_x, right_points_y) if len(right_points_x) >= 2 else (prev_right_curve, prev_right_degree)
    
    if left_curve is not None:
        prev_left_curve, prev_left_degree = left_curve, left_degree
    if right_curve is not None:
        prev_right_curve, prev_right_degree = right_curve, right_degree

    
    left_points, right_points = [], []
    y_start = int(height * 0.15)
    y_end = height
    num_points = 45
    y_coords = np.linspace(y_start, y_end, num_points)
    
    if left_curve is not None:
        for y in y_coords:
            if left_degree == 2:
                x = left_curve[0] * y**2 + left_curve[1] * y + left_curve[2]
            else:
                x = left_curve[0] * y + left_curve[1]
            if 0 <= x < width:
                left_points.append((int(x), int(y)))
    
    if right_curve is not None:
        for y in y_coords:
            if right_degree == 2:
                x = right_curve[0] * y**2 + right_curve[1] * y + right_curve[2]
            else:
                x = right_curve[0] * y + right_curve[1]
            if 0 <= x < width:
                right_points.append((int(x), int(y)))
    
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    left_points_original = transform_points_back(left_points, inv_transform_matrix)
    right_points_original = transform_points_back(right_points, inv_transform_matrix)
    
    print(f"Lines detected: {0 if len(left_points_x) + len(right_points_x) == 0 else len(left_points_x) + len(right_points_x)}")
    
    cv.imshow("Binary Image", cv.resize(binary, (400, 300)))
    
    plt_histogram = np.zeros((400, binary.shape[1], 3), dtype=np.uint8)
    if np.max(histogram) > 0:
        hist_normalized = histogram/np.max(histogram)*300
        for i in range(binary.shape[1]):
            if hist_normalized[i] > 0:
                cv.line(plt_histogram, (i, 399), (i, 399-int(hist_normalized[i])), (0, 0, 255), 1)
    cv.line(plt_histogram, (left_base_x, 0), (left_base_x, 399), (0, 255, 0), 2)
    cv.line(plt_histogram, (right_base_x, 0), (right_base_x, 399), (0, 0, 255), 2)
    
    conf_text1 = f"Combined: {confidence:.2f}"
    conf_text2 = f"Left: {left_confidence:.2f}"
    conf_text3 = f"Right: {right_confidence:.2f}"
    cv.putText(plt_histogram, conf_text1, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(plt_histogram, conf_text2, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(plt_histogram, conf_text3, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imshow("Histogram", plt_histogram)
    
    point_debug = warped_image.copy()
    
    for x, y in zip(left_points_x, left_points_y):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < point_debug.shape[1] and 0 <= y_int < point_debug.shape[0]:
            cv.circle(point_debug, (x_int, y_int), 3, (0, 255, 0), -1)
    
    for x, y in zip(right_points_x, right_points_y):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < point_debug.shape[1] and 0 <= y_int < point_debug.shape[0]:
            cv.circle(point_debug, (x_int, y_int), 3, (0, 0, 255), -1)
    
    window_height = binary.shape[0] // 9
    for window in range(9):
        y_top = binary.shape[0] - (window+1) * window_height
        y_bottom = binary.shape[0] - window * window_height
        
        left_x = left_base_x
        if len(left_points_y) > 0:
            window_indices = [i for i, y_val in enumerate(left_points_y) 
                            if y_top <= y_val < y_bottom]
            if window_indices:
                left_x = int(np.mean([left_points_x[i] for i in window_indices]))
        
        cv.rectangle(point_debug, 
                    (int(left_x - 30), y_top), 
                    (int(left_x + 30), y_bottom),
                    (0, 255, 255), 2)
        
        right_x = right_base_x
        if len(right_points_y) > 0:
            window_indices = [i for i, y_val in enumerate(right_points_y) 
                            if y_top <= y_val < y_bottom]
            if window_indices:
                right_x = int(np.mean([right_points_x[i] for i in window_indices]))
        
        cv.rectangle(point_debug, 
                    (int(right_x - 30), y_top), 
                    (int(right_x + 30), y_bottom),
                    (0, 255, 255), 2)
    
    cv.imshow("Detected Points", cv.resize(point_debug, (400, 300)))
    
    return left_points_original, right_points_original, roi_image, roi_polygon, warped_image
    
# Function to process the video and apply lane detection on each frame

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        if 'seek_to' in locals():
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)
            del seek_to
        ret, frame = cap.read()
        if not ret:
            break
            
        left_points, right_points, cropped_image, roi_polygon, warped_image = lane_detection(frame)
        
        lane_lines = display_lane_lines(frame, left_points, right_points)
        lane_path = create_lane_path(frame, left_points, right_points)
        
        combo_image = cv.addWeighted(frame, 0.9, lane_lines, 1, 1)
        combo_image = cv.addWeighted(combo_image, 0.8, lane_path, 0.3, 0)
        
        cv.imshow('Lane Detection', combo_image)
        
        roi_border_img = cropped_image.copy()
        roi_polygon_int = roi_polygon.astype(np.int32)
        cv.polylines(roi_border_img, [roi_polygon_int], isClosed=True, color=(255, 0, 255), thickness=3)
        roi_resized = cv.resize(roi_border_img, (400, 300))
        cv.imshow("ROI-Image", roi_resized)
        
        warped_resized = cv.resize(warped_image, (400, 300))
        cv.imshow("Bird's Eye View", warped_resized)
        
        
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('l'):  # Skip forward 30 frames
            frame_count = min(frame_count + 30, total_frames - 1)
            seek_to = True
            continue
        elif key == ord('j'):  # Skip backward 30 frames
            frame_count = max(frame_count - 30, 0)
            seek_to = True
            continue
        else:
            frame_count += 1
            
    cap.release()
    cv.destroyAllWindows()

# Main function to run the lane detection on a video file
if __name__ == "__main__":
    video_path = "../test_vids/nl_highway.mp4"
    print(f"Processing video: {video_path}")
    process_video(video_path)
