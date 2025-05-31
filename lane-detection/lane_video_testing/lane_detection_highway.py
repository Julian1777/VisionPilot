import cv2 as cv
import numpy as np
import os

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

def find_lane_base_points(warped_image):
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    return leftx_base, rightx_base, histogram

def enhance_lane_from_color(warped_color):
    hls = cv.cvtColor(warped_color, cv.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 90) & (s_channel <= 255)] = 255
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 120) & (l_channel <= 255)] = 255
    
    combined_binary = cv.bitwise_or(s_binary, l_binary)
    
    kernel = np.ones((3,3), np.uint8)
    binary = cv.morphologyEx(combined_binary, cv.MORPH_CLOSE, kernel)
    
    return binary

def transform_points_back(points, matrix):
    if not points:
        return []
    
    pts = np.array(points, dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    
    transformed_pts = cv.perspectiveTransform(pts, matrix)
    
    return [tuple(map(int, pt[0])) for pt in transformed_pts]

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

def create_lane_path(image, left_points, right_points):
    lane_path = np.zeros_like(image)
    
    if not left_points or not right_points:
        return lane_path
    
    if left_points and right_points and len(left_points) > 1 and len(right_points) > 1:
        lane_full_points = np.vstack((left_points, right_points[::-1]))
        cv.fillPoly(lane_path, [np.array(lane_full_points, dtype=np.int32)], (0, 0, 255))  # Blue
    
    return lane_path

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

def lane_detection(frame):
    global prev_left_curve, prev_left_degree, prev_right_curve, prev_right_degree
    global prev_left_base, prev_right_base, left_base_history, right_base_history
    height, width = frame.shape[:2]
    
    processed_image = preprocess_image(frame)

    roi_image, roi_polygon = roi(processed_image)

    warped_image, transform_matrix = perspective_transform(roi_image)
    binary = enhance_lane_from_color(warped_image)

    left_base_x, right_base_x, histogram = find_lane_base_points(binary, prev_left_base, prev_right_base)

    if prev_left_base is not None and abs(left_base_x - prev_left_base) > binary.shape[1] * 0.1:
        left_base_x = prev_left_base
    
    if prev_right_base is not None and abs(right_base_x - prev_right_base) > binary.shape[1] * 0.1:
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
    
    lines = cv.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=20,
        maxLineGap=20
    )
    
    left_points_x, left_points_y, right_points_x, right_points_y = [], [], [], []

    if lines is not None:
        window_width = binary.shape[1] * 0.07
        
        left_window_min = max(0, left_base_x - window_width)
        left_window_max = left_base_x + window_width
        
        right_window_min = right_base_x - window_width
        right_window_max = min(binary.shape[1], right_base_x + window_width)
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if abs(x2 - x1) < 1:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            mid_x = (x1 + x2) / 2
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            
            if slope < -0.1 and left_window_min < mid_x < left_window_max and length > 10:
                left_points_x.extend([x1, x2])
                left_points_y.extend([y1, y2])
            elif slope > 0.1 and right_window_min < mid_x < right_window_max and length > 10:
                right_points_x.extend([x1, x2])
                right_points_y.extend([y1, y2])

    left_curve, left_degree = fit_lane_curve(left_points_x, left_points_y) if len(left_points_x) >= 2 else (prev_left_curve, prev_left_degree)
    right_curve, right_degree = fit_lane_curve(right_points_x, right_points_y) if len(right_points_x) >= 2 else (prev_right_curve, prev_right_degree)

    if left_curve is not None:
        prev_left_curve, prev_left_degree = left_curve, left_degree
    if right_curve is not None:
        prev_right_curve, prev_right_degree = right_curve, right_degree

    left_points, right_points = [], []
    y_start = int(height * 0.25)
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
    
    print(f"Lines detected: {0 if lines is None else len(lines)}")
    print(f"Left points: {len(left_points_x)}, Right points: {len(right_points_x)}")

    cv.imshow("Binary Image", cv.resize(binary, (400, 300)))

    plt_histogram = np.zeros((400, binary.shape[1], 3), dtype=np.uint8)
    if np.max(histogram) > 0:
        hist_normalized = histogram/np.max(histogram)*300
        for i in range(binary.shape[1]):
            if hist_normalized[i] > 0:
                cv.line(plt_histogram, (i, 399), (i, 399-int(hist_normalized[i])), (0, 0, 255), 1)
    cv.line(plt_histogram, (left_base_x, 0), (left_base_x, 399), (0, 255, 0), 2)
    cv.line(plt_histogram, (right_base_x, 0), (right_base_x, 399), (0, 0, 255), 2)
    cv.imshow("Histogram", plt_histogram)

    point_debug = warped_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(point_debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
    for x, y in zip(left_points_x, left_points_y):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < point_debug.shape[1] and 0 <= y_int < point_debug.shape[0]:
            cv.circle(point_debug, (x_int, y_int), 5, (0, 255, 0), -1)
    
    for x, y in zip(right_points_x, right_points_y):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < point_debug.shape[1] and 0 <= y_int < point_debug.shape[0]:
            cv.circle(point_debug, (x_int, y_int), 5, (0, 0, 255), -1)

    cv.rectangle(point_debug, 
             (int(left_window_min), 0), 
             (int(left_window_max), binary.shape[0]),
             (0, 255, 255), 2)  # Yellow rectangle
             
    cv.rectangle(point_debug, 
                (int(right_window_min), 0), 
                (int(right_window_max), binary.shape[0]),
                (0, 255, 255), 2)  # Yellow rectangle
    
    cv.imshow("Detected Points", cv.resize(point_debug, (400, 300)))
    
    return left_points_original, right_points_original, roi_image, roi_polygon, warped_image

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    
    while cap.isOpened():
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
        
        
        if cv.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../test_vids/nl_highway.mp4"
    print(f"Processing video: {video_path}")
    process_video(video_path)
