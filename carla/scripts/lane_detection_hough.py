import cv2 as cv
import numpy as np



def canny(image):
    gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def roi(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0.0 * width, 0.78 * height),      # Bottom left
        (0.15 * width, 0.65 * height),     # Mid left
        (0.4 * width, 0.45 * height),      # Top left
        (0.6 * width, 0.45 * height),      # Top right
        (0.95 * width, 0.7 * height),      # Mid right
        (1.0 * width, 0.78 * height)       # Bottom right
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
        mid_x = (x1 + x2) / 2

        if slope < 0 and mid_x < image_width * 0.5:
            left_lines.append(line)
        elif slope > 0 and mid_x > image_width * 0.5:
            right_lines.append(line) 

    return left_lines, right_lines

def make_coordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*0.65)
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

prev_right_fit_average = np.array([-0.5, 300])
prev_left_fit_average = np.array([0.5, -50])

prev_left_curve = None
prev_left_degree = 1
prev_right_curve = None
prev_right_degree = 1

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

            if slope < -0.2 and mid_x < width * 0.6:
                left_points_x.extend([x1, x2])
                left_points_y.extend([y1, y2])
            elif slope > 0.3 and mid_x > width * 0.5:
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

    

def create_lane_path(image, lane_lines):
    lane_path = np.zeros_like(image)
    if lane_lines is None or len(lane_lines) < 2:
        return lane_path
    
    left_line = lane_lines[0]
    right_line = lane_lines[1]
    
    lane_points = np.array([
        [left_line[0], left_line[1]],   # Bottom left
        [left_line[2], left_line[3]],   # Top left
        [right_line[2], right_line[3]], # Top right
        [right_line[0], right_line[1]]  # Bottom right
    ], dtype=np.int32)
    
    cv.fillPoly(lane_path, [lane_points], (0, 100, 0))
    
    return lane_path
 
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),8)

    return line_image

def lane_detection(frame):

    height, width = frame.shape[:2]

    canny_image=canny(frame)
    
    cropped_image, roi_polygon=roi(canny_image)
    
    lines = cv.HoughLinesP(
        cropped_image,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=15,
        maxLineGap=20
    )

    left_points, right_points = fit_lanes(frame, lines)


    result_lines = []
    for i in range(len(left_points) - 1):
        result_lines.append((left_points[i], left_points[i+1]))
    
    for i in range(len(right_points) - 1):
        result_lines.append((right_points[i], right_points[i+1]))
    
    return result_lines