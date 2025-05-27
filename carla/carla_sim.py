import carla
import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv
import os
import threading
import time
import queue
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from ultralytics import YOLO

FRAME_SKIP = 2
MODELS = {}

carla_vehicle = None
carla_camera = None

last_lane_update = 0
last_light_update = 0
last_sign_update = 0
last_vehicle_update = 0

update_interval = 5
window_width, window_height = 800, 600
frame_count = 0

playing = False
delay = 30

main_photo = None
lane_hough_photo = None
lane_ml_photo = None
sign_photo = None
light_detect_class_photo = None
vehicle_ped_photo = None

current_control = carla.VehicleControl()

def carla_simulation_setup():
    global carla_vehicle, carla_camera, carla_front_camera, main_image_id

    window_width, window_height = 800, 600

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town01')

    blueprint_library = world.get_blueprint_library()
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]
    carla_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(
        carla.Location(x=-6, y=0, z=3),
        carla.Rotation(pitch=-15)
    )
    carla_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=carla_vehicle)
    carla_camera.listen(carla_camera_callback)

    front_camera_transform = carla.Transform(
        carla.Location(x=1.2, y=0, z=1.4),
        carla.Rotation(pitch=0)
    )
    carla_front_camera = world.spawn_actor(camera_bp, front_camera_transform, attach_to=carla_vehicle)
    carla_front_camera.listen(carla_front_camera_callback)
    
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla_vehicle.get_transform().transform(carla.Location(x=-12, y=0, z=6)),
        carla_vehicle.get_transform().rotation
    )
    spectator.set_transform(transform)

    main_image_id = main_canvas.create_image(0, 0, anchor=tk.NW)


photo_refs = {}
def numpy_to_tkinter(array, window_id="main"):
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    if array.shape[0] != window_height or array.shape[1] != window_width:
        array = cv.resize(array, (window_width, window_height))

    img = Image.fromarray(array)
    photo = ImageTk.PhotoImage(image=img)

    photo_refs[window_id] = photo
    
    return photo

photo_refs_list = []

def carla_camera_callback(image):
    global frame_count, photo_refs_list
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = array[:, :, :3]  # BGR
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.array(frame, copy=True)

    frame_main = frame.copy()
    photo_main = numpy_to_tkinter(frame_main, window_id="main")
    
    photo_refs_list.append(photo_main)
    if len(photo_refs_list) > 20:
        photo_refs_list = photo_refs_list[-20:]
    
    try:
        if main_canvas.winfo_exists():
            main_canvas.itemconfig(main_image_id, image=photo_main)
    except Exception as e:
        print(f"Main canvas error: {e}")
        

def carla_front_camera_callback(image):
    global frame_count, photo_refs_list
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = array[:, :, :3]  # BGR
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.array(frame, copy=True)

    show_image(frame)

def on_key(event):
    global current_control, carla_vehicle
    key = event.keysym.lower()
    if key == 'w':
        current_control.throttle = min(current_control.throttle + 0.05, 1.0)
    elif key == 's':
        current_control.brake = min(current_control.brake + 0.1, 1.0)
    elif key == 'a':
        current_control.steer = max(current_control.steer - 0.05, -1.0)
    elif key == 'd':
        current_control.steer = min(current_control.steer + 0.05, 1.0)
    elif key == 'space':
        current_control = carla.VehicleControl()  # Reset controls
    if carla_vehicle:
        carla_vehicle.apply_control(current_control)

def on_key_release(event):
    global current_control, carla_vehicle
    key = event.keysym.lower()
    if key in ['a', 'd']:
        current_control.steer = 0.0
    if key in ['s']:
        current_control.brake = 0.0
    if carla_vehicle:
        carla_vehicle.apply_control(current_control)

vehicle_window = None
lane_hough_window = None
sign_window = None
light_window = None

root = tk.Tk()
root.title("Carla Self-Driving Car Simulation")

control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

lane_hough_window = tk.Toplevel(root)
lane_hough_window.title("Lane Detection using Hough Transform")
lane_hough_window.geometry(f"{window_width}x{window_height}+{window_width}+0")

lane_ml_window = tk.Toplevel(root)
lane_ml_window.title("Lane Detection using Machine Learning")
lane_ml_window.geometry(f"{window_width}x{window_height}+{2*window_width}+0")

sign_window = tk.Toplevel(root)
sign_window.title("Sign Detection")
sign_window.geometry(f"{window_width}x{window_height}+0+{window_height}")

light_window = tk.Toplevel(root)
light_window.title("Traffic Light Detection")
light_window.geometry(f"{window_width}x{window_height}+{window_width}+{window_height}")

vehicle_window = tk.Toplevel(root)
vehicle_window.title("Vehicle & Pedestrian Detection")
vehicle_window.geometry(f"{window_width}x{window_height}+{2*window_width}+{window_height}")

main_canvas = tk.Canvas(root, width=window_width, height=window_height)
main_canvas.pack(fill="both", expand=True)

lane_hough_canvas = tk.Canvas(lane_hough_window, width=window_width, height=window_height)
lane_hough_canvas.pack(fill="both", expand=True)

lane_ml_canvas = tk.Canvas(lane_ml_window, width=window_width, height=window_height)
lane_ml_canvas.pack(fill="both", expand=True)

sign_canvas = tk.Canvas(sign_window, width=window_width, height=window_height)
sign_canvas.pack(fill="both", expand=True)

light_canvas = tk.Canvas(light_window, width=window_width, height=window_height)
light_canvas.pack(fill="both", expand=True)

vehicle_canvas = tk.Canvas(vehicle_window, width=window_width, height=window_height)
vehicle_canvas.pack(fill="both", expand=True)

main_image_id = main_canvas.create_image(0, 0, anchor=tk.NW)
lane_hough_image_id = lane_hough_canvas.create_image(0, 0, anchor=tk.NW)
lane_ml_image_id = lane_ml_canvas.create_image(0, 0, anchor=tk.NW)
sign_image_id = sign_canvas.create_image(0, 0, anchor=tk.NW)
light_image_id = light_canvas.create_image(0, 0, anchor=tk.NW)
vehicle_image_id = vehicle_canvas.create_image(0, 0, anchor=tk.NW)

root.bind("<KeyPress>", on_key)
root.bind("<KeyRelease>", on_key_release)

@register_keras_serializable()
def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

def load_all_models():
    try:
        print("Loading models, please wait...")

        # Load vehicle and pedestrian detection model
        vehicle_model_path = os.path.join("model", "vehicle_pedestrian_detection.pt")
        if not os.path.exists(vehicle_model_path):
            print(f"ERROR: Model file not found: {vehicle_model_path}")
            return False
        MODELS['vehicle'] = YOLO(os.path.join("model", "vehicle_pedestrian_detection.pt"))
        print("Vehicle detection model loaded")
        
        # Load sign detection model
        MODELS['sign_detect'] = YOLO(os.path.join("model", "sign_detection.pt"))
        print("Sign detection model loaded")
        
        # Load sign classification model
        MODELS['sign_classify'] = tf.keras.models.load_model(os.path.join("model", "traffic_sign_classification.h5"), compile=False, custom_objects={"random_brightness": random_brightness})
        print("Sign classification model loaded")
        
        # Load traffic light detection model
        MODELS['traffic_light'] = YOLO(os.path.join("model", "traffic_light_detect_class.pt"))
        print("Traffic light model loaded")
        
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_image(frame):
    global main_photo, lane_ml_photo, sign_photo, light_detect_class_photo, vehicle_ped_photo
    global frame_count, last_lane_update, last_sign_update, last_light_update, last_vehicle_update

    frame_count += 1
    
    rgb_image = frame
    
    if frame_count % 5 == 0 and frame_count - last_lane_update >= update_interval:
        try:
            lane_results_hough = detect_lanes_hough(rgb_image)
            lane_image_hough = rgb_image.copy()
            
            if lane_results_hough is not None and isinstance(lane_results_hough, (list, tuple, np.ndarray)) and len(lane_results_hough) > 0:
                print(f"Drawing {len(lane_results_hough)} lane lines")
                for line in lane_results_hough:
                    if isinstance(line, tuple) and len(line) == 2:
                        cv.line(lane_image_hough, line[0], line[1], (255, 0, 0), 2)
            else:
                cv.putText(lane_image_hough, "No lanes detected", (50, 100), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("No lane lines detected")
                
                h, w = lane_image_hough.shape[:2]
                for x in range(0, w, 50):
                    cv.line(lane_image_hough, (x, 0), (x, h), (200, 200, 200), 1)
                for y in range(0, h, 50):
                    cv.line(lane_image_hough, (0, y), (w, y), (200, 200, 200), 1)
            
            cv.putText(lane_image_hough, f"Frame: {frame_count}", (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            lane_hough_photo = numpy_to_tkinter(lane_image_hough, "lane_hough")
            lane_hough_canvas.itemconfig(lane_hough_image_id, image=lane_hough_photo)
            last_lane_update = frame_count
        except Exception as e:
            print(f"Error in lane_detection_hough: {e}")

    # elif frame_count % 5 == 1:
    #     try:
    #         lane_results_ml = detect_lanes_ml(rgb_image)
    #         lane_image_ml = rgb_image.copy()
            
    #         if lane_results_ml is not None and isinstance(lane_results_ml, (list, tuple, np.ndarray)) and len(lane_results_ml) > 0:
    #             for line in lane_results_ml:
    #                 if isinstance(line, tuple) and len(line) == 2:
    #                     cv.line(lane_image_ml, line[0], line[1], (0, 255, 0), 2)
            
    #         cv.putText(lane_image_ml, f"Frame: {frame_count}", (10, 30), 
    #                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
    #         lane_ml_photo = numpy_to_tkinter(lane_image_ml)
    #         lane_ml_canvas.itemconfig(lane_ml_image_id, image=lane_ml_photo)
    #     except Exception as e:
    #         print(f"Error in lane_detection_ml: {e}")
    
    elif frame_count % 5 == 2 and frame_count - last_sign_update >= update_interval:
        try:
            sign_results = detect_classify_signs(rgb_image)
            sign_image = rgb_image.copy()
            
            if sign_results is not None and len(sign_results) > 0:
                for sign in sign_results:
                    if isinstance(sign, dict) and 'bbox' in sign:
                        x1, y1, x2, y2 = sign['bbox']
                        
                        if sign.get('verified', False):
                            color = (0, 255, 0)  # Green for verified signs (both models)
                        elif 'classification_confidence' in sign and sign['classification_confidence'] > 0.6:
                            color = (0, 200, 0)  # Lighter green for high confidence
                        elif 'classification_confidence' in sign and sign['classification_confidence'] > 0.6:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 160, 255)  # Orange for lower confidence
                        
                        thickness = 3 if sign.get('verified', False) else 2
                        cv.rectangle(sign_image, (x1, y1), (x2, y2), color, thickness)
                        
                        if 'classification' in sign:
                            class_text = sign['classification']
                            if len(class_text) > 20:
                                class_text = class_text[:20] + "..."
                            
                            confidence = sign['classification_confidence']
                            
                            source_tag = ""
                            if 'source' in sign:
                                if sign['source'] == 'sign_model':
                                    source_tag = " (SM)"
                                elif sign['source'] == 'vehicle_model':
                                    source_tag = " (VM)"
                            
                            label = f"{class_text}: {confidence:.2f}{source_tag}"
                            
                            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv.rectangle(sign_image, (x1, y1 - text_size[1] - 5), 
                                        (x1 + text_size[0], y1), color, -1)
                            
                            cv.putText(sign_image, label, (x1, y1 - 5), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                      
            sign_photo = numpy_to_tkinter(sign_image, "sign")
            sign_canvas.itemconfig(sign_image_id, image=sign_photo)
            last_sign_update = frame_count
        except Exception as e:
            print(f"Error in sign_detection: {e}")
    
    elif frame_count % 5 == 3 and frame_count - last_light_update >= update_interval:
        try:
            thread_image = rgb_image.copy()

            result_queue = queue.Queue()
            
            def traffic_light_worker():
                try:
                    result = detect_class_traffic_lights(thread_image)
                    result_queue.put(("success", result))
                except Exception as e:
                    print(f"Error in traffic light detection thread: {e}")
                    result_queue.put(("error", str(e)))
            
            worker_thread = threading.Thread(target=traffic_light_worker)
            worker_thread.daemon = True
            worker_thread.start()
            
            worker_thread.join(timeout=5.0)
            
            if worker_thread.is_alive():
                print("Traffic light detection timed out!")
                timeout_img = thread_image.copy()
                cv.putText(timeout_img, "Detection Timeout!", (50, 50), 
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                light_detect_class_photo = numpy_to_tkinter(timeout_img, "light")
                light_canvas.itemconfig(light_image_id, image=light_detect_class_photo)
                last_light_update = frame_count
                return
            
            if not result_queue.empty():
                status, light_results_detect = result_queue.get()
                
                if status == "success":
                    light_detect_image = thread_image.copy()
            
                    if light_results_detect is not None and isinstance(light_results_detect, (list, tuple, np.ndarray)) and len(light_results_detect) > 0:
                        for light in light_results_detect:
                            if isinstance(light, dict) and 'bbox' in light:
                                x1, y1, x2, y2 = light['bbox']

                                if 'class' in light:
                                    if light['class'] == 'red':
                                        color = (0, 0, 255)
                                    elif light['class'] == 'yellow':
                                        color = (0, 255, 255)
                                    elif light['class'] == 'green':
                                        color = (0, 255, 0)
                                    else:
                                        color = (255, 255, 255)
                                
                                thickness = 3 if light.get('agreement', False) else 1

                                cv.rectangle(light_detect_image, (x1, y1), (x2, y2), color, thickness)
                            
                            confidence_text = f"{light.get('confidence', 0):.2f}"
                            source_tag = ""
                            verified_tag = ""
                            
                            if 'source' in light:
                                if light['source'] == 'traffic_light_model':
                                    source_tag = " (TL)"
                                elif light['source'] == 'vehicle_model':
                                    source_tag = " (VM)"
                            
                            if light.get('verified', False):
                                verified_tag = " ✓"
                            
                            label = f"{light['class']}: {confidence_text}{source_tag}{verified_tag}"
                            
                            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv.rectangle(light_detect_image, 
                                        (x1, y1 - text_size[1] - 5), 
                                        (x1 + text_size[0], y1), 
                                        color, -1)
                            cv.putText(light_detect_image, label, 
                                    (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            if 'probabilities' in light:
                                bar_width = 60
                                bar_height = 15
                                bar_x = x2 + 5
                                bar_y = y1
                                
                                cv.rectangle(light_detect_image, 
                                            (bar_x, bar_y), 
                                            (bar_x + bar_width, bar_y + bar_height * 3), 
                                            (50, 50, 50), -1)
                                
                                red_height = int(bar_height * light['probabilities']['red'])
                                cv.rectangle(light_detect_image,
                                            (bar_x, bar_y),
                                            (bar_x + bar_width, bar_y + red_height),
                                            (0, 0, 255), -1)
                                
                                yellow_height = int(bar_height * light['probabilities']['yellow'])
                                cv.rectangle(light_detect_image,
                                            (bar_x, bar_y + bar_height),
                                            (bar_x + bar_width, bar_y + bar_height + yellow_height),
                                            (0, 255, 255), -1)
                                
                                green_height = int(bar_height * light['probabilities']['green'])
                                cv.rectangle(light_detect_image,
                                            (bar_x, bar_y + 2*bar_height),
                                            (bar_x + bar_width, bar_y + 2*bar_height + green_height),
                                            (0, 255, 0), -1)
            
                cv.putText(light_detect_image, f"Frame: {frame_count}", (10, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                light_detect_class_photo = numpy_to_tkinter(light_detect_image, "light")
                light_canvas.itemconfig(light_image_id, image=light_detect_class_photo)
                last_light_update = frame_count
                
        except Exception as e:
            print(f"Error in traffic light detection: {e}")

    elif frame_count % 5 == 4 and frame_count - last_vehicle_update >= update_interval:
        try:
            vehicle_ped_results = detect_vehicles_pedestrians(rgb_image)
            vehicle_ped_image = rgb_image.copy()

            if vehicle_ped_results and len(vehicle_ped_results) > 0:
                for detection in vehicle_ped_results:
                    if 'bbox' in detection and 'class' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        class_name = detection['class'].lower()
                        conf = detection['confidence']
                        
                        if class_name == 'pedestrian':
                            color = (0, 255, 0)
                        elif class_name in ['car', 'truck', 'bus']:
                            color = (0, 0, 255)
                        elif class_name in ['bicycle', 'motorcycle']:
                            color = (255, 165, 0)
                        else:
                            color = (255, 255, 255)

                        cv.rectangle(vehicle_ped_image, (x1, y1), (x2, y2), color, 2)

                        label = f"{class_name}: {conf:.2f}"
                        cv.putText(vehicle_ped_image, label, 
                                  (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv.putText(vehicle_ped_image, f"Frame: {frame_count}", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            vehicle_ped_photo = numpy_to_tkinter(vehicle_ped_image, "vehicle")
            vehicle_canvas.itemconfig(vehicle_image_id, image=vehicle_ped_photo)
            last_vehicle_update = frame_count
        except Exception as e:
            print(f"Error in vehicle_pedestrian_detection: {e}")


def detect_lanes_hough(frames):
    from lane_detection_hough import lane_detection

    print(f"Lane detection input shape: {frames.shape}, dtype: {frames.dtype}")

    frames_bgr = cv.cvtColor(frames, cv.COLOR_RGB2BGR)

    lane_hough_results = lane_detection(frames_bgr)
    print(f"Got lane results: {lane_hough_results[:2] if lane_hough_results else 'None'}")

    return lane_hough_results

# def detect_lanes_ml(frames):
#     from lane_detection_model import predict_lane

#     lane_ml_results = predict_lane(frames)

#     return lane_ml_results

def detect_class_traffic_lights(frames):
    from traffic_light_detect_class import combined_traffic_light_detection

    light_detect_results = combined_traffic_light_detection(frames)
    print(f"Got traffic light results: {light_detect_results[:2] if light_detect_results else 'None'}")

    return light_detect_results

def detect_classify_signs(frames):
    from sign_detection_classification import combined_sign_detection_classification

    sign_results = combined_sign_detection_classification(frames)
    print(f"Got sign results: {sign_results[:2] if sign_results else 'None'}")

    return sign_results

def detect_vehicles_pedestrians(frames):
    from vehicle_pedestrian_detection import detect_vehicles_pedestrians

    print(f"Vehicle detection input shape: {frames.shape}, dtype: {frames.dtype}")

    vehicle_ped_results = detect_vehicles_pedestrians(frames)
    print(f"Got vehicle results: {vehicle_ped_results[:2] if vehicle_ped_results else 'None'}")

    return vehicle_ped_results

def on_closing():
    global carla_camera, carla_front_camera, carla_vehicle
    print("Cleaning up CARLA resources...")
    try:
        if carla_camera is not None:
            carla_camera.stop()
        if carla_front_camera is not None:
            carla_front_camera.stop()
        if carla_vehicle is not None:
            carla_vehicle.destroy()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

def main():
    if not load_all_models():
        print("Failed to load models. Exiting...")
        return

    carla_simulation_setup()

    root.mainloop()

if __name__ == "__main__":
    main()