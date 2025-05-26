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

cap = None
playing = False
delay = 30

main_photo = None
lane_hough_photo = None
lane_ml_photo = None
sign_photo = None
light_detect_photo = None
light_class_photo = None
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

    frame = np.array(frame, copy=True)

    label = f"Frame count: {frame_count}"
    cv.putText(frame, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    
    frame = np.array(frame, copy=True)
    
    label = f"Frame count: {frame_count}"
    cv.putText(frame, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
    
    frame_lane_hough = frame.copy()
    frame_lane_ml = frame.copy()
    frame_sign = frame.copy()
    frame_light = frame.copy()
    frame_vehicle = frame.copy()
    
    photo_lane_hough = numpy_to_tkinter(frame_lane_hough, window_id="lane_hough")
    photo_lane_ml = numpy_to_tkinter(frame_lane_ml, window_id="lane_ml")
    photo_sign = numpy_to_tkinter(frame_sign, window_id="sign")
    photo_light = numpy_to_tkinter(frame_light, window_id="light")
    photo_vehicle = numpy_to_tkinter(frame_vehicle, window_id="vehicle")
    
    photo_refs_list.extend([
        photo_lane_hough, photo_lane_ml, 
        photo_sign, photo_light, photo_vehicle
    ])
    if len(photo_refs_list) > 20:
        photo_refs_list = photo_refs_list[-20:]
    
    try:
        if lane_hough_canvas.winfo_exists() and lane_hough_image_id:
            lane_hough_canvas.itemconfig(lane_hough_image_id, image=photo_lane_hough)
        
        if lane_ml_canvas.winfo_exists() and lane_ml_image_id:
            lane_ml_canvas.itemconfig(lane_ml_image_id, image=photo_lane_ml)
        
        if sign_canvas.winfo_exists() and sign_image_id:
            sign_canvas.itemconfig(sign_image_id, image=photo_sign)
        
        if light_canvas.winfo_exists() and light_image_id:
            light_canvas.itemconfig(light_image_id, image=photo_light)
        
        if vehicle_canvas.winfo_exists() and vehicle_image_id:
            vehicle_canvas.itemconfig(vehicle_image_id, image=photo_vehicle)
    except Exception as e:
        print(f"Canvas update error: {e}")

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

def main():
    carla_simulation_setup()
    root.mainloop()

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

if __name__ == "__main__":
    main()