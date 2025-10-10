import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import cv2
import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

from beamng_sim.lane_detection.main import process_frame as cv_lane_detection
from beamng_sim.utils.pid_controller import PIDController

# python video_lane_detection.py nl_highway.mp4 --output processed.mp4

IMG_SIZE_UNET = (256, 320)
UNET_MODEL_PATH = "./lane-detection-cnn/unet/lane_detection_unet.h5"

class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self._dtype})
        return config

def load_unet_model():
    try:
        if os.path.exists(UNET_MODEL_PATH):
            print(f"Loading UNet model from {UNET_MODEL_PATH}")
            model = load_model(UNET_MODEL_PATH, compile=False)
            return model
        else:
            print(f"UNet model not found at {UNET_MODEL_PATH}")
            print("UNet detection will be skipped")
            return None
    except Exception as e:
        print(f"Error loading UNet model: {e}")
        print("UNet detection will be skipped")
        return None

def preprocess_for_unet(image):
    resized = cv2.resize(image, (IMG_SIZE_UNET[1], IMG_SIZE_UNET[0]))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def unet_lane_detection(image, model):
    if model is None:
        return np.zeros_like(image[:, :, 0])
    
    try:
        input_tensor = preprocess_for_unet(image)
        
        prediction = model.predict(input_tensor, verbose=0)
        
        pred_mask = (prediction[0].squeeze() >= 0.5).astype(np.uint8) * 255
        
        original_size = (image.shape[1], image.shape[0])
        pred_mask_resized = cv2.resize(pred_mask, original_size)
        
        return pred_mask_resized
        
    except Exception as e:
        print(f"Error in UNet prediction: {e}")
        return np.zeros_like(image[:, :, 0])

def cv_lane_detection_simple(image, speed_kph=30, previous_steering=0):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result, metrics = cv_lane_detection(
            image_rgb, 
            speed=speed_kph, 
            previous_steering=previous_steering, 
            debug_display=False
        )
        
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result_bgr, metrics
        
    except Exception as e:
        print(f"Error in CV lane detection: {e}")
        return image, {'deviation': 0, 'lane_center': 0, 'vehicle_center': 0}

def process_video(video_path, output_path=None):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    unet_model = load_unet_model()
    
    previous_steering = 0.0
    speed_kph = 30.0
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_width = width * 3
        writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            original = frame.copy()
            
            cv_result, metrics = cv_lane_detection_simple(frame, speed_kph, previous_steering)
            
            unet_mask = unet_lane_detection(frame, unet_model)
            
            unet_display = cv2.applyColorMap(unet_mask, cv2.COLORMAP_JET)
            
            cv2.putText(original, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(cv_result, 'CV Lane Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(unet_display, 'UNet Lane Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            deviation = metrics.get('deviation', 0)
            lane_center = metrics.get('lane_center', 0)
            vehicle_center = metrics.get('vehicle_center', 0)
            
            cv2.putText(cv_result, f'Deviation: {deviation:.2f}m', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(cv_result, f'Lane Center: {lane_center:.1f}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(cv_result, f'Vehicle Center: {vehicle_center:.1f}', (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Original', original)
            cv2.imshow('CV Lane Detection', cv_result)
            cv2.imshow('UNet Lane Detection', unet_display)
            
            if writer:
                combined = np.hstack([original, cv_result, unet_display])
                writer.write(combined)
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - {fps_actual:.1f} FPS")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping playback...")
                break
            elif key == ord(' '):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed:.1f} seconds")
        print(f"Average FPS: {frame_count/elapsed:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Video Lane Detection with CV and UNet')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to output video file (optional)')
    parser.add_argument('--unet-model', help=f'Path to UNet model (default: {UNET_MODEL_PATH})')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    if args.unet_model:
        global UNET_MODEL_PATH
        UNET_MODEL_PATH = args.unet_model
    
    print(f"Processing video: {args.video_path}")
    if args.output:
        print(f"Output will be saved to: {args.output}")
    
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press spacebar to pause/resume")
    print()
    
    process_video(args.video_path, args.output)

if __name__ == "__main__":
    main()