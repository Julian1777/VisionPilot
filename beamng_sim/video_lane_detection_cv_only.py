import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import cv2
import numpy as np
import argparse
import time

# Import lane detection modules from beamng_sim
from beamng_sim.lane_detection.main import process_frame as cv_lane_detection
from beamng_sim.utils.pid_controller import PIDController

def cv_lane_detection_simple(image, speed_kph=30, previous_steering=0):
    """Run CV-based lane detection using the BeamNG functions"""
    try:
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process frame using the same function as BeamNG
        result, metrics = cv_lane_detection(
            image_rgb, 
            speed=speed_kph, 
            previous_steering=previous_steering, 
            debug_display=False
        )
        
        # Convert back to BGR for display
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result_bgr, metrics
        
    except Exception as e:
        print(f"Error in CV lane detection: {e}")
        return image, {'deviation': 0, 'lane_center': 0, 'vehicle_center': 0}

def process_video_cv_only(video_path, output_path=None):
    """Process video with CV lane detection only"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize variables for CV lane detection
    previous_steering = 0.0
    speed_kph = 30.0  # Simulated speed
    
    # Video writer for output (if specified)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_width = width * 2  # Two windows side by side
        writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Original frame
            original = frame.copy()
            
            # CV lane detection
            cv_result, metrics = cv_lane_detection_simple(frame, speed_kph, previous_steering)
            
            # Add text overlays
            cv2.putText(original, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(cv_result, 'CV Lane Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add metrics to CV result
            deviation = metrics.get('deviation', 0)
            lane_center = metrics.get('lane_center', 0)
            vehicle_center = metrics.get('vehicle_center', 0)
            
            cv2.putText(cv_result, f'Deviation: {deviation:.2f}m', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(cv_result, f'Lane Center: {lane_center:.1f}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(cv_result, f'Vehicle Center: {vehicle_center:.1f}', (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display windows
            cv2.imshow('Original', original)
            cv2.imshow('CV Lane Detection', cv_result)
            
            # Save combined frame if output specified
            if writer:
                combined = np.hstack([original, cv_result])
                writer.write(combined)
            
            # Print progress
            if frame_count % 30 == 0:  # Every second at 30fps
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - {fps_actual:.1f} FPS")
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping playback...")
                break
            elif key == ord(' '):  # Spacebar to pause
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed:.1f} seconds")
        print(f"Average FPS: {frame_count/elapsed:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Video Lane Detection with CV')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to output video file (optional)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    print(f"Processing video: {args.video_path}")
    if args.output:
        print(f"Output will be saved to: {args.output}")
    
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press spacebar to pause/resume")
    print()
    
    process_video_cv_only(args.video_path, args.output)

if __name__ == "__main__":
    main()