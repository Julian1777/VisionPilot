from beamng_sim.vehicle_obstacle.vehicle_obstacle_detection import detect_vehicles_pedestrians
import cv2

def process_frame(frame, draw_detections=True):
    try:
        detections = detect_vehicles_pedestrians(frame)
        result_img = frame
        
        if draw_detections:
            result_img = frame.copy()
            for det in detections:
                bbox = det['bbox']
                label = f"{det['class']} ({det['confidence']:.2f})"
                cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(result_img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections, result_img
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []
