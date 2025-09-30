from beamng_sim.sign.detect_classify import combined_sign_detection_classification

def process_frame(img, draw_detections=True):
    try:
        detections = combined_sign_detection_classification(img)
        
        if not detections:
            detections = []
            
        result_img = img
        
        if draw_detections:
            import cv2
            result_img = img.copy()
            for det in detections:
                x1,y1,x2,y2 = det['bbox']
                label = f"{det['classification']} ({det['classification_confidence']:.2f})"
                cv2.rectangle(result_img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(result_img, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return detections, result_img
    except Exception as e:
        print(f"Error processing sign frame: {e}")
        return [], img
