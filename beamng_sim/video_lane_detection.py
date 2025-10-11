import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

VIDEO_PATH = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\beamng_sim\\nl_highway_cut.mp4"
UNET_MODEL_PATH = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\models\\lane_detection_unet.h5"
IMG_SIZE_UNET = (256, 320)
FRAME_SKIP = 5
DISPLAY_WIDTH = 640

def preprocess_for_unet(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (IMG_SIZE_UNET[1], IMG_SIZE_UNET[0]))
    if resized.shape != (256, 320, 3):
        print(f"Warning: input shape is {resized.shape}, expected (256, 320, 3)")
    normalized = resized.astype(np.float32)
    return np.expand_dims(normalized, axis=0)

def unet_mask(image, model):
    input_tensor = preprocess_for_unet(image)
    pred = model.predict(input_tensor, verbose=0)
    pred_map = pred[0].squeeze()
    print(f"UNet pred stats: min={pred_map.min():.4f}, max={pred_map.max():.4f}, mean={pred_map.mean():.4f}")
    raw_pred_vis = (pred_map * 255).astype(np.uint8)
    raw_pred_vis = cv2.resize(raw_pred_vis, (image.shape[1], image.shape[0]))
    disp_raw_pred = resize_for_display(raw_pred_vis)
    cv2.imshow('UNet Raw Output', disp_raw_pred)
    mask = (pred_map >= 0.2).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


    # POST PROCESSING

    # NOISE REMOVAL
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # FILL GAPS
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # KEEP LARGEST COMPONENT
    min_area = 2000
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    mask_filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask_filtered[labels == label] = 1
    mask = mask_filtered

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask

def resize_for_display(img, width=DISPLAY_WIDTH):
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h))

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return
    if not os.path.exists(UNET_MODEL_PATH):
        print(f"Model not found: {UNET_MODEL_PATH}")
        return
    print("Loading model...")
    model = load_model(UNET_MODEL_PATH, compile=False)
    print("Model loaded.")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            mask = unet_mask(frame, model)
            overlay = frame.copy()
            alpha = 0.4
            mask_bool = mask > 0
            color = [0, 255, 0]
            for c in range(3):
                overlay[..., c] = np.where(
                    mask_bool,
                    (1 - alpha) * overlay[..., c] + alpha * color[c],
                    overlay[..., c]
                )
            overlay = overlay.astype(np.uint8)
            disp_frame = resize_for_display(frame)
            disp_overlay = resize_for_display(overlay)
            disp_frame = (disp_frame.astype(np.float32) / 255.0)
            disp_overlay = (disp_overlay.astype(np.float32) / 255.0)
            cv2.imshow('Original', disp_frame)
            cv2.imshow('UNet Overlay', disp_overlay)
        else:
            disp_frame = resize_for_display(frame)
            cv2.imshow('Original', disp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
