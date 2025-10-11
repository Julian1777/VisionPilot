import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math

IMG_SIZE = (256, 320)
MODEL_PATH = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\models\\lane_detection_unet.h5"
VAL_IMG_DIR = "val_img"

model = tf.keras.models.load_model(MODEL_PATH)

img_files = [os.path.join(VAL_IMG_DIR, f) for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
img_files.sort()

images = []
pred_masks = []

for img_path in img_files:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
    images.append(img_resized)
    input_tensor = np.expand_dims(img_resized, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]
    print(f"{os.path.basename(img_path)}: pred min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
    pred_mask = (pred.squeeze() >= 0.5).astype(np.uint8)
    pred_masks.append(pred_mask)

num_imgs = len(images)
rows = 3
cols = math.ceil(num_imgs / rows)
fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 4 * rows), squeeze=False)
for idx in range(num_imgs):
    row = idx % rows
    col = (idx // rows) * 2
    axes[row, col].imshow(images[idx] / 255.)
    axes[row, col].set_title(f"Image {idx+1}", fontsize=14)
    axes[row, col].axis('off')
    axes[row, col+1].imshow(pred_masks[idx], cmap='gray')
    axes[row, col+1].set_title(f"Predicted Mask {idx+1}", fontsize=14)
    axes[row, col+1].axis('off')
# Hide any unused subplots
for r in range(rows):
    for c in range(cols * 2):
        if r + (c//2)*rows >= num_imgs and c % 2 == 0:
            axes[r, c].axis('off')
            axes[r, c+1].axis('off')
plt.subplots_adjust(wspace=0.05, hspace=0.15)
plt.show()