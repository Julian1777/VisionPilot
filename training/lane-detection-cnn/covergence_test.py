import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config.config import DATASETS_DIR

IMAGES_DIR = DATASETS_DIR / "lane-detection" / "processed" / "culane" / "images"
MASKS_DIR = DATASETS_DIR / "lane-detection" / "processed" / "culane" / "masks"

def check_lane_coverage(num_samples=5):
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith('.png')])
    
    indices = np.random.choice(len(image_files), min(num_samples, len(image_files)), replace=False)
    
    total_coverage = 0
    
    plt.figure(figsize=(12, 4*num_samples))
    for i, idx in enumerate(indices):
        img_path = os.path.join(IMAGES_DIR, image_files[idx])
        mask_path = os.path.join(MASKS_DIR, mask_files[idx])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, 0)
        
        coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        total_coverage += coverage
        
        plt.subplot(num_samples, 2, i*2+1)
        plt.imshow(img)
        plt.title(f"Image: {image_files[idx]}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, i*2+2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: Coverage = {coverage*100:.2f}%")
        plt.axis('off')
    
    avg_coverage = total_coverage / num_samples
    print(f"Average lane coverage: {avg_coverage*100:.2f}%")
    print(f"This means approximately {avg_coverage*100:.2f}% of pixels are lane markings")
    
    plt.tight_layout()
    plt.show()
    
    return avg_coverage

check_lane_coverage(5)