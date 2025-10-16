import cv2
import numpy as np

def apply_roi_mask(mask, src_points):
    """
    Apply a region of interest (ROI) mask defined by source points
    
    Args:
        mask (np.array): Binary mask to apply ROI to
        src_points (np.array): Array of source points defining the ROI polygon
    
    Returns:
        np.array: Masked binary image
    """
    roi_mask = np.zeros_like(mask)
    polygon = np.array([src_points], dtype=np.int32)
    cv2.fillPoly(roi_mask, polygon, 1)
    
    return mask * roi_mask

def remove_noise(mask, kernel_size=(5, 5)):
    """
    Remove noise using morphological operations
    
    Args:
        mask (np.array): Binary mask to clean
        kernel_size (tuple): Size of the kernel for morphological operations
    
    Returns:
        np.array: Cleaned binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def fill_gaps(mask, kernel_size=(5, 5)):
    """
    Fill gaps in the mask using morphological operations
    
    Args:
        mask (np.array): Binary mask to fill gaps in
        kernel_size (tuple): Size of the kernel for morphological operations
    
    Returns:
        np.array: Binary mask with filled gaps
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def filter_components(mask, min_area=2000):
    """
    Keep only components larger than a minimum area
    
    Args:
        mask (np.array): Binary mask to filter
        min_area (int): Minimum area threshold for components to keep
    
    Returns:
        np.array: Filtered binary mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    mask_filtered = np.zeros_like(mask)
    
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask_filtered[labels == label] = 1
            
    return mask_filtered

def thin_mask(mask, kernel_size=(5, 5), iterations=1):
    """
    Thin the mask using erosion
    
    Args:
        mask (np.array): Binary mask to thin
        kernel_size (tuple): Size of the kernel for erosion
        iterations (int): Number of erosion iterations
    
    Returns:
        np.array: Thinned binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    
    return mask

def process_unet_mask(mask, src_points, min_area=2000):
    """
    Apply all post-processing steps to the UNet mask
    
    Args:
        mask (np.array): Binary mask from UNet prediction
        src_points (np.array): Array of source points defining the ROI polygon
        min_area (int): Minimum area threshold for components to keep
    
    Returns:
        np.array: Post-processed binary mask
    """
    # Apply ROI mask
    mask = apply_roi_mask(mask, src_points)
    
    # Remove noise and fill gaps
    mask = remove_noise(mask)
    mask = fill_gaps(mask)
    
    # Filter components by area
    mask = filter_components(mask, min_area=min_area)
    
    # Thin the mask
    mask = thin_mask(mask)
    
    return mask

def run_unet_on_frame(img, model):
    """
    Run UNet model prediction on a frame
    
    Args:
        img (np.array): Input image
        model: UNet model
    
    Returns:
        np.array: Binary prediction mask
    """
    IMG_SIZE = (256, 320)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
    img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
    input_tensor = np.expand_dims(img_resized, axis=0)
    
    print(f"UNet input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    pred = model.predict(input_tensor, verbose=0)[0]
    
    print(f"UNet raw prediction: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
    
    pred_mask = (pred.squeeze() >= 0.2).astype(np.uint8)
    
    return pred_mask
