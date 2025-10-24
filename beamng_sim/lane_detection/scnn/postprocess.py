"""
SCNN Postprocessing - MINIMAL Processing
Returns bare mask and confidence - trusts the model's end-to-end training
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def run_scnn_on_frame(img, model, device='cpu', debug_display=False):
    """
    Run SCNN model prediction on a frame - MINIMAL POST-PROCESSING
    
    Args:
        img (np.array): Input BGR image (BeamNG format)
        model: SCNN model
        device (str or torch.device): Device to run inference on
        debug_display (bool): Whether to show debug windows
    
    Returns:
        tuple: (seg_mask, exist_pred, confidence)
            - seg_mask (np.array): Binary mask (H, W) - all lanes combined
            - exist_pred (np.array): Lane existence predictions (6,)
            - confidence (float): Overall confidence score
    """
    # SCNN expects (800, 288) input size
    SCNN_INPUT_SIZE = (800, 288)  # (width, height)
    
    # Store original size for later
    original_h, original_w = img.shape[:2]
    
    # DEBUG: Input image
    if debug_display:
        print(f"[SCNN] Input image shape: {img.shape}")
    
    # Convert BGR to RGB (SCNN was trained on RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to SCNN input size
    img_resized = cv2.resize(img_rgb, SCNN_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    
    if debug_display:
        print(f"[SCNN] Resized to: {img_resized.shape}")
    
    # Convert to tensor: (H, W, C) -> (C, H, W)
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    input_tensor = input_tensor.unsqueeze(0)
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    if debug_display:
        print(f"[SCNN] Input tensor shape: {input_tensor.shape}")
        print(f"[SCNN] Tensor min/max: {input_tensor.min():.2f} / {input_tensor.max():.2f}")
    
    # Run inference
    with torch.no_grad():
        seg_pred, exist_pred, _, _, _ = model(input_tensor)
    
    if debug_display:
        print(f"[SCNN] Segmentation output shape: {seg_pred.shape}")
        print(f"[SCNN] Existence output shape: {exist_pred.shape}")
    
    # Process segmentation output
    # Apply softmax to get probabilities
    seg_pred_softmax = F.softmax(seg_pred, dim=1)
    
    # Get class predictions (argmax)
    seg_pred_class = torch.argmax(seg_pred_softmax, dim=1)
    
    # Convert to numpy
    seg_pred_np = seg_pred_class[0].cpu().numpy()  # Shape: (288, 800)
    seg_pred_softmax_np = seg_pred_softmax[0].cpu().numpy()  # Shape: (7, 288, 800)
    exist_pred_np = exist_pred[0].cpu().numpy()  # Shape: (6,)
    
    if debug_display:
        print(f"[SCNN] Segmentation class map shape: {seg_pred_np.shape}")
        unique_classes, counts = np.unique(seg_pred_np, return_counts=True)
        print(f"[SCNN] Predicted classes: {unique_classes}")
        print(f"[SCNN] Pixels per class: {counts}")
        print(f"[SCNN] Existence predictions: {exist_pred_np}")
        for i, prob in enumerate(exist_pred_np):
            status = "DETECTED" if prob > 0.5 else "NOT DETECTED"
            print(f"[SCNN]   Lane {i+1}: {prob:.4f} - {status}")
    
    # Create binary mask: all lanes (classes 1-6) vs background (class 0)
    binary_mask = (seg_pred_np > 0).astype(np.uint8)
    
    if debug_display:
        lane_pixels = binary_mask.sum()
        total_pixels = binary_mask.size
        print(f"[SCNN] Lane pixels: {lane_pixels} / {total_pixels} ({lane_pixels/total_pixels*100:.2f}%)")
    
    # Calculate confidence score based on:
    # 1. Existence predictions (how confident the model is that lanes exist)
    # 2. Segmentation quality (how confident the model is in its predictions)
    
    # Confidence from existence predictions
    exist_conf = np.mean(exist_pred_np[exist_pred_np > 0.5]) if np.any(exist_pred_np > 0.5) else 0.0
    
    # Confidence from segmentation (average max probability for lane pixels)
    lane_probs = seg_pred_softmax_np[1:, :, :]  # All lane channels (exclude background)
    lane_prob_max = np.max(lane_probs, axis=0)  # Max probability per pixel
    
    # Only consider pixels where lanes are predicted
    if binary_mask.sum() > 0:
        seg_conf = lane_prob_max[binary_mask > 0].mean()
    else:
        seg_conf = 0.0
    
    # Combined confidence (weighted average) - for simple compatibility
    simple_confidence = 0.6 * exist_conf + 0.4 * seg_conf
    
    # Store segmentation quality separately for advanced confidence calculation
    segmentation_quality = float(seg_conf)
    
    if debug_display:
        print(f"[SCNN] Existence confidence: {exist_conf:.4f}")
        print(f"[SCNN] Segmentation confidence: {seg_conf:.4f}")
        print(f"[SCNN] Simple confidence: {simple_confidence:.4f}")
        print(f"[SCNN] Segmentation quality (for advanced conf): {segmentation_quality:.4f}")
    
    # Show debug windows if requested
    if debug_display:
        # Raw mask at SCNN resolution
        debug_mask = (binary_mask * 255).astype(np.uint8)
        cv2.imshow('SCNN 1. Raw Prediction (800x288)', debug_mask)
        
        # Class visualization (colored)
        colored_mask = np.zeros((seg_pred_np.shape[0], seg_pred_np.shape[1], 3), dtype=np.uint8)
        lane_colors = [
            [255, 0, 0],    # Lane 1 - Red
            [0, 255, 0],    # Lane 2 - Green
            [0, 0, 255],    # Lane 3 - Blue
            [255, 255, 0],  # Lane 4 - Yellow
            [255, 0, 255],  # Lane 5 - Magenta
            [0, 255, 255],  # Lane 6 - Cyan
        ]
        for lane_idx in range(1, 7):
            mask = (seg_pred_np == lane_idx)
            if mask.any():
                colored_mask[mask] = lane_colors[lane_idx - 1]
        
        cv2.imshow('SCNN 2. Colored Lanes (800x288)', colored_mask)
        
        # Confidence heatmap
        heatmap = (lane_prob_max * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imshow('SCNN 3. Confidence Heatmap (800x288)', heatmap_colored)
    
    # Return the mask at SCNN resolution (800x288)
    # The calling function will resize it to match the original image
    # Returns: (binary_mask, exist_predictions, simple_confidence, segmentation_quality)
    return binary_mask, exist_pred_np, simple_confidence, segmentation_quality
