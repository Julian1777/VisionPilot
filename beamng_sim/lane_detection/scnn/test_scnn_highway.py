"""
SCNN Lane Detection Test Script
Tests the SCNN model (scnn.pth) on a Dutch highway video
Outputs: original frame, detection overlay, and raw prediction
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print("=" * 80)
print("SCNN LANE DETECTION TEST SCRIPT")
print("=" * 80)

# ============================================================================
# DEBUG: Environment Information
# ============================================================================
print("\n[DEBUG] Environment Information:")
print(f"  - PyTorch version: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
print(f"  - Working directory: {os.getcwd()}")

# ============================================================================
# SCNN Model Definition (from training notebook)
# ============================================================================
print("\n[DEBUG] Loading SCNN model definition...")

class SCNN(nn.Module):
    """
    Spatial CNN for Lane Detection
    Based on: https://arxiv.org/abs/1712.06080
    
    Architecture:
    - VGG16-BN backbone with dilated convolutions
    - Spatial message passing (up-down, down-up, left-right, right-left)
    - Lane segmentation + existence prediction
    """
    def __init__(self, input_size, ms_ks=9, pretrained=False):
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        # Loss parameters (not needed for inference but kept for compatibility)
        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([self.scale_background, 1, 1, 1, 1, 1, 1])
        )
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        # Backbone feature extraction
        x = self.backbone(img)
        
        # SCNN layers
        x = self.layer1(x)
        
        # Message passing
        x = self.message_passing_forward(x)
        
        # Segmentation prediction
        x = self.layer2(x)
        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        # Existence prediction
        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)

        # Calculate loss if ground truth is provided
        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt.long())
            loss_exist = self.bce_loss(exist_pred, exist_gt.float())
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def message_passing_forward(self, x):
        """Apply spatial message passing in 4 directions"""
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Single direction message passing
        Args:
            x: input tensor (B, C, H, W)
            conv: convolution layer
            vertical: True for vertical (up-down), False for horizontal (left-right)
            reverse: False for forward, True for backward
        """
        nB, C, H, W = x.shape
        
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
            
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
            
        if reverse:
            out = out[::-1]
            
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        """Initialize network architecture"""
        input_w, input_h = input_size
        self.fc_input_feature = 7 * int(input_w/16) * int(input_h/16)
        
        # VGG16-BN backbone
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # Modify backbone with dilated convolutions
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, 
                stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), 
                dilation=2, 
                bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
            
        # Remove pooling layers
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # SCNN feature extraction layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Message passing layers (4 directions)
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', 
            nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', 
            nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))

        # Segmentation head
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 7, 1)  # 7 channels: background + 6 lanes
        )

        # Existence prediction head
        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 6 lane existence predictions
            nn.Sigmoid()
        )

    def weight_init(self):
        """Initialize weights for non-pretrained backbone"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()


# ============================================================================
# Configuration
# ============================================================================
print("\n[DEBUG] Configuration:")

# Paths
MODEL_PATH = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\models\\scnn.pth"
VIDEO_PATH = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\beamng_sim\\nl_highway_cut.mp4"
OUTPUT_DIR = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\beamng_sim\\lane_detection\\scnn\\scnn_test"

print(f"  - Model path: {MODEL_PATH}")
print(f"  - Video path: {VIDEO_PATH}")
print(f"  - Output directory: {OUTPUT_DIR}")

# Model parameters (from training)
INPUT_SIZE = (800, 288)  # (width, height)
print(f"  - Input size: {INPUT_SIZE} (width x height)")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  - Device: {device}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n[DEBUG] Created output directory: {OUTPUT_DIR}")

# ============================================================================
# Load Model
# ============================================================================
print("\n[DEBUG] Loading SCNN model...")

try:
    # Initialize model
    model = SCNN(input_size=INPUT_SIZE, pretrained=False)
    print(f"  - Model initialized with input size: {INPUT_SIZE}")
    
    # Load weights
    print(f"  - Loading weights from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Debug checkpoint contents
    print(f"  - Checkpoint keys: {checkpoint.keys()}")
    if 'epoch' in checkpoint:
        print(f"  - Trained for {checkpoint['epoch']} epochs")
    if 'best_val_loss' in checkpoint:
        print(f"  - Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load state dict
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()
    print(f"  - Model loaded successfully and set to evaluation mode")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    raise

# ============================================================================
# Load Video
# ============================================================================
print("\n[DEBUG] Loading video...")

try:
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  - Video FPS: {fps}")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Original size: {width}x{height}")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load video: {e}")
    raise

# ============================================================================
# Select Frame to Test (middle frame)
# ============================================================================
test_frame_idx = frame_count // 1.2
print(f"\n[DEBUG] Selecting test frame: {test_frame_idx} (middle of video)")

# Seek to frame
cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
ret, frame = cap.read()

if not ret:
    raise ValueError(f"Could not read frame {test_frame_idx}")

print(f"  - Frame shape (original): {frame.shape}")
cap.release()

# ============================================================================
# Preprocess Frame
# ============================================================================
print("\n[DEBUG] Preprocessing frame...")

# Store original for visualization
original_frame = frame.copy()
print(f"  - Original frame shape: {original_frame.shape}")

# Resize to model input size (800x288)
resized_frame = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
print(f"  - Resized to: {resized_frame.shape} (HxWxC)")

# Convert BGR to RGB
rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
print(f"  - Converted to RGB")

# Convert to tensor and normalize
# Shape: (H, W, C) -> (C, H, W)
input_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1)
print(f"  - Tensor shape after permute: {input_tensor.shape}")

# Add batch dimension: (C, H, W) -> (1, C, H, W)
input_tensor = input_tensor.unsqueeze(0)
print(f"  - Tensor shape with batch: {input_tensor.shape}")

# Move to device
input_tensor = input_tensor.to(device)
print(f"  - Moved to device: {device}")

# Print tensor statistics
print(f"  - Tensor min/max: {input_tensor.min():.2f} / {input_tensor.max():.2f}")
print(f"  - Tensor mean/std: {input_tensor.mean():.2f} / {input_tensor.std():.2f}")

# ============================================================================
# Run Inference
# ============================================================================
print("\n[DEBUG] Running inference...")

try:
    with torch.no_grad():
        # Forward pass
        seg_pred, exist_pred, _, _, _ = model(input_tensor)
        
        print(f"  - Segmentation prediction shape: {seg_pred.shape}")
        print(f"  - Segmentation prediction min/max: {seg_pred.min():.4f} / {seg_pred.max():.4f}")
        
        print(f"  - Existence prediction shape: {exist_pred.shape}")
        print(f"  - Existence prediction values: {exist_pred[0].cpu().numpy()}")
        
        # Apply softmax to segmentation
        seg_pred_softmax = F.softmax(seg_pred, dim=1)
        print(f"  - Applied softmax to segmentation")
        
        # Get class predictions (argmax)
        seg_pred_class = torch.argmax(seg_pred_softmax, dim=1)
        print(f"  - Class prediction shape: {seg_pred_class.shape}")
        
        # Count pixels per class
        unique_classes, counts = torch.unique(seg_pred_class, return_counts=True)
        print(f"  - Predicted classes: {unique_classes.cpu().numpy()}")
        print(f"  - Pixels per class: {counts.cpu().numpy()}")
        
        # Check lane existence
        exist_pred_np = exist_pred[0].cpu().numpy()
        lanes_exist = exist_pred_np > 0.5
        print(f"  - Lanes detected (>0.5 threshold): {np.where(lanes_exist)[0]}")
        
except Exception as e:
    print(f"\n[ERROR] Inference failed: {e}")
    raise

# ============================================================================
# Visualize Results
# ============================================================================
print("\n[DEBUG] Creating visualizations...")

# Convert predictions to numpy
seg_pred_np = seg_pred_class[0].cpu().numpy()  # Shape: (H, W)
seg_pred_softmax_np = seg_pred_softmax[0].cpu().numpy()  # Shape: (7, H, W)

print(f"  - Segmentation class map shape: {seg_pred_np.shape}")
print(f"  - Segmentation softmax shape: {seg_pred_softmax_np.shape}")

# 1. Original Frame (resized to input size for comparison)
frame_original = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

# 2. Detection Overlay
# Create colored lane mask
lane_colors = [
    [255, 0, 0],    # Lane 1 - Red
    [0, 255, 0],    # Lane 2 - Green
    [0, 0, 255],    # Lane 3 - Blue
    [255, 255, 0],  # Lane 4 - Yellow
    [255, 0, 255],  # Lane 5 - Magenta
    [0, 255, 255],  # Lane 6 - Cyan
]

overlay = frame_original.copy()

for lane_idx in range(1, 7):  # Classes 1-6 are lanes (0 is background)
    mask = (seg_pred_np == lane_idx)
    if mask.any():
        color = lane_colors[lane_idx - 1]
        overlay[mask] = color
        print(f"  - Lane {lane_idx} pixels: {mask.sum()}")

# Blend with original
alpha = 0.5
frame_overlay = cv2.addWeighted(frame_original, 1 - alpha, overlay, alpha, 0)

# 3. Raw Prediction Visualization
# Show max probability across all lane classes
lane_prob_max = np.max(seg_pred_softmax_np[1:, :, :], axis=0)  # Max over lanes 1-6
raw_pred = (lane_prob_max * 255).astype(np.uint8)
raw_pred_colored = cv2.applyColorMap(raw_pred, cv2.COLORMAP_JET)
raw_pred_colored = cv2.cvtColor(raw_pred_colored, cv2.COLOR_BGR2RGB)

print(f"  - Raw prediction range: {raw_pred.min()} to {raw_pred.max()}")

# ============================================================================
# Create Figure with 3 Subplots
# ============================================================================
print("\n[DEBUG] Creating final visualization...")

fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 3, figure=fig)

# Plot 1: Original
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(frame_original)
ax1.set_title('Original Frame', fontsize=14, fontweight='bold')
ax1.axis('off')

# Plot 2: Detection Overlay
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(frame_overlay)
ax2.set_title('Detection Overlay', fontsize=14, fontweight='bold')
ax2.axis('off')

# Add lane existence information
exist_text = "Lanes Detected:\n"
for i, prob in enumerate(exist_pred_np):
    if prob > 0.5:
        exist_text += f"Lane {i+1}: {prob:.2f}\n"
ax2.text(10, 30, exist_text, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10, verticalalignment='top')

# Plot 3: Raw Prediction
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(raw_pred_colored)
ax3.set_title('Raw Prediction (Heatmap)', fontsize=14, fontweight='bold')
ax3.axis('off')

plt.tight_layout()

# Save figure
output_path = os.path.join(OUTPUT_DIR, 'scnn_test_result.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  - Saved visualization to: {output_path}")

# Also save individual frames
cv2.imwrite(os.path.join(OUTPUT_DIR, '1_original.png'), 
            cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, '2_overlay.png'), 
            cv2.cvtColor(frame_overlay, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, '3_raw_prediction.png'), 
            cv2.cvtColor(raw_pred_colored, cv2.COLOR_RGB2BGR))

print(f"  - Saved individual frames to: {OUTPUT_DIR}")

plt.show()

# ============================================================================
# Additional Debug Information
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)

print("\n[Lane Existence Predictions]")
for i, prob in enumerate(exist_pred_np):
    status = "DETECTED" if prob > 0.5 else "NOT DETECTED"
    print(f"  Lane {i+1}: {prob:.4f} - {status}")

print("\n[Segmentation Statistics]")
total_pixels = seg_pred_np.size
for cls in range(7):
    count = (seg_pred_np == cls).sum()
    percentage = (count / total_pixels) * 100
    label = "Background" if cls == 0 else f"Lane {cls}"
    print(f"  {label}: {count} pixels ({percentage:.2f}%)")

print("\n[Confidence Statistics]")
for cls in range(1, 7):  # Only lanes
    lane_probs = seg_pred_softmax_np[cls]
    mean_conf = lane_probs.mean()
    max_conf = lane_probs.max()
    print(f"  Lane {cls}: Mean={mean_conf:.4f}, Max={max_conf:.4f}")

print("\n" + "=" * 80)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - Combined visualization: scnn_test_result.png")
print(f"  - Original frame: 1_original.png")
print(f"  - Detection overlay: 2_overlay.png")
print(f"  - Raw prediction: 3_raw_prediction.png")
