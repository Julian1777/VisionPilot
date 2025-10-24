"""
SCNN Model Definition
Spatial CNN for Lane Detection
Based on: https://arxiv.org/abs/1712.06080
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCNN(nn.Module):
    """
    Spatial CNN for Lane Detection
    
    Architecture:
    - VGG16-BN backbone with dilated convolutions
    - Spatial message passing (up-down, down-up, left-right, right-left)
    - Lane segmentation + existence prediction
    
    Input: RGB image tensor (B, 3, H, W)
    Output: 
        - Segmentation: (B, 7, H, W) - 7 classes (background + 6 lanes)
        - Existence: (B, 6) - probability of each lane existing
    """
    
    def __init__(self, input_size, ms_ks=9, pretrained=False):
        """
        Args:
            input_size (tuple): (width, height) of input images
            ms_ks (int): kernel size for message passing
            pretrained (bool): whether to use pretrained VGG16 backbone
        """
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
        """
        Forward pass
        
        Args:
            img: Input tensor (B, 3, H, W)
            seg_gt: Ground truth segmentation (optional, for training)
            exist_gt: Ground truth existence (optional, for training)
            
        Returns:
            seg_pred: Segmentation prediction (B, 7, H, W)
            exist_pred: Existence prediction (B, 6)
            loss_seg: Segmentation loss (if gt provided)
            loss_exist: Existence loss (if gt provided)
            loss: Total loss (if gt provided)
        """
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
            
        Returns:
            Message-passed tensor
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
