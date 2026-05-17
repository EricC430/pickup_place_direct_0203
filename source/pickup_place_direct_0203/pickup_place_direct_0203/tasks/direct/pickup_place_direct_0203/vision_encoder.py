# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Vision Encoder Module for CNN-based Feature Extraction
This module provides CNN models to extract features from RGB and Depth images,
converting high-dimensional visual data into compact feature vectors suitable
for RL policy networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class SimpleVisonEncoder(nn.Module):
    """
    Lightweight CNN encoder for RGB + Depth images.
    
    Input: RGB (3 channels) + Depth (1 channel) concatenated = 4 channels total
    Output: Compact feature vector (default 128 dims)
    
    Architecture:
    - Conv layers with ReLU activations
    - Max pooling for spatial reduction
    - Flatten and FC layers for feature extraction
    
    Expected input shape: (B, 4, H, W) where B=batch, H=height, W=width
    """
    
    def __init__(
        self,
        image_height: int = 120,
        image_width: int = 80,
        input_channels: int = 3,  # RGB (3)
        feature_dim: int = 128,
        use_batch_norm: bool = True,
    ):
        """
        Initialize the Simple Vision Encoder.
        
        Args:
            image_height: Input image height (default 120)
            image_width: Input image width (default 80)
            input_channels: Input channels, typically 4 (RGB+Depth)
            feature_dim: Output feature dimension (default 128)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        conv_channels = [16, 32, 64]
        
        layers = []
        in_channels = input_channels
        
        # Conv Block 1: 4 -> 16
        layers.append(nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(conv_channels[0]))
        layers.append(nn.ReLU(inplace=True))
        
        # Conv Block 2: 16 -> 32
        layers.append(nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(conv_channels[1]))
        layers.append(nn.ReLU(inplace=True))
        
        # Conv Block 3: 32 -> 64
        layers.append(nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(conv_channels[2]))
        layers.append(nn.ReLU(inplace=True))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        # After 3x stride-2 operations: H' = H / 8, W' = W / 8
        flat_h = image_height // 8
        flat_w = image_width // 8
        flat_size = flat_h * flat_w * conv_channels[2]
        
        # FC layers for feature extraction
        fc_layers = [
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
        ]
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        print(f"[VisionEncoder] Initialized with output dim={feature_dim}, "
              f"input_shape=(B, {input_channels}, {image_height}, {image_width})")
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            image: Input tensor of shape (B, C, H, W) where C should be input_channels
        
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        # Ensure input is on the same device as model
        if image.device != next(self.parameters()).device:
            image = image.to(next(self.parameters()).device)
        
        # Pass through conv layers
        x = self.conv_layers(image)
        
        # Pass through FC layers
        features = self.fc_layers(x)
        
        return features


class ResNetVisonEncoder(nn.Module):
    """
    ResNet-based Vision Encoder for more sophisticated feature extraction.
    
    Uses a lightweight ResNet backbone (pre-trained on ImageNet if available)
    with a small adaptation head to merge RGB and Depth information.
    
    Input: RGB (3 channels)
    Output: Compact feature vector (default 128 dims)
    """
    
    def __init__(
        self,
        image_height: int = 120,
        image_width: int = 80,
        feature_dim: int = 128,
        pretrained: bool = False,
    ):
        """
        Initialize ResNet Vision Encoder.
        
        Args:
            image_height: Input image height
            image_width: Input image width
            feature_dim: Output feature dimension
            pretrained: Use pre-trained ImageNet weights (requires torchvision)
        """
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.feature_dim = feature_dim
        
        try:
            from torchvision import models
            
            # Create ResNet18 backbone (pretrained on ImageNet)
            if pretrained:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                backbone = models.resnet18(weights=None)
            
            # Remove the final classification layer
            layers = list(backbone.children())[:-1]
            self.backbone = nn.Sequential(*layers)
            
            # Head for feature extraction
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),  # ResNet18 outputs 512-dim features
                nn.ReLU(inplace=True),
                nn.Linear(256, feature_dim),
            )
            
            self.backbone_name = "ResNet18"
            print(f"[VisionEncoder] Initialized ResNet18 encoder with feature_dim={feature_dim}")
            
        except ImportError:
            print("[VisionEncoder] Warning: torchvision not available. Falling back to SimpleVisonEncoder.")
            self._fallback_encoder = SimpleVisonEncoder(
                image_height=image_height,
                image_width=image_width,
                feature_dim=feature_dim,
            )
            self.use_fallback = True
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet encoder.
        
        Args:
            image: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self._fallback_encoder(image)
        
        # Pass through backbone
        x = self.backbone(image)
        
        # Pass through head
        features = self.head(x)
        
        return features


def get_vision_encoder(
    encoder_type: str = "simple",
    image_height: int = 120,
    image_width: int = 80,
    feature_dim: int = 128,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Factory function to create vision encoders.
    
    Args:
        encoder_type: Type of encoder ("simple" or "resnet")
        image_height: Input image height
        image_width: Input image width
        feature_dim: Output feature dimension
        device: Device to place model on
    
    Returns:
        Vision encoder model
    """
    if encoder_type.lower() == "simple":
        encoder = SimpleVisonEncoder(
            image_height=image_height,
            image_width=image_width,
            feature_dim=feature_dim,
        )
    elif encoder_type.lower() == "resnet":
        encoder = ResNetVisonEncoder(
            image_height=image_height,
            image_width=image_width,
            feature_dim=feature_dim,
            pretrained=True,  # Download ImageNet weights
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    encoder = encoder.to(device)
    encoder.eval()  # Set to evaluation mode (we won't train this)
    
    return encoder

class PointNetEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, ptcloud):
        """
        ptcloud: (B, N, 3)
        returns: (B, out_dim)
        """
        x = ptcloud.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2)[0]
        return x

