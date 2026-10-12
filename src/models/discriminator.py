"""
Multi-Scale PatchGAN Critic with Spectral Normalization for Digital Elevation Models.
Discriminates at multiple spatial resolutions (1x and 0.5x) to simultaneously capture
micro-geomorphic features (slope, ridges, channels) and macro-geomorphic basins.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List, Tuple


class PatchGANCritic(nn.Module):
    """
    Single-scale PatchGAN Critic.
    Maps an elevation raster (B, in_channels, H, W) to a 2D spatial grid of realism scores.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        def conv_layer(in_c: int, out_c: int, stride: int = 2):
            conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            return conv

        # First layer (no batch/instance norm to preserve absolute contrast)
        layers = [
            conv_layer(in_channels, base_features, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers with doubling feature channels
        curr_features = base_features
        for i in range(1, n_layers):
            next_features = min(curr_features * 2, 512)
            layers.extend([
                conv_layer(curr_features, next_features, stride=2),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            curr_features = next_features
            
        # Stride 1 penultimate layer
        layers.extend([
            conv_layer(curr_features, curr_features, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Final patch-level 1-channel linear projection
        final_conv = nn.Conv2d(curr_features, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        layers.append(final_conv)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScalePatchGANCritic(nn.Module):
    """
    Multi-Scale PatchGAN Discriminator.
    Consists of num_scales distinct sub-critics evaluating the elevation surface at:
    - Scale 1: Full resolution (128x128) -> captures slopes, ridges, rill channels.
    - Scale 2: 2x downsampled (64x64)   -> captures macro-valleys, drainage basins.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 64,
        num_scales: int = 2,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.critics = nn.ModuleList([
            PatchGANCritic(
                in_channels=in_channels,
                base_features=base_features,
                n_layers=3,
                use_spectral_norm=use_spectral_norm
            )
            for _ in range(num_scales)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input elevation map (B, 1, H, W)
        Returns:
            List of critic prediction tensors at each scale [(B, 1, H_1, W_1), (B, 1, H_2, W_2), ...]
        """
        outputs = []
        curr_x = x
        for i, critic in enumerate(self.critics):
            if i > 0:
                curr_x = self.downsample(curr_x)
            outputs.append(critic(curr_x))
        return outputs
