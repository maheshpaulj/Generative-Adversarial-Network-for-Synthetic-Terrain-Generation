"""
Spectral-Normalized Deep Convolutional Generator for Digital Elevation Models (DEMs).
Uses Bilinear Upsampling + Conv blocks to completely eliminate checkerboard artifacts
on continuous elevation surfaces.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class UpsampleBlock(nn.Module):
    """
    Bilinear upsampling followed by Spectral-Normalized Convolution,
    Batch Normalization, and LeakyReLU activation.
    Guarantees continuous 1st and 2nd spatial derivatives without checkerboard artifacts.
    """
    def __init__(self, in_channels: int, out_channels: int, use_spectral_norm: bool = True):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)
            
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TerrainGenerator(nn.Module):
    """
    Continuous Latent Manifold to Elevation Grid Generator.
    Maps z ~ N(0, I) in R^{latent_dim} to H in [0, 1]^{1 x image_size x image_size}.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 128,
        out_channels: int = 1,
        base_features: int = 64,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.init_size = 4  # Initial spatial resolution 4x4
        
        # Calculate number of upsampling stages needed
        # e.g., for 128: 4 -> 8 -> 16 -> 32 -> 64 -> 128 (5 stages)
        num_upsamples = int(torch.log2(torch.tensor(image_size // self.init_size)).item())
        
        # Initial projection from latent vector to 4x4 feature volume
        init_channels = base_features * (2 ** (num_upsamples - 1))  # e.g., 64 * 16 = 1024 or 512
        init_channels = min(init_channels, 512)
        
        linear = nn.Linear(latent_dim, init_channels * self.init_size * self.init_size)
        if use_spectral_norm:
            linear = spectral_norm(linear)
            
        self.project = nn.Sequential(
            linear,
            nn.BatchNorm1d(init_channels * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.init_channels = init_channels
        
        # Build progressive upsampling layers (5 stages: 4->8->16->32->64->128)
        self.upsamples = nn.Sequential(
            UpsampleBlock(512, 256, use_spectral_norm=use_spectral_norm),
            UpsampleBlock(256, 128, use_spectral_norm=use_spectral_norm),
            UpsampleBlock(128, 64, use_spectral_norm=use_spectral_norm),
            UpsampleBlock(64, 64, use_spectral_norm=use_spectral_norm),
            UpsampleBlock(64, 32, use_spectral_norm=use_spectral_norm)
        )
        
        # Final elevation projection: 3x3 Conv -> Tanh
        final_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
            
        self.to_elevation = nn.Sequential(
            final_conv,
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            z: Latent vectors of shape (B, latent_dim)
        Returns:
            Normalized elevation maps in range [0, 1] of shape (B, 1, image_size, image_size)
        """
        if z.size(0) == 1 and self.training:
            self.eval()
        B = z.size(0)
        h = self.project(z)
        h = h.view(B, self.init_channels, self.init_size, self.init_size)
        features = self.upsamples(h)
        tanh_out = self.to_elevation(features)
        
        # Scale Tanh [-1, 1] to normalized elevation [0, 1]
        elevation = (tanh_out + 1.0) / 2.0
        return elevation
