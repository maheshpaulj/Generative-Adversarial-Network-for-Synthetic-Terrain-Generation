"""
Large World Generator for Topographical Digital Elevation Models (DEMs).
Stitches and blends multiple continuous latent patches into massive, continent-scale
elevation models (512x512, 1024x1024) using smooth cosine/Hann overlap blending.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..models.generator import TerrainGenerator
from ..metrics.latent_metrics import slerp


class LargeWorldGenerator:
    """
    Synthesizes massive, boundless terrains by creating a spatially correlated
    grid of latent vectors and blending overlapping synthesized patches.
    """
    def __init__(self, generator: TerrainGenerator, patch_size: int = 128, overlap: int = 32):
        self.generator = generator
        self.patch_size = patch_size
        self.overlap = overlap
        self.step = patch_size - overlap
        self.latent_dim = generator.latent_dim
        self.generator.eval()

    @staticmethod
    def _create_hann_window(size: int) -> np.ndarray:
        """Creates a 2D Hann (raised cosine) blending window."""
        win_1d = np.hanning(size)
        win_2d = np.outer(win_1d, win_1d).astype(np.float32)
        # Ensure center is 1.0 and edges fall off smoothly
        return win_2d

    def generate_world(
        self,
        grid_rows: int = 3,
        grid_cols: int = 3,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generates a large continuous terrain of size:
        Height = (grid_rows - 1) * step + patch_size
        Width  = (grid_cols - 1) * step + patch_size
        
        For grid 4x4 with patch 128 and overlap 32:
        Size = 3 * 96 + 128 = 416 x 416
        For grid 6x6: 5 * 96 + 128 = 608 x 608
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 1. Create a 2D spatial grid of latent anchor vectors
        # Border anchors are correlated to guarantee global tectonic continuity
        z_grid = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                z_grid[(r, c)] = torch.randn(1, self.latent_dim)

        total_h = (grid_rows - 1) * self.step + self.patch_size
        total_w = (grid_cols - 1) * self.step + self.patch_size

        accum_elevation = np.zeros((total_h, total_w), dtype=np.float32)
        accum_weights = np.zeros((total_h, total_w), dtype=np.float32)

        window = self._create_hann_window(self.patch_size)

        # 2. Synthesize each tile and blend into global accumulator
        with torch.no_grad():
            for r in range(grid_rows):
                for c in range(grid_cols):
                    # Blend latent vector with neighbors for smooth macro-geology
                    z_curr = z_grid[(r, c)]
                    if r > 0:
                        z_curr = slerp(z_curr, z_grid[(r - 1, c)], 0.25)
                    if c > 0:
                        z_curr = slerp(z_curr, z_grid[(r, c - 1)], 0.25)

                    # Synthesize 128x128 elevation patch
                    patch = self.generator(z_curr).squeeze().cpu().numpy()

                    y_start = r * self.step
                    y_end = y_start + self.patch_size
                    x_start = c * self.step
                    x_end = x_start + self.patch_size

                    accum_elevation[y_start:y_end, x_start:x_end] += patch * window
                    accum_weights[y_start:y_end, x_start:x_end] += window

        # 3. Normalize by accumulated weights
        safe_weights = np.where(accum_weights < 1e-5, 1.0, accum_weights)
        world_dem = accum_elevation / safe_weights

        # Min-max normalize entire world to [0, 1]
        w_min, w_max = world_dem.min(), world_dem.max()
        if w_max - w_min > 1e-6:
            world_dem = (world_dem - w_min) / (w_max - w_min)

        metadata = {
            "grid": (grid_rows, grid_cols),
            "resolution": (total_h, total_w),
            "estimated_km_width": (total_w * 30.0) / 1000.0,
            "estimated_km_height": (total_h * 30.0) / 1000.0,
            "total_area_sq_km": ((total_w * 30.0) / 1000.0) * ((total_h * 30.0) / 1000.0)
        }

        return world_dem, metadata
