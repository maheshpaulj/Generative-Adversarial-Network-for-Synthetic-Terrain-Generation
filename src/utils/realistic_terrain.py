"""
Realistic Hierarchical Terrain & Continent Synthesis Engine.
Solves the "egg-carton" repetitive spike problem by combining:
1. Macro-geological continental structure (directional mountain chains, vast flat plains, river basins).
2. Deep WGAN-GP synthesized micro-geomorphological erosion details.
3. Alluvial hypsometric remapping to create genuine flat plains and natural valleys.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from scipy import ndimage
from PIL import Image

from ..models.generator import TerrainGenerator


class RealisticLandscapeEngine:
    """
    Synthesizes authentic, non-repetitive landscapes with expansive flat plains,
    coherent mountain chains, and deep river drainage basins.
    """
    def __init__(self, generator: Optional[TerrainGenerator] = None):
        self.generator = generator
        if self.generator is not None:
            self.generator.eval()

    @staticmethod
    def generate_macro_continental_base(
        size: int = 512,
        biome: str = "plains_mountains",
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates the macro-tectonic foundation: defines where flat plains,
        valleys, and mountain chains lie across the continent.
        """
        if seed is not None:
            np.random.seed(seed)

        # Low-frequency tectonic gradient (e.g. directional mountain chain)
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)

        if biome == "plains_mountains":
            # Mountain range cutting diagonally across, with vast flat plains on both sides
            angle = np.random.uniform(0.3, 0.8)
            axis = xx * np.cos(angle) + yy * np.sin(angle)
            
            # Mountain ridge concentrated along a diagonal corridor
            ridge_dist = np.abs(axis - np.random.uniform(-0.2, 0.2))
            macro = np.exp(- (ridge_dist ** 2) / 0.15)
            
            # Add secondary low-frequency rolling elevation
            low_noise = np.random.randn(8, 8).astype(np.float32)
            low_upsampled = np.array(Image.fromarray(low_noise).resize((size, size), resample=Image.BICUBIC))
            low_upsampled = (low_upsampled - low_upsampled.min()) / (low_upsampled.max() - low_upsampled.min() + 1e-6)
            macro = 0.75 * macro + 0.25 * low_upsampled

        elif biome == "rolling_plains":
            # Mostly low-elevation plains with gentle undulating swells
            low_noise = np.random.randn(6, 6).astype(np.float32)
            macro = np.array(Image.fromarray(low_noise).resize((size, size), resample=Image.BICUBIC))
            macro = (macro - macro.min()) / (macro.max() - macro.min() + 1e-6)
            macro = macro * 0.4  # Keep overall relief low

        elif biome == "valley_basin":
            # Central flat river basin / valley surrounded by distant hills
            dist_from_center = np.sqrt(xx**2 + yy**2)
            macro = np.clip((dist_from_center - 0.2) / 0.8, 0.0, 1.0)
            macro = np.power(macro, 1.5)
        else:
            macro = np.ones((size, size), dtype=np.float32) * 0.5

        return macro.astype(np.float32)

    def synthesize_landscape(
        self,
        size: int = 512,
        biome: str = "plains_mountains",
        plains_flatness: float = 0.5,
        mountain_prominence: float = 0.8,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Synthesizes a realistic, large-scale terrain:
        - Vast plains at lower elevations
        - Realistic mountain chains without repetitive spikes
        - High-frequency eroded rock textures infused from WGAN-GP
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 1. Macro-Tectonic Continental Foundation
        macro = self.generate_macro_continental_base(size=size, biome=biome, seed=seed)

        # 2. High-Frequency AI Detail Layer (WGAN-GP synthesis)
        # We sample multiple latent vectors and tile them smoothly across the macro layer
        detail_layer = np.zeros((size, size), dtype=np.float32)
        patch_sz = 128
        stride = 96
        
        # Calculate how many patches needed to cover size
        num_patches_y = int(np.ceil((size - patch_sz) / stride)) + 1
        num_patches_x = int(np.ceil((size - patch_sz) / stride)) + 1
        
        weight_acc = np.zeros((size, size), dtype=np.float32)
        win = np.outer(np.hanning(patch_sz), np.hanning(patch_sz)).astype(np.float32)

        with torch.no_grad():
            for py in range(num_patches_y):
                for px in range(num_patches_x):
                    # Sample latent vector
                    z = torch.randn(1, 128)
                    if self.generator is not None:
                        patch = self.generator(z).squeeze().cpu().numpy()
                    else:
                        from ..data.dataset import ProceduralTerrainGenerator
                        patch = ProceduralTerrainGenerator.generate_fbm(size=patch_sz)

                    y0 = min(py * stride, size - patch_sz)
                    x0 = min(px * stride, size - patch_sz)
                    
                    detail_layer[y0:y0+patch_sz, x0:x0+patch_sz] += patch * win
                    weight_acc[y0:y0+patch_sz, x0:x0+patch_sz] += win

        safe_w = np.where(weight_acc < 1e-5, 1.0, weight_acc)
        detail_layer = detail_layer / safe_w
        detail_layer = (detail_layer - detail_layer.min()) / (detail_layer.max() - detail_layer.min() + 1e-6)

        # 3. Structural Integration (Modulated Coupling)
        # Mountains get sharp AI micro-erosion; plains remain smooth and flat!
        # Mask where mountains are allowed to rise based on macro
        mountain_mask = np.clip((macro - (1.0 - mountain_prominence)) / (mountain_prominence + 1e-5), 0.0, 1.0)
        mountain_mask = np.power(mountain_mask, 1.5)

        # Combine macro continent with AI detail
        combined = (macro * 0.6) + (detail_layer * mountain_mask * 0.4)

        # 4. Alluvial Plain Flattening (Geomorphological Sediment Remapping)
        # In real geology, lower elevations are sediment-filled flat floodplains
        # We apply an S-curve that flattens low elevations into true plains
        threshold = plains_flatness * 0.5  # E.g. bottom 25-35% of terrain
        diff = np.maximum(combined - threshold, 0.0)
        flat_plains = np.where(
            combined < threshold,
            combined * 0.2,  # Flatten significantly into broad flat lowlands
            threshold * 0.2 + (diff ** 1.15)
        )

        # Final min-max normalization
        f_min, f_max = flat_plains.min(), flat_plains.max()
        terrain = (flat_plains - f_min) / (f_max - f_min + 1e-6)

        # Slight hydraulic blur on flat plains to mimic sediment deposition
        plain_mask = (terrain < 0.2).astype(np.float32)
        smoothed = ndimage.gaussian_filter(terrain, sigma=0.8)
        terrain = terrain * (1.0 - plain_mask * 0.5) + smoothed * (plain_mask * 0.5)

        km_span = (size * 30.0) / 1000.0
        meta = {
            "size": (size, size),
            "biome": biome,
            "km_span": km_span,
            "total_area_sq_km": km_span * km_span,
            "plain_area_percent": float(np.mean(terrain < 0.25) * 100.0)
        }

        return terrain.astype(np.float32), meta
