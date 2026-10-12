"""
Dataset and data loader module for Digital Elevation Models (DEMs).
Supports real-world GeoTIFF / NumPy / PNG elevation rasters and provides
an on-the-fly Procedural Synthetic DEM dataset with hydraulic erosion fallback.
"""

import os
import glob
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class RealDEMDataset(Dataset):
    """
    Dataset for real-world Digital Elevation Models stored as:
    - .npy files (single 2D float arrays)
    - .png / .tif files (16-bit or 8-bit heightmaps)
    
    Each elevation patch is normalized to [0, 1] and augmented with
    dihedral group D4 symmetries (rotations and reflections) which are physically valid for terrain.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        patch_size: int = 128,
        augment: bool = True,
        elevation_min: float = 0.0,
        elevation_max: float = 1.0
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        
        # Scan for supported elevation files
        self.file_list = (
            sorted(glob.glob(str(self.data_dir / "**/*.npy"), recursive=True)) +
            sorted(glob.glob(str(self.data_dir / "**/*.tif"), recursive=True)) +
            sorted(glob.glob(str(self.data_dir / "**/*.png"), recursive=True))
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        filepath = self.file_list[idx]
        
        # Load raw raster
        if filepath.endswith(".npy"):
            elevation = np.load(filepath).astype(np.float32)
        else:
            img = Image.open(filepath)
            elevation = np.array(img, dtype=np.float32)
            
        # Ensure 2D
        if elevation.ndim == 3:
            elevation = elevation[:, :, 0]
            
        # Resize or crop to target patch_size if needed
        h, w = elevation.shape
        if h != self.patch_size or w != self.patch_size:
            if h >= self.patch_size and w >= self.patch_size:
                # Random spatial crop
                top = np.random.randint(0, h - self.patch_size + 1)
                left = np.random.randint(0, w - self.patch_size + 1)
                elevation = elevation[top:top+self.patch_size, left:left+self.patch_size]
            else:
                # Bilinear resize via PIL
                img = Image.fromarray(elevation)
                img = img.resize((self.patch_size, self.patch_size), resample=Image.BILINEAR)
                elevation = np.array(img, dtype=np.float32)

        # Min-max normalization to [0, 1]
        e_min = elevation.min()
        e_max = elevation.max()
        if e_max - e_min > 1e-5:
            elevation = (elevation - e_min) / (e_max - e_min)
        else:
            elevation = np.zeros_like(elevation)
            
        # Data Augmentation: Dihedral Group D4 (preserves terrain validity)
        if self.augment:
            if np.random.rand() > 0.5:
                elevation = np.fliplr(elevation)
            if np.random.rand() > 0.5:
                elevation = np.flipud(elevation)
            k = np.random.choice([0, 1, 2, 3])
            elevation = np.rot90(elevation, k)
            
        elevation = np.ascontiguousarray(elevation, dtype=np.float32)
        tensor = torch.from_numpy(elevation).unsqueeze(0)  # Shape: (1, H, W)
        return tensor


class ProceduralTerrainGenerator:
    """
    High-fidelity procedural terrain generator using multi-octave Fractional Brownian Motion (fBm)
    coupled with thermal and hydraulic erosion simulation.
    Acts as a self-contained, realistic synthetic DEM synthesizer.
    """
    @staticmethod
    def generate_fbm(
        size: int = 128,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        terrain = np.zeros((size, size), dtype=np.float32)
        frequency = 1.0
        amplitude = 1.0
        
        for _ in range(octaves):
            # Generate random 2D wave lattice
            grid_res = max(2, int(frequency * 4))
            random_lattice = np.random.randn(grid_res, grid_res).astype(np.float32)
            
            # Upsample lattice to full size
            lattice_img = Image.fromarray(random_lattice)
            upsampled = lattice_img.resize((size, size), resample=Image.BICUBIC)
            terrain += np.array(upsampled, dtype=np.float32) * amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
            
        # Normalize
        t_min, t_max = terrain.min(), terrain.max()
        if t_max - t_min > 1e-6:
            terrain = (terrain - t_min) / (t_max - t_min)
            
        # Ridge and valley tectonic exponentiation
        terrain = np.power(terrain, 1.4)
        return terrain

    @staticmethod
    def apply_hydraulic_erosion(
        terrain: np.ndarray,
        num_droplets: int = 2000,
        inertia: float = 0.1,
        sediment_capacity_factor: float = 4.0,
        min_sediment_capacity: float = 0.01,
        erode_speed: float = 0.3,
        deposit_speed: float = 0.3,
        evaporate_speed: float = 0.02,
        gravity: float = 4.0,
        max_droplet_lifetime: int = 30
    ) -> np.ndarray:
        """
        Particle-based hydraulic erosion simulation (Musgrave / Benes algorithm).
        Carves realistic dendritic fluvial valleys and creates talus deposits.
        """
        h = terrain.copy()
        size = h.shape[0]
        
        for _ in range(num_droplets):
            px = np.random.uniform(1, size - 2)
            py = np.random.uniform(1, size - 2)
            dir_x, dir_y = 0.0, 0.0
            speed = 1.0
            water = 1.0
            sediment = 0.0
            
            for _ in range(max_droplet_lifetime):
                ix, iy = int(px), int(py)
                u, v = px - ix, py - iy
                
                # Bilinear gradient
                gx = (h[ix+1, iy] - h[ix, iy]) * (1 - v) + (h[ix+1, iy+1] - h[ix, iy+1]) * v
                gy = (h[ix, iy+1] - h[ix, iy]) * (1 - u) + (h[ix+1, iy+1] - h[ix+1, iy]) * u
                
                dir_x = dir_x * inertia - gx * (1 - inertia)
                dir_y = dir_y * inertia - gy * (1 - inertia)
                
                mag = np.hypot(dir_x, dir_y)
                if mag < 1e-6:
                    break
                dir_x /= mag
                dir_y /= mag
                
                new_px = px + dir_x
                new_py = py + dir_y
                
                if new_px < 1 or new_px >= size - 2 or new_py < 1 or new_py >= size - 2:
                    break
                    
                # Elevation change
                diff = h[int(new_px), int(new_py)] - h[ix, iy]
                
                # Capacity
                sediment_capacity = max(-diff * speed * water * sediment_capacity_factor, min_sediment_capacity)
                
                if sediment > sediment_capacity or diff > 0:
                    deposit_amount = (sediment - sediment_capacity) * deposit_speed if diff <= 0 else min(sediment, diff)
                    sediment -= deposit_amount
                    h[ix, iy] += deposit_amount * 0.5
                else:
                    erode_amount = min((sediment_capacity - sediment) * erode_speed, -diff)
                    sediment += erode_amount
                    h[ix, iy] -= erode_amount * 0.5
                    
                speed = np.sqrt(max(0.0, speed * speed + diff * gravity))
                water *= (1.0 - evaporate_speed)
                px, py = new_px, new_py
                
        # Re-normalize
        t_min, t_max = h.min(), h.max()
        if t_max - t_min > 1e-6:
            h = (h - t_min) / (t_max - t_min)
        return h.astype(np.float32)


class SyntheticDEMDataset(Dataset):
    """
    On-the-fly Procedural DEM Dataset.
    Generates synthetic eroded mountain terrain patches on demand.
    Ideal for training without external downloads or for reproducible benchmarking.
    """
    def __init__(self, length: int = 1000, patch_size: int = 128, erode: bool = True):
        self.length = length
        self.patch_size = patch_size
        self.erode = erode

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Generate varied landscapes (alpine, rolling hills, plateau)
        octaves = np.random.choice([5, 6, 7])
        persistence = np.random.uniform(0.45, 0.55)
        lacunarity = np.random.uniform(1.9, 2.1)
        
        terrain = ProceduralTerrainGenerator.generate_fbm(
            size=self.patch_size,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            seed=None
        )
        
        if self.erode and np.random.rand() > 0.3:
            droplets = np.random.randint(1500, 3000)
            terrain = ProceduralTerrainGenerator.apply_hydraulic_erosion(terrain, num_droplets=droplets)
            
        tensor = torch.from_numpy(terrain).unsqueeze(0)
        return tensor
