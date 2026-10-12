"""
Export tools for Digital Elevation Models (DEMs).
Supports exporting to:
- 16-bit Grayscale PNG Heightmaps (Unreal Engine / Unity / Blender compatible)
- Wavefront OBJ 3D Polygonal Meshes
- Geospatial GeoTIFF rasters (QGIS / ArcGIS compatible)
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from PIL import Image


def export_as_png_heightmap(
    dem: Union[np.ndarray, torch.Tensor],
    filepath: Union[str, Path]
):
    """
    Exports elevation raster as a 16-bit grayscale PNG heightmap (0 - 65,535).
    Standard format for Unreal Engine, Unity, and Blender displacement.
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize to 16-bit unsigned integer range [0, 65535]
    d_min, d_max = dem.min(), dem.max()
    if d_max - d_min > 1e-6:
        norm = (dem - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(dem)
        
    u16 = (norm * 65535.0).astype(np.uint16)
    img = Image.fromarray(u16, mode='I;16')
    img.save(filepath)
    print(f"Exported 16-bit heightmap to {filepath}")


def export_as_obj_mesh(
    dem: Union[np.ndarray, torch.Tensor],
    filepath: Union[str, Path],
    horizontal_scale: float = 1.0,
    vertical_scale: float = 25.0
):
    """
    Exports elevation surface as a Wavefront OBJ 3D polygonal mesh.
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    H, W = dem.shape
    
    with open(filepath, "w") as f:
        f.write("# Synthetic Topographical Terrain Mesh (WGAN-GP)\n")
        f.write(f"# Resolution: {W} x {H}\n")
        
        # Write vertices (v x y z)
        for r in range(H):
            for c in range(W):
                x = c * horizontal_scale
                z = r * horizontal_scale
                y = dem[r, c] * vertical_scale
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
                
        # Write quad faces (1-indexed)
        for r in range(H - 1):
            for c in range(W - 1):
                # 1-indexed vertex IDs
                v1 = r * W + c + 1
                v2 = r * W + (c + 1) + 1
                v3 = (r + 1) * W + (c + 1) + 1
                v4 = (r + 1) * W + c + 1
                f.write(f"f {v1} {v2} {v3} {v4}\n")
                
    print(f"Exported 3D OBJ mesh to {filepath}")


def export_as_geotiff(
    dem: Union[np.ndarray, torch.Tensor],
    filepath: Union[str, Path],
    resolution: float = 30.0
):
    """
    Exports elevation raster as a 32-bit floating point GeoTIFF using rasterio.
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import rasterio
        from rasterio.transform import from_origin
        
        H, W = dem.shape
        transform = from_origin(0.0, 0.0, resolution, resolution)
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=1,
            dtype=rasterio.float32,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(dem.astype(np.float32), 1)
        print(f"Exported GeoTIFF to {filepath}")
    except ImportError:
        # Fallback to NumPy save if rasterio is not installed
        npy_path = filepath.with_suffix(".npy")
        np.save(npy_path, dem.astype(np.float32))
        print(f"rasterio not available; saved raw array as {npy_path}")
