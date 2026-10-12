"""
Visualization Suite for Digital Elevation Models (DEMs).
Implements:
- Lambertian analytical hillshading (shaded relief with configurable solar azimuth and zenith)
- Multi-sample terrain grid visualization
- 3D perspective surface mesh plotting
- Hypsometric curve and slope distribution comparative plotting
- Latent manifold morphing sequence visualization
"""

from typing import Optional, List, Union, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import torch


def compute_hillshade(
    dem: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    cell_size: float = 30.0,
    z_factor: float = 1.0
) -> np.ndarray:
    """
    Computes shaded relief (hillshade) via Lambertian reflectance model:
    I = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
    """
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    # Matplotlib LightSource expects elevation in 2D array
    shaded = ls.hillshade(dem, vert_exag=z_factor, dx=cell_size, dy=cell_size)
    return shaded.astype(np.float32)


def plot_terrain_grid(
    dems: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    nrow: int = 4,
    ncol: int = 4,
    title: str = "Synthetic Topographical Terrains (WGAN-GP)"
):
    """
    Renders a grid of synthetic terrain elevation maps with realistic topographical hillshading.
    """
    if isinstance(dems, torch.Tensor):
        dems = dems.detach().cpu().numpy()
    if dems.ndim == 4:
        dems = dems.squeeze(1)
        
    num_samples = min(len(dems), nrow * ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 3.2))
    axes = np.array(axes).reshape(-1)
    
    ls = LightSource(azdeg=315, altdeg=45)
    
    for i in range(nrow * ncol):
        ax = axes[i]
        if i < num_samples:
            h = dems[i]
            # Blend colormap with hillshade for striking cartographic realism
            rgb = ls.shade(h, cmap=plt.cm.gist_earth, blend_mode='overlay', vert_exag=1.5)
            ax.imshow(rgb)
            ax.set_title(f"DEM #{i+1}", fontsize=10)
        ax.axis('off')
        
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_terrain(
    dem: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "3D Topographical Surface Mesh"
):
    """
    Generates a 3D perspective surface rendering of a digital elevation model.
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    H, W = dem.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        X, Y, dem,
        cmap=plt.cm.terrain,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True
    )
    
    ax.view_init(elev=50, azim=225)
    ax.set_zlim(0.0, 1.2)
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.set_zlabel("Normalized Elevation")
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hypsometric_curves(
    real_curves: List[Tuple[np.ndarray, np.ndarray]],
    syn_curves: List[Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots cumulative hypsometric curves (relative area vs relative elevation).
    """
    plt.figure(figsize=(7, 6))
    
    # Real curves (blue)
    for i, (elev, area) in enumerate(real_curves[:20]):
        label = "Real Terrain (USGS/SRTM)" if i == 0 else ""
        plt.plot(area, elev, color='tab:blue', alpha=0.3, linewidth=1.5, label=label)
        
    # Synthetic curves (orange)
    for i, (elev, area) in enumerate(syn_curves[:20]):
        label = "Synthetic (WGAN-GP Ours)" if i == 0 else ""
        plt.plot(area, elev, color='tab:orange', alpha=0.3, linewidth=1.5, label=label)
        
    plt.title("Hypsometric Curve Comparison (Area-Altitude Dynamics)", fontsize=12, fontweight='bold')
    plt.xlabel("Relative Surface Area (a / A)", fontsize=11)
    plt.ylabel("Relative Elevation (h / H)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_latent_interpolation_strip(
    dems: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Latent Manifold Geodesic Morphing (slerp)"
):
    """
    Plots a continuous horizontal sequence of interpolated DEMs demonstrating smooth terrain morphing.
    """
    if isinstance(dems, torch.Tensor):
        dems = dems.detach().cpu().numpy()
    if dems.ndim == 4:
        dems = dems.squeeze(1)
        
    num_steps = len(dems)
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2.2, 2.5))
    ls = LightSource(azdeg=315, altdeg=45)
    
    t_vals = np.linspace(0.0, 1.0, num_steps)
    for i in range(num_steps):
        ax = axes[i]
        rgb = ls.shade(dems[i], cmap=plt.cm.gist_earth, blend_mode='overlay', vert_exag=1.5)
        ax.imshow(rgb)
        ax.set_title(f"t = {t_vals[i]:.2f}", fontsize=9)
        ax.axis('off')
        
    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
