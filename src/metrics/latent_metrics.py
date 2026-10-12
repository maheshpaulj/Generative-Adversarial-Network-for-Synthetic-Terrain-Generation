"""
Latent Manifold Dynamics, Geodesic Interpolation, and Path Smoothness Metrics.
Implements:
- Spherical Linear Interpolation (slerp) on latent hyperspheres
- Perceptual Path Length (PPL) and Elevation Path Length (EPL)
- Latent Space Arithmetic for semantic terrain editing
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical Linear Interpolation (slerp) between two latent Gaussian vectors.
    Traverses the great-circle arc on the hypersphere.
    
    slerp(z1, z2; t) = sin((1-t)*theta)/sin(theta) * z1 + sin(t*theta)/sin(theta) * z2
    """
    # Normalize vectors to find angle
    z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)
    
    dot = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # If vectors are nearly collinear, fall back to linear lerp
    if torch.abs(sin_theta).max() < eps:
        return (1.0 - t) * z1 + t * z2
        
    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta
    
    return w1 * z1 + w2 * z2


def generate_interpolation_path(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    num_steps: int = 16,
    method: str = "slerp"
) -> torch.Tensor:
    """
    Generates a sequence of interpolated latent vectors of shape (num_steps, latent_dim).
    """
    t_values = np.linspace(0.0, 1.0, num_steps)
    path = []
    
    for t in t_values:
        if method == "slerp":
            z_t = slerp(z_start, z_end, float(t))
        else:
            z_t = (1.0 - t) * z_start + t * z_end
        path.append(z_t)
        
    return torch.cat(path, dim=0)


def compute_perceptual_path_length(
    generator: nn.Module,
    latent_dim: int = 128,
    num_samples: int = 100,
    epsilon: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Tuple[float, float]:
    """
    Computes Perceptual and Elevation Path Length (PPL) on the latent manifold.
    PPL measures the rate of change of generated terrain as latent coordinates vary.
    Lower PPL indicates smoother, more continuous terrain morphing without mode jumps.
    
    Returns:
        mean_ppl: Average path length
        std_ppl: Standard deviation (variance indicates consistency)
    """
    generator.eval()
    path_lengths = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)
            
            # Sample random step point t in [0, 1 - epsilon]
            t = np.random.uniform(0.0, 1.0 - epsilon)
            zt = slerp(z1, z2, t)
            zt_eps = slerp(z1, z2, t + epsilon)
            
            dem_t = generator(zt)
            dem_eps = generator(zt_eps)
            
            # L1 elevation distance + gradient difference
            elev_diff = torch.mean(torch.abs(dem_t - dem_eps))
            
            # Normalize by step epsilon^2
            d = elev_diff / (epsilon ** 2)
            path_lengths.append(d.item())
            
    mean_ppl = float(np.mean(path_lengths))
    std_ppl = float(np.std(path_lengths))
    return mean_ppl, std_ppl


def latent_vector_arithmetic(
    generator: nn.Module,
    z_base: torch.Tensor,
    z_feature_pos: torch.Tensor,
    z_feature_neg: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Executes latent space arithmetic: z_out = z_base + alpha * (z_feature_pos - z_feature_neg)
    Enables semantic topographical editing (e.g. Base + alpha * (Alpine - Plain)).
    """
    generator.eval()
    with torch.no_grad():
        direction = z_feature_pos - z_feature_neg
        z_target = z_base + alpha * direction
        return generator(z_target)
