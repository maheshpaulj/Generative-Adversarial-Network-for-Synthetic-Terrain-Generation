"""
Fréchet Inception Distance (FID) adapted for Digital Elevation Models.
Maps 1-channel scalar elevation rasters to a 3-channel composite [Elevation, Hillshade, Slope]
to capture both macro-elevations and micro-textural relief shadows via Inception-V3 embeddings.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from scipy import linalg
from ..utils.visualization import compute_hillshade
from .geomorphology import calculate_slope


class DEMInceptionFeatureExtractor:
    """
    Extracts 2048-dimensional feature representations from Inception-V3 pool3 layer.
    """
    def __init__(self, device: torch.device):
        self.device = device
        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for Inception FID calculations.")
        # Load pre-trained Inception-V3
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()  # Extract 2048-dim pool3 embeddings
        model.eval()
        self.model = model.to(device)
        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    def prepare_dem_tensor(self, dem_batch: torch.Tensor) -> torch.Tensor:
        """
        Converts (B, 1, H, W) in [0, 1] to (B, 3, 299, 299) Inception inputs:
        - Ch 0: Normalized elevation
        - Ch 1: Analytical hillshaded relief
        - Ch 2: Gradient magnitude (slope)
        """
        B, C, H, W = dem_batch.shape
        dem_np = dem_batch.squeeze(1).detach().cpu().numpy()
        
        composite_batch = []
        for i in range(B):
            h_map = dem_np[i]
            shade = compute_hillshade(h_map, azimuth=315.0, altitude=45.0)
            slope = calculate_slope(h_map) / 90.0  # Normalized to [0, 1]
            
            comp = np.stack([h_map, shade, slope], axis=0)  # (3, H, W)
            composite_batch.append(comp)
            
        comp_tensor = torch.from_numpy(np.array(composite_batch, dtype=np.float32)).to(self.device)
        # Upsample to 299x299
        comp_299 = self.upsample(comp_tensor)
        # Standardize to [-1, 1]
        comp_norm = (comp_299 * 2.0) - 1.0
        return comp_norm

    @torch.no_grad()
    def get_features(self, dem_batch: torch.Tensor) -> np.ndarray:
        inputs = self.prepare_dem_tensor(dem_batch)
        features = self.model(inputs)
        return features.cpu().numpy()


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Computes the Fréchet Inception Distance between two multivariate Gaussians:
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary artifact removal
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_fid_score(
    real_features: np.ndarray,
    syn_features: np.ndarray
) -> float:
    """
    Computes FID score between real and synthetic feature matrices (N x 2048).
    """
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_syn = np.mean(syn_features, axis=0)
    sigma_syn = np.cov(syn_features, rowvar=False)
    
    return calculate_frechet_distance(mu_real, sigma_real, mu_syn, sigma_syn)
