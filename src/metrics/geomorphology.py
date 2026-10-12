"""
Differential Geomorphometry and Topographical Metrics Module.
Implements:
- First-order derivatives: Slope magnitude and Aspect orientation
- Second-order derivatives: Profile curvature, Planform curvature, and Laplacian
- Hypsometric Integral (HI) and cumulative area-elevation hypsometric curves
- Distributional discrepancy metrics (Wasserstein distance, Kolmogorov-Smirnov)
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
from scipy import stats


def compute_derivatives_numpy(dem: np.ndarray, cell_size: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes first- and second-order partial derivatives using Horn's 3x3 finite difference operator.
    Returns (p, q, r, t, s) where:
    p = dh/dx, q = dh/dy, r = d2h/dx2, t = d2h/dy2, s = d2h/dxdy
    """
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    pad_h = np.pad(dem, 1, mode='edge')
    
    z_nw = pad_h[:-2, :-2]
    z_n  = pad_h[:-2, 1:-1]
    z_ne = pad_h[:-2, 2:]
    z_w  = pad_h[1:-1, :-2]
    z_c  = pad_h[1:-1, 1:-1]
    z_e  = pad_h[1:-1, 2:]
    z_sw = pad_h[2:, :-2]
    z_s  = pad_h[2:, 1:-1]
    z_se = pad_h[2:, 2:]
    
    p = ((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / (8.0 * cell_size)
    q = ((z_sw + 2.0 * z_s + z_se) - (z_nw + 2.0 * z_n + z_ne)) / (8.0 * cell_size)
    
    r = (z_e - 2.0 * z_c + z_w) / (cell_size ** 2)
    t = (z_s - 2.0 * z_c + z_n) / (cell_size ** 2)
    s = ((z_se - z_sw) - (z_ne - z_nw)) / (4.0 * (cell_size ** 2))
    
    return p, q, r, t, s


def calculate_slope(dem: Union[np.ndarray, torch.Tensor], cell_size: float = 30.0) -> np.ndarray:
    """
    Computes slope angle in degrees [0, 90].
    S = arctan(sqrt(p^2 + q^2)) * (180 / pi)
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    p, q, _, _, _ = compute_derivatives_numpy(dem, cell_size=cell_size)
    grad_mag = np.sqrt(p**2 + q**2)
    slope_rad = np.arctan(grad_mag)
    return np.degrees(slope_rad)


def calculate_aspect(dem: Union[np.ndarray, torch.Tensor], cell_size: float = 30.0) -> np.ndarray:
    """
    Computes aspect compass direction in degrees [0, 360).
    Aspect = atan2(-q, -p) mapped to compass degrees.
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    p, q, _, _, _ = compute_derivatives_numpy(dem, cell_size=cell_size)
    aspect_rad = np.arctan2(-q, -p)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)
    return aspect_deg


def calculate_curvatures(
    dem: Union[np.ndarray, torch.Tensor],
    cell_size: float = 30.0
) -> Dict[str, np.ndarray]:
    """
    Calculates Profile Curvature (kp), Planform Curvature (kc), and Laplacian Curvature.
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    p, q, r, t, s = compute_derivatives_numpy(dem, cell_size=cell_size)
    
    p2 = p ** 2
    q2 = q ** 2
    pq = p * q
    denom_grad = p2 + q2
    denom_grad_safe = np.where(denom_grad < 1e-7, 1e-7, denom_grad)
    
    # Profile Curvature (acceleration along steepest descent)
    kp = -(r * p2 + 2.0 * s * pq + t * q2) / (denom_grad_safe * (1.0 + denom_grad_safe) ** 1.5)
    
    # Planform Curvature (contour convergence/divergence)
    kc = -(t * p2 - 2.0 * s * pq + r * q2) / (denom_grad_safe ** 1.5)
    
    # Laplacian Curvature
    laplacian = r + t
    
    return {
        "profile_curvature": kp,
        "planform_curvature": kc,
        "laplacian_curvature": laplacian
    }


def compute_hypsometric_curve(
    dem: Union[np.ndarray, torch.Tensor],
    num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the normalized hypsometric curve and Hypsometric Integral (HI).
    Returns:
        rel_elevations: array of normalized relative heights in [0, 1]
        area_above: relative area of surface above each height in [0, 1]
        hi: Hypsometric Integral (scalar in [0, 1])
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    e_min = float(dem.min())
    e_max = float(dem.max())
    
    if e_max - e_min < 1e-6:
        return np.linspace(0, 1, num_bins), np.ones(num_bins), 0.5
        
    norm_dem = (dem - e_min) / (e_max - e_min)
    
    # Hypsometric Integral = (mean - min) / (max - min)
    hi = float(norm_dem.mean())
    
    rel_elevations = np.linspace(0.0, 1.0, num_bins)
    area_above = np.array([(norm_dem >= h_val).mean() for h_val in rel_elevations])
    
    return rel_elevations, area_above, hi


def evaluate_geomorphic_fidelity(
    real_dems: np.ndarray,
    syn_dems: np.ndarray,
    cell_size: float = 30.0
) -> Dict[str, float]:
    """
    Computes comprehensive geomorphological discrepancy metrics between real and synthetic sets:
    - Slope Earth Mover's Distance (Wasserstein-1)
    - Profile Curvature KS statistic
    - Hypsometric Integral absolute error
    - Elevation Skewness & Kurtosis error
    """
    real_slopes = []
    syn_slopes = []
    real_his = []
    syn_his = []
    real_curvs = []
    syn_curvs = []
    
    for i in range(min(len(real_dems), len(syn_dems))):
        r_dem = real_dems[i]
        s_dem = syn_dems[i]
        
        # Slopes
        real_slopes.append(calculate_slope(r_dem, cell_size).flatten())
        syn_slopes.append(calculate_slope(s_dem, cell_size).flatten())
        
        # Hypsometry
        _, _, r_hi = compute_hypsometric_curve(r_dem)
        _, _, s_hi = compute_hypsometric_curve(s_dem)
        real_his.append(r_hi)
        syn_his.append(s_hi)
        
        # Curvatures
        r_curv = calculate_curvatures(r_dem, cell_size)["profile_curvature"].flatten()
        s_curv = calculate_curvatures(s_dem, cell_size)["profile_curvature"].flatten()
        real_curvs.append(r_curv)
        syn_curvs.append(s_curv)
        
    r_slopes_all = np.concatenate(real_slopes)
    s_slopes_all = np.concatenate(syn_slopes)
    r_curvs_all = np.concatenate(real_curvs)
    s_curvs_all = np.concatenate(syn_curvs)
    
    slope_emd = stats.wasserstein_distance(r_slopes_all, s_slopes_all)
    curv_ks, _ = stats.ks_2samp(r_curvs_all[:20000], s_curvs_all[:20000])
    hi_mae = np.mean(np.abs(np.array(real_his) - np.array(syn_his)))
    
    return {
        "slope_wasserstein_distance": float(slope_emd),
        "curvature_ks_statistic": float(curv_ks),
        "hypsometric_integral_mae": float(hi_mae),
        "mean_real_hi": float(np.mean(real_his)),
        "mean_syn_hi": float(np.mean(syn_his))
    }
