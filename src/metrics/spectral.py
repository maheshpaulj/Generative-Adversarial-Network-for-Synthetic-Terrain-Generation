"""
Radially Averaged Power Spectral Density (RAPSD) and Fractal Dimension Analysis.
Verifies Mandelbrot scale-invariance (P(f) ~ f^-beta) on 2D Digital Elevation Models.
"""

from typing import Tuple, Dict, Union
import numpy as np
import torch


def compute_rapsd(
    dem: Union[np.ndarray, torch.Tensor],
    cell_size: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Computes the 2D Radially Averaged Power Spectral Density (RAPSD).
    
    Returns:
        radial_freq: Spatial frequencies (cycles / meter)
        power_spectrum: Radially averaged power density
        spectral_slope_beta: Power-law decay exponent (P(k) ~ k^-beta)
        fractal_dimension: Mandelbrot fractal dimension D = (7 - beta) / 2
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    N = dem.shape[0]
    # Remove mean to eliminate DC spike
    dem_zero_mean = dem - np.mean(dem)
    
    # 2D Fast Fourier Transform
    fft2 = np.fft.fftshift(np.fft.fft2(dem_zero_mean))
    power2d = np.abs(fft2) ** 2 / (N * N)
    
    # Grid coordinates
    center = N // 2
    y, x = np.indices((N, N))
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_int = r.astype(np.int32)
    
    # Radial averaging in concentric rings
    max_radius = center
    tbin = np.bincount(r_int.ravel(), power2d.ravel())
    nr = np.bincount(r_int.ravel())
    
    radial_profile = tbin[:max_radius] / np.maximum(nr[:max_radius], 1)
    frequencies = np.arange(1, max_radius) / (N * cell_size)
    powers = radial_profile[1:max_radius]
    
    # Safe log-log linear regression in inertial range (skip low and very high noisy freqs)
    valid_idx = (frequencies > 0) & (powers > 1e-12)
    if np.sum(valid_idx) > 5:
        log_k = np.log10(frequencies[valid_idx])
        log_p = np.log10(powers[valid_idx])
        
        # Trim extreme endpoints
        start_cut = int(0.1 * len(log_k))
        end_cut = int(0.85 * len(log_k))
        if end_cut - start_cut > 3:
            log_k = log_k[start_cut:end_cut]
            log_p = log_p[start_cut:end_cut]
            
        slope, intercept = np.polyfit(log_k, log_p, 1)
        beta = -slope  # Convention: P(k) ~ k^-beta
        fractal_dim = (7.0 - beta) / 2.0
    else:
        beta = 0.0
        fractal_dim = 2.0
        
    return frequencies, powers, float(beta), float(fractal_dim)


def compare_spectral_slopes(
    real_dems: np.ndarray,
    syn_dems: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates spectral scale invariance fidelity across datasets.
    """
    real_betas = []
    syn_betas = []
    
    for i in range(min(len(real_dems), len(syn_dems))):
        _, _, r_beta, _ = compute_rapsd(real_dems[i])
        _, _, s_beta, _ = compute_rapsd(syn_dems[i])
        real_betas.append(r_beta)
        syn_betas.append(s_beta)
        
    r_mean = float(np.mean(real_betas))
    s_mean = float(np.mean(syn_betas))
    beta_diff = abs(r_mean - s_mean)
    
    return {
        "real_spectral_slope_beta": r_mean,
        "synthetic_spectral_slope_beta": s_mean,
        "spectral_slope_difference": beta_diff
    }
