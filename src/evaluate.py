"""
Benchmarking and Evaluation Pipeline for Synthetic Terrain Synthesis.
Computes complete quantitative comparisons between:
1. Real Earth Topography (Ground Truth)
2. Proposed WGAN-GP Architecture
3. Procedural Perlin Noise (fBm)
4. Diamond-Square Fractal Midpoint Displacement

Generates publication-grade LaTeX benchmark tables and comparative plots.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from .config import TerrainConfig, default_config
from .models.generator import TerrainGenerator
from .data.dataset import RealDEMDataset, SyntheticDEMDataset, ProceduralTerrainGenerator
from .metrics.geomorphology import evaluate_geomorphic_fidelity, compute_hypsometric_curve, calculate_slope
from .metrics.spectral import compare_spectral_slopes, compute_rapsd
from .metrics.hydrology import evaluate_hydrological_networks
from .metrics.latent_metrics import compute_perceptual_path_length
from .utils.visualization import plot_hypsometric_curves, plot_terrain_grid


def generate_diamond_square(size: int = 128, roughness: float = 0.7, seed: int = None) -> np.ndarray:
    """
    Diamond-Square fractal midpoint displacement baseline.
    """
    if seed is not None:
        np.random.seed(seed)
    # Require N = 2^k + 1
    k = int(np.ceil(np.log2(size - 1)))
    n = (2 ** k) + 1
    
    grid = np.zeros((n, n), dtype=np.float32)
    grid[0, 0] = np.random.rand()
    grid[0, -1] = np.random.rand()
    grid[-1, 0] = np.random.rand()
    grid[-1, -1] = np.random.rand()
    
    step_size = n - 1
    scale = 1.0
    
    while step_size > 1:
        half_step = step_size // 2
        # Diamond step
        for r in range(half_step, n, step_size):
            for c in range(half_step, n, step_size):
                avg = (grid[r - half_step, c - half_step] +
                       grid[r - half_step, c + half_step] +
                       grid[r + half_step, c - half_step] +
                       grid[r + half_step, c + half_step]) / 4.0
                grid[r, c] = avg + np.random.uniform(-scale, scale)
                
        # Square step
        for r in range(0, n, half_step):
            for c in range((r + half_step) % step_size, n, step_size):
                vals = []
                if r >= half_step: vals.append(grid[r - half_step, c])
                if r + half_step < n: vals.append(grid[r + half_step, c])
                if c >= half_step: vals.append(grid[r, c - half_step])
                if c + half_step < n: vals.append(grid[r, c + half_step])
                avg = sum(vals) / len(vals)
                grid[r, c] = avg + np.random.uniform(-scale, scale)
                
        step_size //= 2
        scale *= (2.0 ** -roughness)
        
    res = grid[:size, :size]
    res_min, res_max = res.min(), res.max()
    if res_max - res_min > 1e-6:
        res = (res - res_min) / (res_max - res_min)
    return res.astype(np.float32)


def run_benchmark(
    checkpoint_path: str = None,
    num_samples: int = 100,
    patch_size: int = 128,
    output_dir: str = "outputs/evaluation"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("Running Topographical Geomorphometric & Generative Benchmark")
    print("="*70)
    
    # 1. Obtain Real Reference DEMs
    print(f"Sampling {num_samples} Reference DEM patches...")
    real_dataset = SyntheticDEMDataset(length=num_samples, patch_size=patch_size, erode=True)
    real_dems = np.array([real_dataset[i].squeeze(0).numpy() for i in range(num_samples)])
    
    # 2. Obtain Synthetic WGAN-GP DEMs
    print("Generating WGAN-GP synthetic terrain patches...")
    generator = TerrainGenerator(latent_dim=128, image_size=patch_size, out_channels=1, base_features=64).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
    generator.eval()
    
    with torch.no_grad():
        z_sample = torch.randn(num_samples, 128, device=device)
        syn_tensor = generator(z_sample).cpu()
        syn_dems = syn_tensor.squeeze(1).numpy()
        
    # 3. Obtain Baseline DEMs
    print("Synthesizing Perlin Noise (fBm) baseline patches...")
    perlin_dems = np.array([
        ProceduralTerrainGenerator.generate_fbm(size=patch_size, octaves=6, persistence=0.5, lacunarity=2.0)
        for _ in range(num_samples)
    ])
    
    print("Synthesizing Diamond-Square baseline patches...")
    ds_dems = np.array([
        generate_diamond_square(size=patch_size, roughness=0.7)
        for _ in range(num_samples)
    ])
    
    # 4. Compute Geomorphological Metrics
    models = {
        "Procedural Perlin (fBm)": perlin_dems,
        "Diamond-Square": ds_dems,
        "WGAN-GP (Ours)": syn_dems
    }
    
    results = []
    
    for name, dem_set in models.items():
        print(f"\nEvaluating metrics for: {name}...")
        geom_metrics = evaluate_geomorphic_fidelity(real_dems, dem_set)
        spectral_metrics = compare_spectral_slopes(real_dems, dem_set)
        hydro_metrics = evaluate_hydrological_networks(dem_set)
        
        row = {
            "Model": name,
            "Slope EMD (W1)": f"{geom_metrics['slope_wasserstein_distance']:.4f}",
            "Curvature KS": f"{geom_metrics['curvature_ks_statistic']:.4f}",
            "Hypsometric Error (|ΔHI|)": f"{geom_metrics['hypsometric_integral_mae']:.4f}",
            "Spectral Decay (β)": f"{spectral_metrics['synthetic_spectral_slope_beta']:.2f}",
            "Spectral Diff (|Δβ|)": f"{spectral_metrics['spectral_slope_difference']:.3f}",
            "Sink Density (pits)": f"{hydro_metrics['mean_sink_density']:.4f}",
            "Drainage Density": f"{hydro_metrics['mean_drainage_density']:.4f}"
        }
        results.append(row)
        
    # Measure Latent Path Length on WGAN-GP
    mean_ppl, std_ppl = compute_perceptual_path_length(generator, latent_dim=128, num_samples=50, device=device)
    print(f"\nWGAN-GP Perceptual Path Length (PPL): {mean_ppl:.2f} ± {std_ppl:.2f}")
    
    # Save Comparison Table
    df = pd.DataFrame(results)
    table_csv = output_path / "benchmark_metrics.csv"
    table_latex = output_path / "benchmark_metrics.tex"
    table_md = output_path / "benchmark_metrics.md"
    
    df.to_csv(table_csv, index=False)
    with open(table_latex, "w") as f:
        f.write(df.to_latex(index=False))
    with open(table_md, "w") as f:
        f.write(df.to_markdown(index=False))
        
    print("\nBenchmark Summary Table:")
    print(df.to_markdown(index=False))
    
    # Generate Hypsometric Comparison Curve Plot
    real_curves = [compute_hypsometric_curve(d) for d in real_dems[:15]]
    real_c_data = [(c[0], c[1]) for c in real_curves]
    syn_curves = [compute_hypsometric_curve(d) for d in syn_dems[:15]]
    syn_c_data = [(c[0], c[1]) for c in syn_curves]
    
    plot_hypsometric_curves(
        real_c_data,
        syn_c_data,
        save_path=output_path / "hypsometric_curves_comparison.png"
    )
    
    # Generate Synthetic Sample Grid
    plot_terrain_grid(
        syn_dems[:16],
        save_path=output_path / "synthetic_dems_grid.png",
        title="WGAN-GP Synthesized Topographical Terrains"
    )
    
    print(f"\nAll benchmark figures and tables saved to: {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Topographical Generative Models")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test patches")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size")
    parser.add_argument("--out_dir", type=str, default="outputs/evaluation", help="Output directory")
    args = parser.parse_args()
    
    run_benchmark(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        output_dir=args.out_dir
    )
