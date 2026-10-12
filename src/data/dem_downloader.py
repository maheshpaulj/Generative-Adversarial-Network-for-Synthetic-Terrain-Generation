"""
Data fetching and pre-generation utilities for Digital Elevation Models (DEMs).
Provides scripts to synthesize a local cached dataset of high-fidelity eroded DEM patches.
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .dataset import ProceduralTerrainGenerator


def pregenerate_dataset(
    output_dir: Path,
    num_samples: int = 1000,
    patch_size: int = 128,
    erode: bool = True
):
    """
    Pre-generates and caches a dataset of eroded DEM patches on disk (.npy).
    Accelerates training significantly by decoupling erosion simulation from the training loop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} topographical DEM patches of size {patch_size}x{patch_size} in {output_dir}...")
    
    for i in tqdm(range(num_samples), desc="Generating DEM Patches"):
        octaves = np.random.choice([5, 6, 7])
        persistence = np.random.uniform(0.42, 0.58)
        lacunarity = np.random.uniform(1.85, 2.15)
        
        terrain = ProceduralTerrainGenerator.generate_fbm(
            size=patch_size,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        
        if erode:
            droplets = np.random.randint(1200, 2500)
            terrain = ProceduralTerrainGenerator.apply_hydraulic_erosion(terrain, num_droplets=droplets)
            
        file_path = output_dir / f"dem_patch_{i:05d}.npy"
        np.save(file_path, terrain)
        
    print(f"Successfully generated {num_samples} patches in {output_dir}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-generate synthetic topographical DEM dataset.")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of elevation patches to generate")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch resolution (NxN)")
    parser.add_argument("--no_erosion", action="store_true", help="Disable hydraulic erosion simulation")
    args = parser.parse_args()
    
    pregenerate_dataset(
        output_dir=Path(args.out_dir),
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        erode=not args.no_erosion
    )
