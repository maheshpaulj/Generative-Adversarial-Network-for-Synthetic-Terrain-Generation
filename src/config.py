"""
Configuration module for Synthetic Topographical Terrain Synthesis via WGAN-GP.
Centralizes all hyperparameters, architecture options, and filesystem paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class TerrainConfig:
    # Model Architecture
    latent_dim: int = 128
    image_size: int = 128          # 128x128 or 256x256 elevation patches
    channels: int = 1              # Single-channel scalar Digital Elevation Model (DEM)
    gen_features: int = 64         # Base channel multiplier for Generator
    disc_features: int = 64        # Base channel multiplier for Critic
    use_spectral_norm: bool = True # Apply Spectral Normalization to Gen and Critic
    num_scales: int = 2            # Number of scales for Multi-Scale PatchGAN Critic

    # WGAN-GP Hyperparameters
    lambda_gp: float = 10.0        # Gradient penalty weight
    critic_iterations: int = 5     # Number of Critic updates per Generator update
    drift_penalty: float = 0.001   # Minor drift penalty to prevent critic outputs drifting
    
    # Training Parameters
    batch_size: int = 32
    num_epochs: int = 100
    lr_critic: float = 2e-4        # TTUR (Two Time-Scale Update Rule)
    lr_gen: float = 1e-4
    beta1: float = 0.0             # Wasserstein preferred beta1
    beta2: float = 0.9

    # Data Settings
    data_dir: Path = Path("data/processed")
    raw_data_dir: Path = Path("data/raw")
    output_dir: Path = Path("outputs")
    checkpoints_dir: Path = Path("outputs/checkpoints")
    samples_dir: Path = Path("outputs/samples")
    metrics_dir: Path = Path("outputs/metrics")
    
    # Normalization & Physical Units
    elevation_min: float = 0.0     # Normalized scale min
    elevation_max: float = 1.0     # Normalized scale max
    physical_grid_spacing: float = 30.0  # Assumed grid resolution in meters (e.g. 30m SRTM)
    
    # Checkpointing & Logging
    save_interval: int = 10        # Save checkpoint every N epochs
    sample_interval: int = 5       # Generate sample hillshades every N epochs
    log_interval: int = 50         # Log step metrics every N batches
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def setup_directories(self):
        """Ensure all required output and data directories exist."""
        for p in [self.data_dir, self.raw_data_dir, self.output_dir, 
                  self.checkpoints_dir, self.samples_dir, self.metrics_dir]:
            p.mkdir(parents=True, exist_ok=True)


default_config = TerrainConfig()
