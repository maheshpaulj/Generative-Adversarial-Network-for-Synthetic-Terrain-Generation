"""
Training pipeline for Synthetic Topographical Terrain Synthesis via WGAN-GP.
Features:
- Two Time-Scale Update Rule (TTUR)
- 1-Lipschitz Gradient Penalty calculation on Multi-Scale PatchGAN Critic
- Periodic visual evaluation with hillshaded relief rendering
- Google Drive / local checkpoint saving
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TerrainConfig, default_config
from .models.generator import TerrainGenerator
from .models.discriminator import MultiScalePatchGANCritic
from .models.wgan_gp import compute_gradient_penalty, critic_loss, generator_loss
from .data.dataset import RealDEMDataset, SyntheticDEMDataset
from .utils.visualization import plot_terrain_grid, plot_3d_terrain, plot_latent_interpolation_strip
from .metrics.latent_metrics import generate_interpolation_path


def train(config: TerrainConfig, resume_checkpoint: str = None):
    # Setup directories
    config.setup_directories()
    
    # Device and Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    print(f"Executing training pipeline on device: {device}")
    
    # Initialize Dataset and DataLoader
    if config.data_dir.exists() and len(list(config.data_dir.glob("*.npy"))) > 0:
        print(f"Loading real DEM dataset from {config.data_dir}...")
        dataset = RealDEMDataset(config.data_dir, patch_size=config.image_size, augment=True)
    else:
        print("No offline DEM rasters found. Using on-the-fly SyntheticDEMDataset (fBm + Hydraulic Erosion)...")
        dataset = SyntheticDEMDataset(length=1200, patch_size=config.image_size, erode=True)
        
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Cross-platform safe
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )
    
    # Initialize Networks
    generator = TerrainGenerator(
        latent_dim=config.latent_dim,
        image_size=config.image_size,
        out_channels=config.channels,
        base_features=config.gen_features,
        use_spectral_norm=config.use_spectral_norm
    ).to(device)
    
    critic = MultiScalePatchGANCritic(
        in_channels=config.channels,
        base_features=config.disc_features,
        num_scales=config.num_scales,
        use_spectral_norm=config.use_spectral_norm
    ).to(device)
    
    # Optimizers with TTUR (Two Time-Scale Update Rule)
    opt_gen = optim.Adam(generator.parameters(), lr=config.lr_gen, betas=(config.beta1, config.beta2))
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic, betas=(config.beta1, config.beta2))
    
    start_epoch = 0
    history = {"w_distance": [], "critic_loss": [], "gen_loss": []}
    
    # Resume Checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        critic.load_state_dict(ckpt["critic"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        opt_critic.load_state_dict(ckpt["opt_critic"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        
    # Fixed evaluation latent vectors
    fixed_z = torch.randn(16, config.latent_dim, device=device)
    fixed_z_start = torch.randn(1, config.latent_dim, device=device)
    fixed_z_end = torch.randn(1, config.latent_dim, device=device)
    
    print("\n" + "="*70)
    print(f"Starting WGAN-GP Topographical Terrain Synthesis Training")
    print(f"Total Epochs: {config.num_epochs} | Batch Size: {config.batch_size} | Image Resolution: {config.image_size}x{config.image_size}")
    print(f"Critic Steps / Gen Step: {config.critic_iterations} | Gradient Penalty Lambda: {config.lambda_gp}")
    print("="*70 + "\n")
    
    step_count = 0
    for epoch in range(start_epoch, config.num_epochs):
        generator.train()
        critic.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config.num_epochs}]")
        epoch_w_dist = []
        epoch_c_loss = []
        epoch_g_loss = []
        
        for real_dems in pbar:
            real_dems = real_dems.to(device)
            batch_sz = real_dems.size(0)
            
            # ----------------------------------------------------
            # 1. Update Critic: maximize E[D(real)] - E[D(fake)] - lambda * GP
            # ----------------------------------------------------
            opt_critic.zero_grad()
            
            z = torch.randn(batch_sz, config.latent_dim, device=device)
            with torch.no_grad():
                fake_dems = generator(z)
                
            d_real = critic(real_dems)
            d_fake = critic(fake_dems)
            
            c_loss, w_dist = critic_loss(d_real, d_fake, drift_penalty=config.drift_penalty)
            gp = compute_gradient_penalty(critic, real_dems, fake_dems, device)
            
            total_critic_loss = c_loss + config.lambda_gp * gp
            total_critic_loss.backward()
            opt_critic.step()
            
            epoch_w_dist.append(w_dist)
            epoch_c_loss.append(total_critic_loss.item())
            
            step_count += 1
            
            # ----------------------------------------------------
            # 2. Update Generator: maximize E[D(G(z))] every n_critic steps
            # ----------------------------------------------------
            if step_count % config.critic_iterations == 0:
                opt_gen.zero_grad()
                z_gen = torch.randn(batch_sz, config.latent_dim, device=device)
                gen_dems = generator(z_gen)
                d_fake_gen = critic(gen_dems)
                
                g_loss = generator_loss(d_fake_gen)
                g_loss.backward()
                opt_gen.step()
                
                epoch_g_loss.append(g_loss.item())
                
            pbar.set_postfix({
                "W-Dist": f"{w_dist:.4f}",
                "C-Loss": f"{total_critic_loss.item():.4f}",
                "G-Loss": f"{epoch_g_loss[-1] if len(epoch_g_loss) > 0 else 0.0:.4f}"
            })
            
        # Record epoch statistics
        avg_w = float(np.mean(epoch_w_dist))
        avg_c = float(np.mean(epoch_c_loss))
        avg_g = float(np.mean(epoch_g_loss)) if epoch_g_loss else 0.0
        history["w_distance"].append(avg_w)
        history["critic_loss"].append(avg_c)
        history["gen_loss"].append(avg_g)
        
        # Periodic sample generation
        if (epoch + 1) % config.sample_interval == 0 or epoch == config.num_epochs - 1:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z).cpu()
                sample_file = config.samples_dir / f"samples_epoch_{epoch+1:04d}.png"
                plot_terrain_grid(samples, save_path=sample_file, title=f"Epoch {epoch+1} Synthetic Terrains")
                
                # Plot 3D surface of first sample
                plot_3d_terrain(samples[0], save_path=config.samples_dir / f"3d_surface_epoch_{epoch+1:04d}.png")
                
                # Latent space morphing sequence
                interp_path = generate_interpolation_path(fixed_z_start, fixed_z_end, num_steps=8, method="slerp")
                interp_dems = generator(interp_path).cpu()
                plot_latent_interpolation_strip(
                    interp_dems,
                    save_path=config.samples_dir / f"morphing_epoch_{epoch+1:04d}.png"
                )
            print(f"Saved visual evaluation samples to {config.samples_dir}")
            
        # Periodic checkpointing
        if (epoch + 1) % config.save_interval == 0 or epoch == config.num_epochs - 1:
            ckpt_path = config.checkpoints_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "critic": critic.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "opt_critic": opt_critic.state_dict(),
                "history": history,
                "config": config
            }, ckpt_path)
            
            # Save latest checkpoint alias
            latest_path = config.checkpoints_dir / "checkpoint_latest.pt"
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "critic": critic.state_dict(),
                "history": history
            }, latest_path)
            print(f"Saved model checkpoint to {ckpt_path}")
            
    print("\nTraining completed successfully.")
    return generator, critic, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WGAN-GP Topographical Terrain Synthesis")
    parser.add_argument("--epochs", type=int, default=default_config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size, help="Batch size")
    parser.add_argument("--lr_gen", type=float, default=default_config.lr_gen, help="Generator learning rate")
    parser.add_argument("--lr_critic", type=float, default=default_config.lr_critic, help="Critic learning rate")
    parser.add_argument("--image_size", type=int, default=default_config.image_size, help="Patch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    cfg = default_config
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr_gen = args.lr_gen
    cfg.lr_critic = args.lr_critic
    cfg.image_size = args.image_size
    
    train(cfg, resume_checkpoint=args.resume)
