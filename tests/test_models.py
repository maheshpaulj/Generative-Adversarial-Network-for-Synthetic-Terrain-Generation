"""
Unit tests for Generator, MultiScalePatchGANCritic, and WGAN-GP Gradient Penalty.
"""

import unittest
import torch
from src.models.generator import TerrainGenerator
from src.models.discriminator import MultiScalePatchGANCritic, PatchGANCritic
from src.models.wgan_gp import compute_gradient_penalty, critic_loss, generator_loss


class TestTerrainModels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.latent_dim = 128
        self.image_size = 128
        self.batch_size = 4

    def test_generator_forward(self):
        """Verify generator produces correct tensor shape and [0, 1] bounded elevations."""
        gen = TerrainGenerator(
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            out_channels=1,
            base_features=32,
            use_spectral_norm=True
        ).to(self.device)
        
        z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        out = gen(z)
        
        self.assertEqual(out.shape, (self.batch_size, 1, self.image_size, self.image_size))
        self.assertTrue(torch.all(out >= 0.0), "Elevations must be >= 0.0")
        self.assertTrue(torch.all(out <= 1.0), "Elevations must be <= 1.0")

    def test_multi_scale_critic_forward(self):
        """Verify MultiScalePatchGANCritic outputs multi-scale prediction lists."""
        critic = MultiScalePatchGANCritic(
            in_channels=1,
            base_features=32,
            num_scales=2,
            use_spectral_norm=True
        ).to(self.device)
        
        x = torch.rand(self.batch_size, 1, self.image_size, self.image_size, device=self.device)
        preds = critic(x)
        
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 2)
        # Scale 1 shape
        self.assertEqual(preds[0].shape[0], self.batch_size)
        self.assertEqual(preds[0].shape[1], 1)
        # Scale 2 should be smaller spatial dimensions
        self.assertLess(preds[1].shape[2], preds[0].shape[2])

    def test_gradient_penalty_computation(self):
        """Verify Gradient Penalty computes a valid scalar gradient tensor."""
        critic = MultiScalePatchGANCritic(
            in_channels=1,
            base_features=16,
            num_scales=2,
            use_spectral_norm=False
        ).to(self.device)
        
        real = torch.rand(self.batch_size, 1, self.image_size, self.image_size, device=self.device)
        fake = torch.rand(self.batch_size, 1, self.image_size, self.image_size, device=self.device)
        
        gp = compute_gradient_penalty(critic, real, fake, self.device)
        self.assertTrue(torch.is_tensor(gp))
        self.assertEqual(gp.ndim, 0)
        self.assertFalse(torch.isnan(gp))


if __name__ == "__main__":
    unittest.main()
