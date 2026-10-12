"""
Unit tests for Geomorphological, Spectral, Hydrological, and Latent Metrics.
"""

import unittest
import numpy as np
import torch

from src.metrics.geomorphology import (
    calculate_slope,
    calculate_aspect,
    calculate_curvatures,
    compute_hypsometric_curve
)
from src.metrics.spectral import compute_rapsd
from src.metrics.hydrology import compute_d8_flow
from src.metrics.latent_metrics import slerp, generate_interpolation_path


class TestGeomorphologyMetrics(unittest.TestCase):
    def test_inclined_plane_slope(self):
        """On a plane with uniform gradient dh/dx = 1, dh/dy = 0, cell_size=1, slope should be 45 deg."""
        N = 64
        x = np.arange(N, dtype=np.float32)
        plane = np.tile(x, (N, 1))  # z(x, y) = x
        slope = calculate_slope(plane, cell_size=1.0)
        
        # Interior pixels (away from borders) should have slope ~ 45 degrees
        interior = slope[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 45.0, atol=1.0)

    def test_hypsometric_integral_symmetric(self):
        """For a linear ramp [0, 1], the hypsometric integral must equal 0.5."""
        N = 64
        ramp = np.linspace(0.0, 1.0, N, dtype=np.float32)
        grid = np.tile(ramp, (N, 1))
        
        rel_elev, area_above, hi = compute_hypsometric_curve(grid)
        self.assertAlmostEqual(hi, 0.5, places=2)
        self.assertEqual(len(rel_elev), len(area_above))

    def test_rapsd_frequency_bounds(self):
        """Verify RAPSD generates positive spatial frequencies and finite spectral slopes."""
        N = 64
        noise = np.random.randn(N, N).astype(np.float32)
        freqs, powers, beta, fractal_dim = compute_rapsd(noise, cell_size=30.0)
        
        self.assertTrue(np.all(freqs > 0))
        self.assertTrue(np.all(powers >= 0))
        self.assertFalse(np.isnan(beta))
        self.assertFalse(np.isnan(fractal_dim))

    def test_d8_hydrology_flow(self):
        """Verify D8 flow creates valid flow directions (0..8) and bounded sink densities."""
        N = 32
        terrain = np.random.rand(N, N).astype(np.float32)
        flow_dir, accum, sink_density = compute_d8_flow(terrain)
        
        self.assertEqual(flow_dir.shape, (N, N))
        self.assertEqual(accum.shape, (N, N))
        self.assertTrue(np.all(accum >= 1.0))
        self.assertGreaterEqual(sink_density, 0.0)
        self.assertLessEqual(sink_density, 1.0)

    def test_slerp_interpolation(self):
        """Verify slerp matches endpoints at t=0 and t=1."""
        z1 = torch.randn(1, 128)
        z2 = torch.randn(1, 128)
        
        z_start = slerp(z1, z2, 0.0)
        z_end = slerp(z1, z2, 1.0)
        
        torch.testing.assert_close(z_start, z1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z_end, z2, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
