# Synthetic Topographical Terrain Synthesis via Deep Convolutional Generative Adversarial Networks and Latent Manifold Interpolation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Academic Project & Research Manuscript Draft**: An end-to-end framework for high-fidelity Digital Elevation Model (DEM) terrain generation utilizing Wasserstein GAN with Gradient Penalty (WGAN-GP), Spectral Normalization, and a Multi-Scale PatchGAN critic.

---

## 📖 Table of Contents
1. [Overview & Abstract](#-overview--abstract)
2. [Key Innovations: What Makes Us Better?](#-key-innovations-what-makes-us-better)
3. [Comparative Taxonomy](#-comparative-taxonomy)
4. [Repository Architecture](#-repository-architecture)
5. [Documentation & Research Paper](#-documentation--research-paper)
6. [Quick Start: Google Colab (Recommended)](#-quick-start-google-colab-recommended)
7. [Local Usage](#-local-usage)
8. [Geomorphological Evaluation Protocol](#-geomorphological-evaluation-protocol)
9. [Citation](#-citation)

---

## 🌌 Overview & Abstract

Procedural generation algorithms (Perlin noise, Simplex noise, Diamond-Square) have historically powered terrain generation in computer graphics and simulation engines. However, procedural algorithms struggle to synthesize complex macro-geological structures, dendritic drainage basins, and realistic multi-scale erosion patterns without tedious manual parameter tuning. 

This repository implements an end-to-end framework that models terrain synthesis as continuous probabilistic density estimation over Gaussian latent manifolds $\mathcal{Z} \sim \mathcal{N}(0, I_d)$, mapping low-dimensional stochastic vectors directly to high-resolution 2D Digital Elevation Models (DEMs).

---

## 🚀 Key Innovations: What Makes Us Better?

| Research Dimension | Traditional Procedural | Prior GAN Works (*Guérin et al. 2017*, *D-SRGAN*) | **Our Proposed Framework** |
| :--- | :--- | :--- | :--- |
| **Generative Paradigm** | Deterministic formulas (fBm) | Conditional sketch-to-DEM or Super-Resolution | **Unconditional stochastic latent manifold mapping** |
| **Training Stability** | N/A | Heuristic GAN loss (frequent mode collapse) | **1-Lipschitz WGAN-GP ($\lambda=10$) + Spectral Normalization** |
| **Artifact Suppression** | Rectilinear grid creasing | Checkerboard artifacts from transposed convs | **Bilinear upsampling + Conv guarantees smooth 2nd derivatives** |
| **Macro/Micro Balance** | Single frequency band tuning | Single receptive field discriminator | **Multi-Scale PatchGAN ($1\times, 0.5\times$)** |
| **Evaluation Standard** | Visual aesthetics only | Standard ImageNet Inception FID | **Dual-Domain: FID-DEM + 7 Geomorphometric & Hydrological Metrics** |
| **Latent Morphing** | Non-existent (non-invertible) | Impossible (sketch-conditioned) | **Continuous geodesic spherical interpolation (slerp) with low PPL** |

---

## 📊 Comparative Taxonomy

```
                                Terrain Synthesis Paradigms
                                            │
        ┌───────────────────────────────────┼──────────────────────────────────┐
        ▼                                   ▼                                  ▼
1. Procedural & Fractal            2. Physical Erosion               3. Deep Generative Models
   • Diamond-Square (1986)            • Hydraulic Erosion (Musgrave)    • Conditional Sketch GAN (Guérin 2017)
   • Perlin / Simplex Noise           • Stream Power PDE (Benes 2008)   • Super-Resolution (D-SRGAN, TUGAN)
   • Non-geomorphic bumps             • Slow O(N^3), non-invertible     • Unconditional WGAN-GP (Ours)
```

---

## 📂 Repository Architecture

```
AI Engines/
├── docs/
│   ├── literature_review_and_comparative_analysis.md   # Systematic literature survey & comparative analysis
│   ├── geomorphological_metrics_spec.md               # Mathematical definitions of all topographical metrics
│   └── research_paper_draft.md                         # Complete publication-ready IEEE/ACM style paper
├── notebooks/
│   └── WGAN_GP_Terrain_Synthesis_Colab.ipynb          # Turnkey Google Colab notebook (1-click train & eval)
├── src/
│   ├── config.py                                       # Central hyperparameters & directory configurations
│   ├── models/
│   │   ├── generator.py                                # Spectral-Normalized Conv Generator
│   │   ├── discriminator.py                            # Multi-Scale PatchGAN Critic
│   │   └── wgan_gp.py                                  # Gradient penalty and Wasserstein losses
│   ├── data/
│   │   ├── dataset.py                                  # Real DEM loader & procedural eroded fallback dataset
│   │   └── dem_downloader.py                           # Pre-generation and caching utilities
│   ├── metrics/
│   │   ├── geomorphology.py                            # Slope, Aspect, Curvatures, Hypsometric curves & HI
│   │   ├── spectral.py                                 # Radially Averaged Power Spectral Density (RAPSD)
│   │   ├── hydrology.py                                # D8 flow routing & drainage network analysis
│   │   ├── fid_dem.py                                  # Fréchet Inception Distance adapted for DEMs
│   │   └── latent_metrics.py                           # Slerp geodesic interpolation & PPL
│   ├── utils/
│   │   ├── visualization.py                            # Lambertian hillshading, 3D meshes, hypsometric plots
│   │   └── export_tools.py                             # 16-bit PNG heightmaps, OBJ 3D meshes, GeoTIFF
│   ├── train.py                                        # Standalone training script
│   └── evaluate.py                                     # Benchmark suite comparing models & baselines
├── tests/
│   ├── test_models.py                                  # Unit tests for generator and critic
│   └── test_metrics.py                                 # Unit tests for geomorphic math
├── requirements.txt                                    # Project dependencies
└── README.md
```

---

## 📚 Documentation & Research Paper

Full documentation is provided inside the `docs/` folder:
- **[Literature Review & Comparative Analysis](docs/literature_review_and_comparative_analysis.md)**: Exhaustive breakdown of prior literature, mathematical foundations of WGAN-GP, and detailed comparative tables.
- **[Geomorphological Metrics Specification](docs/geomorphological_metrics_spec.md)**: Complete mathematical formulas and physical interpretations of slope, aspect, profile/planform curvature, hypsometric integral, power spectral density, and D8 flow routing.
- **[Research Paper Draft](docs/research_paper_draft.md)**: Ready-to-publish academic manuscript formatted with IEEE/ACM sections, LaTeX equations, experimental tables, and citations.

---

## ⚡ Quick Start: Google Colab (Recommended)

Since training is intended for Google Colab GPUs (free T4 or A100):
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com/)
2. Select **Upload** and upload `notebooks/WGAN_GP_Terrain_Synthesis_Colab.ipynb`.
3. Set the runtime to **GPU** (`Runtime` $\to$ `Change runtime type` $\to$ `T4 GPU`).
4. Click **Run All**. The notebook will:
   - Generate synthetic eroded DEM patches automatically.
   - Train the WGAN-GP model with real-time progress bars and loss plots.
   - Display hillshaded shaded relief maps and interactive 3D perspective surface meshes.
   - Run the full geomorphological benchmark suite and output LaTeX comparison tables.
   - Export 16-bit PNG heightmaps for game engines (Unreal/Unity).

---

## 💻 Local Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m unittest discover tests
```

### 3. Launch Interactive 3D Terrain & Heightmap Studio
Launch the desktop GUI to interactively orbit, morph, and inspect 3D terrains generated by your trained weights:
```bash
python terrain_viewer.py
```
- Click & drag to rotate 3D view in 360°.
- Sliders for Vertical Exaggeration, Colormaps (`terrain`, `gist_earth`, `copper`), and Solar Shading Azimuth.
- Real-time Geodesic Latent Slerp Morphing slider.
- One-click export to 3D `.OBJ` mesh and 16-bit `.PNG` heightmap.

### 4. Pre-generate Cached Training Patches (Optional)
```bash
python -m src.data.dem_downloader --out_dir data/processed --num_samples 500
```

### 4. Train Locally
```bash
python -m src.train --epochs 50 --batch_size 32 --image_size 128
```

### 5. Run Benchmark Evaluation
```bash
python -m src.evaluate --num_samples 50 --patch_size 128
```

---

## 📐 Geomorphological Evaluation Protocol

Our framework evaluates synthetic terrains across 8 rigorous geophysical criteria:
1. **Fréchet Inception Distance ($\text{FID}_{\text{DEM}}$)**: Measures distribution alignment using a 3-channel composite $[h, \text{hillshade}, \text{slope}]$.
2. **Slope Earth Mover's Distance ($W_1(S)$)**: Verifies first derivative gravitational slope plausibility.
3. **Aspect Circular Variance ($V_\theta$)**: Detects non-physical cardinal axis grid bias.
4. **Curvature Continuity ($\kappa_p, \kappa_c$)**: Profile curvature (runoff acceleration) and planform curvature (flow convergence).
5. **Hypsometric Integral (HI) & Curves**: Validates tectonic uplift and landscape erosion developmental stages.
6. **Radially Averaged Power Spectral Density (RAPSD)**: Verifies Mandelbrot fractal scale-invariance ($P(f) \propto f^{-\beta}$, $\beta \approx 2.3$).
7. **Hydrological Flow Routing (D8)**: Measures sink/depression pit density ($\rho_{\text{sink}}$) and dendritic drainage network density.
8. **Perceptual Path Length (PPL)**: Verifies smooth geodesic latent navigation via spherical linear interpolation (slerp).

---

## 📜 Citation

```bibtex
@article{mahesh2026synthetic,
  title={Synthetic Topographical Terrain Synthesis via Deep Convolutional Generative Adversarial Networks and Latent Manifold Interpolation},
  author={Mahesh and collaborators},
  journal={Project Research Manuscript},
  year={2026}
}
```
