# Mathematical Specification of Geomorphological and Generative Metrics
## Rigorous Evaluation Suite for Synthetic Digital Elevation Models (DEMs)

---

### 1. Introduction

Evaluating synthetic Digital Elevation Models (DEMs) requires metrics that go beyond standard RGB image synthesis measures (e.g., PSNR, SSIM, or standard ImageNet FID). A DEM is a continuous scalar elevation field:
\[
h: \Omega \subset \mathbb{R}^2 \to \mathbb{R}, \quad (x, y) \mapsto h(x, y)
\]
A physically plausible terrain must satisfy differential geomorphic constraints, scale-invariant power spectra, and hydrological flow conservation laws. 

This document defines the mathematical formulation, discrete computational kernels, and physical interpretations of all evaluation metrics implemented in our framework.

---

### 2. Differential Geomorphometric Metrics

Terrain morphology is primarily characterized by first- and second-order spatial partial derivatives of the elevation surface.

Given a uniformly sampled discrete elevation grid \(H \in \mathbb{R}^{M \times N}\) with grid spacing \(\Delta x = \Delta y = \delta\), local derivatives are computed using a \(3 \times 3\) window centered at \((i, j)\):

```
┌───────┬───────┬───────┐
│ z_nw  │ z_n   │ z_ne  │
├───────┼───────┼───────┤
│ z_w   │ z_c   │ z_e   │
├───────┼───────┼───────┤
│ z_sw  │ z_s   │ z_se  │
└───────┴───────┴───────┘
```

Using Horn's weighted finite difference operator (robust to high-frequency noise):
\[
p = \frac{\partial h}{\partial x} \approx \frac{(z_{ne} + 2 z_e + z_{se}) - (z_{nw} + 2 z_w + z_{sw})}{8 \delta}
\]
\[
q = \frac{\partial h}{\partial y} \approx \frac{(z_{sw} + 2 z_s + z_{se}) - (z_{nw} + 2 z_n + z_{ne})}{8 \delta}
\]

#### 2.1 Slope (\(S\))
Slope represents the maximum rate of elevation change, corresponding to the gradient magnitude:
\[
S(i, j) = \arctan\left(\sqrt{p^2 + q^2}\right) \quad [\text{radians}], \quad S_{\text{deg}} = S \cdot \frac{180^\circ}{\pi}
\]
*Geomorphic Significance*: Slope governs gravitational mass wasting, sediment shear stress, and water runoff velocity. Natural landscapes exhibit characteristic right-skewed log-normal or Weibull slope distributions.

#### 2.2 Aspect (\(\theta\))
Aspect defines the compass direction of steepest downward slope (direction of gradient vector):
\[
\theta(i, j) = \text{atan2}(-q, -p) \pmod{2\pi} \quad [\text{radians}]
\]
*Geomorphic Significance*: Natural terrains should show smooth directional continuity without non-physical cardinal axis bias (which often appears in procedural grid algorithms like Diamond-Square).

#### 2.3 Curvature Metrics (Second Derivatives)
Second-order partial derivatives are computed as:
\[
r = \frac{\partial^2 h}{\partial x^2}, \quad t = \frac{\partial^2 h}{\partial y^2}, \quad s = \frac{\partial^2 h}{\partial x \partial y}
\]
Discrete approximations:
\[
r \approx \frac{z_e - 2 z_c + z_w}{\delta^2}, \quad t \approx \frac{z_s - 2 z_c + z_n}{\delta^2}, \quad s \approx \frac{(z_{se} - z_{sw}) - (z_{ne} - z_{nw})}{4 \delta^2}
\]

1. **Profile Curvature (\(\kappa_p\))**: Curvature along the direction of maximum slope (steepest descent).
   \[
   \kappa_p = -\frac{r p^2 + 2 s p q + t q^2}{(p^2 + q^2)(1 + p^2 + q^2)^{3/2}}
   \]
   *Geomorphic Significance*: Controls the acceleration or deceleration of surface runoff (erosion vs. deposition zones).
2. **Planform Curvature (\(\kappa_c\))**: Curvature perpendicular to the direction of maximum slope (along the elevation contour).
   \[
   \kappa_c = -\frac{t p^2 - 2 s p q + r q^2}{(p^2 + q^2)^{3/2}}
   \]
   *Geomorphic Significance*: Controls flow convergence (valleys) and divergence (ridges).
3. **Laplacian Curvature (\(\nabla^2 h\))**:
   \[
   \nabla^2 h = \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} = r + t
   \]
   *Geomorphic Significance*: Identifies convexity (\(\nabla^2 h < 0\), ridges/peaks) and concavity (\(\nabla^2 h > 0\), ravines/channels).

---

### 3. Hypsometric Analysis (Area-Elevation Dynamics)

The hypsometric curve describes the distribution of horizontal surface area with respect to elevation, reflecting the geomorphological maturity and stage of landscape erosion (Strahler, 1952).

#### 3.1 Relative Elevation and Area
Let \(h_{\min} = \min(H)\) and \(h_{\max} = \max(H)\). The normalized relative elevation \(h_r\) is:
\[
h_r = \frac{h - h_{\min}}{h_{\max} - h_{\min}} \in [0, 1]
\]
Let \(a(h_r)\) be the relative horizontal area of the landscape above normalized elevation \(h_r\):
\[
a(h_r) = \frac{1}{M N} \sum_{i=1}^M \sum_{j=1}^N \mathbb{I}\left(\frac{H(i, j) - h_{\min}}{h_{\max} - h_{\min}} \ge h_r\right)
\]
The hypsometric curve is the parametric function plotting \(a\) vs. \(h_r\).

#### 3.2 Hypsometric Integral (HI)
The Hypsometric Integral is the area under the hypsometric curve:
\[
HI = \int_0^1 a(h_r) \, dh_r = \frac{\bar{h} - h_{\min}}{h_{\max} - h_{\min}}
\]
where \(\bar{h}\) is the mean elevation of the patch.
*Geomorphic Classification*:
- \(HI > 0.60\): **Youthful Stage** (Deep incised valleys, high plateau retention, tectonic uplift dominance).
- \(0.35 \le HI \le 0.60\): **Mature Equilibrium Stage** (Dynamic equilibrium between fluvial incision and slope wasting).
- \(HI < 0.35\): **Monadnock / Peneplain Stage** (Old-age, heavily eroded, low-relief landscapes).

The model's fidelity is evaluated by the Kolmogorov-Smirnov (KS) distance and Wasserstein distance between real and synthetic hypsometric distributions.

---

### 4. Spectral Scale-Invariance: Radially Averaged Power Spectral Density (RAPSD)

Natural landscapes generated by tectonic uplift and erosion processes exhibit **fractal self-similarity** and scale-invariance across spatial wavelengths (Mandelbrot, 1983).

Given a 2D elevation grid \(H \in \mathbb{R}^{N \times N}\), the 2D Discrete Fourier Transform (DFT) is:
\[
\hat{H}(u, v) = \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} H(x, y) e^{-i \frac{2\pi}{N} (ux + vy)}
\]
The 2D Power Spectrum is:
\[
P(u, v) = \frac{1}{N^2} |\hat{H}(u, v)|^2
\]
The radial spatial frequency \(k_r\) in cycles per unit distance is:
\[
k_r = \sqrt{u^2 + v^2}
\]
The **Radially Averaged Power Spectral Density (RAPSD)** \(S(k_r)\) is obtained by integrating \(P(u, v)\) over annular concentric rings:
\[
S(k) = \frac{1}{N_k} \sum_{k \le \sqrt{u^2 + v^2} < k + \Delta k} P(u, v)
\]
Natural topography obeys a power-law scaling in the inertial range:
\[
S(k) \propto k^{-\beta} \implies \log_{10} S(k) = -\beta \log_{10} k + C
\]
The fractal dimension \(D\) of the 2D surface is related to the spectral decay exponent \(\beta\) by:
\[
D = \frac{7 - \beta}{2}, \quad \text{where typically } 2.0 \le \beta \le 2.8 \implies 2.1 \le D \le 2.5
\]
*Evaluation Criterion*: We fit a linear regression to \(\log S(k)\) vs. \(\log k\) to compute \(\beta_{\text{syn}}\) and compare against \(\beta_{\text{real}}\). Procedural Perlin noise displays steep, unphysical drop-offs outside octave bands, whereas our WGAN-GP reproduces continuous natural power-law decay.

---

### 5. Hydrological Surface Runoff (D8 Routing & Drainage Networks)

Surface water flows strictly along gravitational descent vectors. A realistic synthetic terrain must produce continuous, dendritic drainage networks without unnatural planar trapping voids.

#### 5.1 D8 Flow Direction Algorithm (O'Callaghan & Mark, 1984)
For each grid cell \(c = (i, j)\), the flow direction points to the neighboring cell \(n \in \mathcal{N}_8(c)\) that maximizes the downward hydraulic gradient:
\[
G(c, n) = \frac{H(c) - H(n)}{d(c, n)}
\]
where \(d(c, n) = \delta\) for orthogonal cardinal neighbors and \(d(c, n) = \sqrt{2}\delta\) for diagonal neighbors.

The flow direction index \(F(c)\) is:
\[
F(c) = \arg\max_{n \in \mathcal{N}_8(c)} G(c, n) \quad \text{if } \max_n G(c, n) > 0, \text{ else } 0 \text{ (local sink/pit)}
\]

#### 5.2 Flow Accumulation (\(A\))
Flow accumulation \(A(c)\) represents the total upstream contributing drainage area contributing runoff to cell \(c\):
\[
A(c) = 1 + \sum_{n \in \text{Inflow}(c)} A(n)
\]
This recursive dependency is solved topologically (via queue-based upstream-to-downstream topological sort).

#### 5.3 Stream Network Extraction & Horton-Strahler Bifurcation
Channels are identified where contributing drainage area exceeds a threshold:
\[
\text{Stream}(c) = \begin{cases} 1 & \text{if } A(c) \ge A_{\text{thresh}} \\ 0 & \text{otherwise} \end{cases}
\]
Natural stream networks satisfy **Horton's Law of Stream Numbers**:
\[
\frac{N_\omega}{N_{\omega + 1}} \approx R_B \in [3, 5]
\]
where \(R_B\) is the bifurcation ratio. We quantify the topological plausibility of synthetic stream networks by measuring drainage density \(D_d = \frac{\sum L_{\text{streams}}}{\text{Area}}\) and sink pit density.

---

### 6. Generative Manifold and Perceptual Metrics

#### 6.1 DEM Fréchet Inception Distance (FID-DEM)
Standard FID extracts 2048-dimensional feature representations from an Inception-V3 pool3 layer:
\[
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
\]
To adapt FID for single-channel elevation DEMs, we construct a 3-channel composite tensor \(\Phi(H)\):
1. Channel 1: Normalized elevation \(h_r \in [0, 1]\).
2. Channel 2: Shaded relief hillshade with light source azimuth \(315^\circ\), altitude \(45^\circ\):
   \[
   I_{\text{shade}} = \max\left(0, \cos(\zeta)\cos(S) + \sin(\zeta)\sin(S)\cos(\phi - \theta)\right)
   \]
   where \(\zeta = 90^\circ - 45^\circ = 45^\circ\) (solar zenith angle) and \(\phi = 315^\circ\) (solar azimuth).
3. Channel 3: Normalized gradient magnitude (slope).

This ensures that the feature extractor attends to both elevational distributions and micro-topographical shadows/crests.

#### 6.2 Latent Manifold Smoothness: Perceptual Path Length (PPL)
To prove that the generator maps the continuous Gaussian latent manifold \(\mathcal{Z} \sim \mathcal{N}(0, I_d)\) to a continuous elevation manifold without mode collapses or catastrophic transitions, we evaluate Perceptual Path Length (Karras et al., 2019).

Given two latent endpoints \(z_1, z_2 \sim \mathcal{N}(0, I_d)\), we interpolate via **Spherical Linear Interpolation (slerp)**:
\[
z(t) = \text{slerp}(z_1, z_2; t) = \frac{\sin((1-t)\theta)}{\sin \theta} z_1 + \frac{\sin(t\theta)}{\sin \theta} z_2
\]
The discrete path length over step size \(\epsilon\) is:
\[
\text{PPL} = \mathbb{E}_{z_1, z_2} \left[ \frac{1}{\epsilon^2} d_{\text{DEM}}\left(G(z(t)), G(z(t + \epsilon))\right) \right]
\]
where \(d_{\text{DEM}}(H_1, H_2) = \frac{1}{MN}\|H_1 - H_2\|_1 + \alpha \|\nabla H_1 - \nabla H_2\|_1\). Low PPL values with low variance indicate that the generator performs smooth, physically continuous morphological transitions along the latent manifold.

---

### 7. Consolidated Summary of Evaluation Metrics

| Metric Category | Metric Name | Mathematical Symbol | Ideal Benchmark Behavior |
| :--- | :--- | :--- | :--- |
| **Generative** | Fréchet Inception Distance | \(\text{FID}_{\text{DEM}} \downarrow\) | Lowest score indicating distributional realism. |
| **Generative** | Earth Mover's Distance | \(W_1(P_r, P_g) \downarrow\) | Converges towards zero as distributions align. |
| **Geomorphic 1st Order** | Slope Wasserstein Dist | \(W_1(S_r, S_g) \downarrow\) | Identical gravitational gradient distributions. |
| **Geomorphic 1st Order** | Aspect Circular Variance | \(V_\theta \approx 1\) | Uniform omni-directional orientation (no grid bias). |
| **Geomorphic 2nd Order** | Profile/Plan Curvature KS | \(D_{\text{KS}}(\kappa) \downarrow\) | Smooth transition across ridges and drainage hollows. |
| **Hypsometry** | Hypsometric Integral Error | \(|HI_{\text{real}} - HI_{\text{syn}}| \downarrow\) | Preserves tectonic/erosion developmental stages. |
| **Scale Invariance** | Spectral Slope Error | \(|\beta_{\text{real}} - \beta_{\text{syn}}| \downarrow\) | Adheres to Mandelbrot fractal scaling (\(\beta \in [2.0, 2.5]\)). |
| **Hydrology** | Sink/Pit Density | \(\rho_{\text{sink}} \downarrow\) | Minimal unphysical drainage depressions. |
| **Hydrology** | Drainage Density | \(D_d \approx D_{d, \text{real}}\) | Realistic fluvial stream incision frequency. |
| **Latent Manifold** | Perceptual Path Length | \(\text{PPL} \downarrow\) | Smooth, jerk-free morphing without phase leaps. |
