"""
Realistic 3D Topographical Terrain & Continent Studio.
Desktop GUI application designed for realistic geomorphic landscape generation.
Features:
- Vast flat plains, river valleys, and continuous alpine mountain chains.
- Biome selection: Alpine & Plains, Rolling Lowlands, Valley Basin.
- Dedicated Plains Flatness and Mountain Prominence sliders.
- True-to-scale 3D perspective surface rendering (no egg-carton spikes).
- 16-bit PNG Heightmap and Wavefront 3D OBJ export.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LightSource
from matplotlib import cm
from PIL import Image

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.models.generator import TerrainGenerator
from src.metrics.geomorphology import calculate_slope, compute_hypsometric_curve
from src.metrics.spectral import compute_rapsd
from src.utils.realistic_terrain import RealisticLandscapeEngine


class RealisticTerrainStudio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Realistic 3D Topographical Studio (Plains & Mountain Continents)")
        self.root.geometry("1480x920")
        self.root.minsize(1180, 750)

        # Style
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Generator & Engine
        self.generator: Optional[TerrainGenerator] = None
        self.engine: Optional[RealisticLandscapeEngine] = None
        self._init_models()

        # State Variables
        self.world_size = tk.IntVar(value=512)
        self.biome = tk.StringVar(value="plains_mountains")
        self.plains_flatness = tk.DoubleVar(value=0.60)
        self.mountain_prominence = tk.DoubleVar(value=0.75)
        self.vertical_scale = tk.DoubleVar(value=22.0)
        self.sun_azimuth = tk.DoubleVar(value=315.0)
        self.sun_altitude = tk.DoubleVar(value=45.0)
        self.colormap_name = tk.StringVar(value="gist_earth")
        self.render_mode = tk.StringVar(value="Surface Shaded")
        self.seed_var = tk.IntVar(value=42)

        # Active Data
        self.dem = np.zeros((512, 512), dtype=np.float32)
        self.meta = {}

        # Build UI
        self._build_ui()

        # Initial synthesis
        self.generate_landscape()

    def _init_models(self):
        """Loads WGAN-GP generator weights."""
        self.generator = TerrainGenerator(latent_dim=128, image_size=128)
        weights_path = Path("weights/wgan_terrain_final.pt")

        if weights_path.exists():
            try:
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = ckpt["generator"] if "generator" in ckpt else ckpt
                self.generator.load_state_dict(state_dict)
                self.generator.eval()
                self.engine = RealisticLandscapeEngine(self.generator)
                self.status_msg = "WGAN-GP Neural Generator: ACTIVE (weights/wgan_terrain_final.pt)"
            except Exception as e:
                self.engine = RealisticLandscapeEngine(None)
                self.status_msg = f"Generator load warning ({e}); using procedural engine"
        else:
            self.engine = RealisticLandscapeEngine(None)
            self.status_msg = "Running procedural fallback engine"

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # ---------------- Left Control Sidebar ----------------
        sidebar = ttk.Frame(main_paned, width=390, padding=12)
        main_paned.add(sidebar, weight=0)

        ttk.Label(sidebar, text="Realistic Terrain Studio", font=("Helvetica", 15, "bold")).pack(anchor=tk.W)
        self.status_lbl = ttk.Label(sidebar, text=self.status_msg, foreground="#007acc", font=("Helvetica", 8))
        self.status_lbl.pack(anchor=tk.W, pady=(0, 10))

        # --- Section 1: Biome & Continental Scale ---
        box_biome = ttk.LabelFrame(sidebar, text="1. Continental Scale & Biome", padding=10)
        box_biome.pack(fill=tk.X, pady=4)

        ttk.Label(box_biome, text="Geological Biome:").pack(anchor=tk.W)
        biome_menu = ttk.Combobox(
            box_biome,
            textvariable=self.biome,
            values=[
                "plains_mountains (Alpine Range & Vast Plains)",
                "rolling_plains (Lowland Plains & River Valleys)",
                "valley_basin (Central Flat Basin & Hills)"
            ],
            state="readonly"
        )
        biome_menu.pack(fill=tk.X, pady=2)
        biome_menu.current(0)

        ttk.Label(box_biome, text="Map Resolution / Coverage:").pack(anchor=tk.W, pady=(6, 1))
        size_frame = ttk.Frame(box_biome)
        size_frame.pack(fill=tk.X, pady=2)
        for val, txt in [(256, "256² (~60 km²)"), (512, "512² (~235 km²)"), (768, "768² (~530 km²)")]:
            ttk.Radiobutton(size_frame, text=txt, variable=self.world_size, value=val).pack(anchor=tk.W)

        # --- Section 2: Plains vs Mountains Balance ---
        box_balance = ttk.LabelFrame(sidebar, text="2. Plains & Relief Balance", padding=10)
        box_balance.pack(fill=tk.X, pady=4)

        ttk.Label(box_balance, text="Plains Flatness / Lowland Extent:").pack(anchor=tk.W)
        ttk.Scale(box_balance, from_=0.2, to=0.9, variable=self.plains_flatness).pack(fill=tk.X, pady=2)

        ttk.Label(box_balance, text="Mountain Prominence:").pack(anchor=tk.W, pady=(6, 1))
        ttk.Scale(box_balance, from_=0.3, to=1.0, variable=self.mountain_prominence).pack(fill=tk.X, pady=2)

        # Action Buttons
        btn_box = ttk.Frame(box_balance)
        btn_box.pack(fill=tk.X, pady=8)
        ttk.Button(btn_box, text="Synthesize New Landscape", command=self.generate_landscape).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_box, text="Random Seed", command=self._randomize_seed).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # --- Section 3: 3D Perspective & Shading ---
        box_3d = ttk.LabelFrame(sidebar, text="3. 3D Perspective & Sunlight", padding=10)
        box_3d.pack(fill=tk.X, pady=4)

        ttk.Label(box_3d, text="Vertical Exaggeration:").pack(anchor=tk.W)
        ttk.Scale(box_3d, from_=5.0, to=50.0, variable=self.vertical_scale, command=lambda _: self.render_views()).pack(fill=tk.X, pady=2)

        style_row = ttk.Frame(box_3d)
        style_row.pack(fill=tk.X, pady=4)
        ttk.Label(style_row, text="Palette:").pack(side=tk.LEFT)
        cmap_menu = ttk.Combobox(style_row, textvariable=self.colormap_name, values=["gist_earth", "terrain", "copper", "bone", "viridis"], state="readonly", width=10)
        cmap_menu.pack(side=tk.LEFT, padx=3)
        cmap_menu.bind("<<ComboboxSelected>>", lambda _: self.render_views())

        ttk.Label(style_row, text="Mode:").pack(side=tk.LEFT, padx=(6, 0))
        mode_menu = ttk.Combobox(style_row, textvariable=self.render_mode, values=["Surface Shaded", "Wireframe"], state="readonly", width=12)
        mode_menu.pack(side=tk.LEFT, padx=3)
        mode_menu.bind("<<ComboboxSelected>>", lambda _: self.render_views())

        ttk.Label(box_3d, text="Sun Azimuth (Lighting Angle):").pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(box_3d, from_=0.0, to=360.0, variable=self.sun_azimuth, command=lambda _: self.render_views(update_3d=False)).pack(fill=tk.X, pady=2)

        # --- Section 4: Export Tools ---
        box_exp = ttk.LabelFrame(sidebar, text="4. Export 3D Mesh & Heightmaps", padding=10)
        box_exp.pack(fill=tk.X, pady=4)

        ttk.Button(box_exp, text="Export 3D Model (.OBJ)", command=self.export_obj).pack(fill=tk.X, pady=2)
        ttk.Button(box_exp, text="Export 16-bit Heightmap (.PNG)", command=self.export_heightmap).pack(fill=tk.X, pady=2)

        # Live Geomorphic Metrics
        self.stats_lbl = ttk.Label(sidebar, text="Calculating Geomorphometrics...", font=("Consolas", 8), relief=tk.SOLID, padding=6)
        self.stats_lbl.pack(fill=tk.X, pady=8)

        # ---------------- Right Dual Viewport ----------------
        display_frame = ttk.Frame(main_paned)
        main_paned.add(display_frame, weight=1)

        self.fig = plt.Figure(figsize=(11, 7), dpi=100, facecolor="#f8f9fa")
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_3d = self.fig.add_subplot(122, projection="3d")
        self.fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.94, wspace=0.10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, display_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- Synthesis & Rendering ----------------
    def _randomize_seed(self):
        self.seed_var.set(int(np.random.randint(1, 999999)))
        self.generate_landscape()

    def generate_landscape(self):
        """Generates realistic continental landscape with genuine plains."""
        size = self.world_size.get()
        biome_raw = self.biome.get().split()[0]
        p_flat = self.plains_flatness.get()
        m_prom = self.mountain_prominence.get()
        seed = self.seed_var.get()

        self.status_lbl.config(text=f"Synthesizing {size}x{size} Continent ({biome_raw})...")
        self.root.update_idletasks()

        if self.engine is not None:
            dem, meta = self.engine.synthesize_landscape(
                size=size,
                biome=biome_raw,
                plains_flatness=p_flat,
                mountain_prominence=m_prom,
                seed=seed
            )
            self.dem = dem
            self.meta = meta
            self.status_lbl.config(text=f"Generated {size}x{size} Continent ({meta['total_area_sq_km']:.1f} km², {meta['plain_area_percent']:.1f}% Plains)")
        else:
            messagebox.showerror("Error", "Landscape Engine not initialized.")

        self.render_views()

    def render_views(self, update_3d=True):
        dem = self.dem
        H, W = dem.shape
        v_scale = self.vertical_scale.get()
        az = self.sun_azimuth.get()
        alt = self.sun_altitude.get()
        cmap_name = self.colormap_name.get()
        cmap = getattr(cm, cmap_name, cm.gist_earth)

        # 1. Viewport 1: 2D Shaded Relief
        self.ax_2d.clear()
        ls = LightSource(azdeg=az, altdeg=alt)
        shaded = ls.shade(dem, cmap=cmap, blend_mode="overlay", vert_exag=1.5)
        self.ax_2d.imshow(shaded)
        km_w = (W * 30.0) / 1000.0
        km_h = (H * 30.0) / 1000.0
        self.ax_2d.set_title(f"2D Shaded Relief ({km_w:.1f} x {km_h:.1f} km)", fontsize=11, fontweight="bold")
        self.ax_2d.axis("off")

        # 2. Viewport 2: 3D Perspective Surface Mesh
        if update_3d:
            elev = self.ax_3d.elev if hasattr(self.ax_3d, "elev") and self.ax_3d.elev is not None else 35
            azim = self.ax_3d.azim if hasattr(self.ax_3d, "azim") and self.ax_3d.azim is not None else -55
            self.ax_3d.clear()

            # Stride mesh appropriately for smooth 3D display
            stride = max(2, min(H, W) // 90)
            X, Y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
            Z = dem[::stride, ::stride] * v_scale

            # Set realistic geological aspect ratio (Z vertical is only a fraction of X/Y!)
            # This eliminates the spiky egg-carton look!
            z_ratio = max(0.12, min(0.35, v_scale / 100.0))
            try:
                self.ax_3d.set_box_aspect((1, 1, z_ratio))
            except Exception:
                pass

            mode = self.render_mode.get()
            if mode == "Wireframe":
                self.ax_3d.plot_wireframe(X, Y, Z, color="#444444", linewidth=0.4)
            else:
                self.ax_3d.plot_surface(X, Y, Z, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)

            self.ax_3d.view_init(elev=elev, azim=azim)
            self.ax_3d.set_zlim(0, max(v_scale * 1.1, 10))
            self.ax_3d.set_title("True-to-Scale 3D Landscape (Rotate with Mouse)", fontsize=11, fontweight="bold")
            self.ax_3d.set_xlabel("X (km)", fontsize=7)
            self.ax_3d.set_ylabel("Y (km)", fontsize=7)
            self.ax_3d.set_zlabel("Elev", fontsize=7)
            self.ax_3d.grid(False)

        self.canvas.draw_idle()
        self._update_metrics()

    def _update_metrics(self):
        H, W = self.dem.shape
        slopes = calculate_slope(self.dem, cell_size=30.0)
        _, _, hi = compute_hypsometric_curve(self.dem)
        _, _, beta, _ = compute_rapsd(self.dem)

        km_w = (W * 30.0) / 1000.0
        km_h = (H * 30.0) / 1000.0
        area = km_w * km_h
        plains_pct = float(np.mean(self.dem < 0.25) * 100.0)

        stats_text = (
            f"Coverage    : {km_w:.1f} x {km_h:.1f} km ({area:.1f} sq km)\n"
            f"Plains Area : {plains_pct:.1f}% Flat Lowlands / Valleys\n"
            f"Mean Slope  : {np.mean(slopes):.2f}° (Max Ridge: {np.max(slopes):.1f}°)\n"
            f"Hypsometry  : HI = {hi:.4f}  |  Fractal Scale β = {beta:.2f}"
        )
        self.stats_lbl.config(text=stats_text)

    # ---------------- Exporting ----------------
    def export_obj(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("3D Wavefront Mesh", "*.obj")],
            initialfile="realistic_terrain_mesh.obj"
        )
        if not file_path:
            return

        H, W = self.dem.shape
        v_scale = self.vertical_scale.get()
        stride = 2 if max(H, W) > 300 else 1
        sub_dem = self.dem[::stride, ::stride]
        sH, sW = sub_dem.shape

        try:
            with open(file_path, "w") as f:
                f.write(f"# Realistic Topographical 3D Terrain ({W}x{H})\n")
                for r in range(sH):
                    for c in range(sW):
                        f.write(f"v {c * stride * 30.0 / 1000.0:.3f} {sub_dem[r, c] * v_scale:.3f} {r * stride * 30.0 / 1000.0:.3f}\n")
                for r in range(sH - 1):
                    for c in range(sW - 1):
                        v1 = r * sW + c + 1
                        v2 = r * sW + (c + 1) + 1
                        v3 = (r + 1) * sW + (c + 1) + 1
                        v4 = (r + 1) * sW + c + 1
                        f.write(f"f {v1} {v2} {v3} {v4}\n")
            messagebox.showinfo("Export Successful", f"Saved 3D model to:\n{file_path}\n\nYou can now open it in Windows 3D Viewer or Blender!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save OBJ: {e}")

    def export_heightmap(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("16-bit PNG Heightmap", "*.png")],
            initialfile="realistic_heightmap_16bit.png"
        )
        if not file_path:
            return
        try:
            u16 = (self.dem * 65535.0).astype(np.uint16)
            Image.fromarray(u16, mode="I;16").save(file_path)
            messagebox.showinfo("Export Successful", f"Saved 16-bit heightmap to:\n{file_path}\n\nReady for Unreal Engine 5 or Unity Landscape!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save heightmap: {e}")


def launch_studio():
    root = tk.Tk()
    app = RealisticTerrainStudio(root)
    root.mainloop()


if __name__ == "__main__":
    launch_studio()
