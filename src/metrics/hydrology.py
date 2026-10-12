"""
Hydrological Flow Routing and Drainage Analysis Module.
Implements:
- Deterministic 8-Neighbor (D8) steepest descent flow routing
- Flow Accumulation calculation (upstream catchment area)
- Sink / Depression pit density (quantifies unphysical water trapping)
- Stream network extraction and drainage density
"""

from typing import Tuple, Dict, Union
import numpy as np
import torch


def compute_d8_flow(
    dem: Union[np.ndarray, torch.Tensor],
    cell_size: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Executes D8 flow routing algorithm on an elevation surface.
    
    Returns:
        flow_direction: 2D array of neighbor directions (0 = pit, 1..8 = neighbor indices)
        flow_accumulation: 2D array of upstream contributing cell counts
        sink_density: Fraction of internal cells that are non-draining pits/sinks
    """
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    if dem.ndim == 3:
        dem = dem.squeeze()
        
    H, W = dem.shape
    
    # 8 neighbor offsets: (dy, dx, distance_multiplier)
    neighbors = [
        (-1, 0, 1.0),         # N (1)
        (-1, 1, np.sqrt(2)),  # NE (2)
        (0, 1, 1.0),          # E (3)
        (1, 1, np.sqrt(2)),   # SE (4)
        (1, 0, 1.0),          # S (5)
        (1, -1, np.sqrt(2)),  # SW (6)
        (0, -1, 1.0),         # W (7)
        (-1, -1, np.sqrt(2))  # NW (8)
    ]
    
    flow_dir = np.zeros((H, W), dtype=np.int32)
    slopes_max = np.zeros((H, W), dtype=np.float32)
    
    for dir_idx, (dy, dx, dist_factor) in enumerate(neighbors, start=1):
        y_src = max(0, -dy)
        y_dst = min(H, H - dy)
        x_src = max(0, -dx)
        x_dst = min(W, W - dx)
        
        y_nbr_src = max(0, dy)
        y_nbr_dst = min(H, H + dy)
        x_nbr_src = max(0, dx)
        x_nbr_dst = min(W, W + dx)
        
        diff = dem[y_src:y_dst, x_src:x_dst] - dem[y_nbr_src:y_nbr_dst, x_nbr_src:x_nbr_dst]
        drop = diff / (cell_size * dist_factor)
        
        mask = (drop > slopes_max[y_src:y_dst, x_src:x_dst]) & (drop > 0)
        slopes_max[y_src:y_dst, x_src:x_dst] = np.where(mask, drop, slopes_max[y_src:y_dst, x_src:x_dst])
        flow_dir[y_src:y_dst, x_src:x_dst] = np.where(mask, dir_idx, flow_dir[y_src:y_dst, x_src:x_dst])
        
    # Sink density: cells with flow_dir == 0 (excluding boundaries)
    internal_pits = (flow_dir[1:-1, 1:-1] == 0)
    sink_density = float(np.mean(internal_pits))
    
    # Topological sort based flow accumulation
    # Sort cells by elevation in descending order
    flat_indices = np.argsort(-dem.ravel())
    accumulation = np.ones((H, W), dtype=np.float32)
    
    for idx in flat_indices:
        r = idx // W
        c = idx % W
        d = flow_dir[r, c]
        if d > 0:
            dy, dx, _ = neighbors[d - 1]
            nr, nc = r + dy, c + dx
            if 0 <= nr < H and 0 <= nc < W:
                accumulation[nr, nc] += accumulation[r, c]
                
    return flow_dir, accumulation, sink_density


def evaluate_hydrological_networks(
    dems: np.ndarray,
    stream_threshold: float = 50.0
) -> Dict[str, float]:
    """
    Computes average hydrological realism indicators across a batch of DEMs:
    - Average sink pit density
    - Average drainage density (stream length / area)
    """
    sink_densities = []
    drainage_densities = []
    
    for dem in dems:
        _, accum, sink_p = compute_d8_flow(dem)
        sink_densities.append(sink_p)
        
        # Binary stream network
        streams = (accum >= stream_threshold)
        d_density = float(np.mean(streams))
        drainage_densities.append(d_density)
        
    return {
        "mean_sink_density": float(np.mean(sink_densities)),
        "mean_drainage_density": float(np.mean(drainage_densities))
    }
