 #!/usr/bin/env python3
"""
3-D Voronoi Membrane Generator – v4 (CPU-optimised)
===================================================
Key speed-ups
-------------
* **Numba-parallel Voronoi** – nearest/second-nearest distances computed in C-level
  parallel loops (`prange`) instead of pure-Python or heavy broadcasting.
* **Multi-thread aware** – `--num_threads` lets you saturate the CPU cores available
  on a cluster node (falls back to all logical cores).
* **Minor vector-wise tweaks** – fewer temporary arrays and one-time `sqrt`.

Usage
-----
  python voronoi_membrane_fast.py --img_size 96 --num_threads 32
"""

from __future__ import annotations
import argparse, os, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads
from scipy.ndimage import gaussian_filter, distance_transform_edt

plt.switch_backend("Agg")  # headless servers

# ──────────────────────────────────────────────────────────────────────────────
# Fast Voronoi helpers
# ──────────────────────────────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True)
def _compute_voronoi_numba(seeds: np.ndarray, D: int, H: int, W: int):
    """
    Numba-accelerated scan over voxels.
    Returns boundary_distance (float32) and labels (int32).
    """
    n_seeds = seeds.shape[0]
    nearest = np.full((D, H, W), np.float32(1e30), dtype=np.float32)
    second = np.full((D, H, W), np.float32(1e30), dtype=np.float32)
    labels = np.empty((D, H, W), dtype=np.int32)

    for z in prange(D):
        for y in range(H):
            for x in range(W):
                best, second_best, best_id = 1e30, 1e30, -1
                for s in range(n_seeds):
                    dz = z - seeds[s, 0]
                    dy = y - seeds[s, 1]
                    dx = x - seeds[s, 2]
                    d2 = dz*dz + dy*dy + dx*dx  # squared dist (no sqrt in loop)
                    if d2 < best:
                        second_best = best
                        best = d2
                        best_id = s
                    elif d2 < second_best:
                        second_best = d2
                nearest[z, y, x] = best
                second[z, y, x] = second_best
                labels[z, y, x] = best_id
    # Convert to actual distances once per voxel
    nearest = np.sqrt(nearest, dtype=np.float32)
    second  = np.sqrt(second,  dtype=np.float32)
    return (second - nearest).astype(np.float32), labels

def compute_voronoi_fields(shape: tuple[int, int, int], seeds: np.ndarray):
    D, H, W = shape
    return _compute_voronoi_numba(seeds.astype(np.float32), D, H, W)

# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers (unchanged apart from tiny micro-opts)
# ──────────────────────────────────────────────────────────────────────────────
def generate_voronoi_seeds(shape, num_seeds, *, boundary_buffer=5,
                           min_seed_distance=4., seed=42):
    rng = np.random.default_rng(seed)
    D, H, W = shape
    seeds = []
    attempts, max_attempts = 0, num_seeds * 40
    while len(seeds) < num_seeds and attempts < max_attempts:
        cand = rng.uniform((boundary_buffer,)*3,
                           (D-boundary_buffer, H-boundary_buffer, W-boundary_buffer))
        if not seeds or np.min(np.linalg.norm(np.asarray(seeds)-cand, axis=1)) >= min_seed_distance:
            seeds.append(cand)
        attempts += 1
    if len(seeds) < num_seeds:
        raise RuntimeError("Could not place all seeds – loosen constraints.")
    return np.asarray(seeds, np.float32)

def build_membrane_volume(bdist, *, membrane_thickness, membrane_intensity,
                          cytosol_intensity):
    boundary_mask = bdist <= 0.5
    dist_to_boundary = distance_transform_edt(~boundary_mask)
    membrane_mask = dist_to_boundary <= membrane_thickness
    vol = np.full_like(bdist, cytosol_intensity, dtype=np.float32)
    vol[membrane_mask] = membrane_intensity
    return vol, membrane_mask

def add_organelles(vol, bdist, *, num_organelles, organelle_radius,
                   organelle_intensity, membrane_thickness, safety_margin, seed):
    rng = np.random.default_rng(seed)
    D, H, W = vol.shape
    clearance = membrane_thickness + safety_margin + organelle_radius
    centre_mask = bdist >= clearance
    zc, yc, xc = np.where(centre_mask)
    good = (
        (zc >= organelle_radius) & (zc < D - organelle_radius) &
        (yc >= organelle_radius) & (yc < H - organelle_radius) &
        (xc >= organelle_radius) & (xc < W - organelle_radius)
    )
    centres = np.stack([zc[good], yc[good], xc[good]], axis=1)
    if len(centres) < num_organelles:
        raise RuntimeError("Not enough cytosolic space for organelles.")
    chosen = centres[rng.choice(len(centres), num_organelles, False)]
    r2 = organelle_radius**2
    for cz, cy, cx in chosen:
        z_min, z_max = int(cz - organelle_radius), int(cz + organelle_radius) + 1
        y_min, y_max = int(cy - organelle_radius), int(cy + organelle_radius) + 1
        x_min, x_max = int(cx - organelle_radius), int(cx + organelle_radius) + 1
        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        mask = (zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 <= r2
        vol[z_min:z_max, y_min:y_max, x_min:x_max][mask] = organelle_intensity
    return vol

# ──────────────────────────────────────────────────────────────────────────────
# Generation pipeline
# ──────────────────────────────────────────────────────────────────────────────
def generate_volume(args):
    t0 = time.time()
    shape = (args.img_size,)*3
    seeds = generate_voronoi_seeds(shape, args.num_cells,
                                   min_seed_distance=args.min_seed_distance,
                                   seed=args.seed)
    bdist, _ = compute_voronoi_fields(shape, seeds)

    vol, _ = build_membrane_volume(
        bdist,
        membrane_thickness=args.membrane_thickness,
        membrane_intensity=args.membrane_intensity,
        cytosol_intensity=args.cytosol_intensity,
    )

    if args.num_organelles:
        vol = add_organelles(
            vol, bdist,
            num_organelles=args.num_organelles,
            organelle_radius=args.organelle_radius,
            organelle_intensity=args.organelle_intensity,
            membrane_thickness=args.membrane_thickness,
            safety_margin=args.safety_margin,
            seed=args.seed+99,
        )

    if args.blur_sigma:
        vol = gaussian_filter(vol, args.blur_sigma)

    if args.noise_level:
        rng = np.random.default_rng(args.seed+999)
        vol = np.clip(vol + rng.normal(0, args.noise_level, size=shape), 0, 1)

    return vol, {"elapsed_s": time.time()-t0, "seeds": len(seeds)}

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser("Fast Voronoi membrane generator (CPU)")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_cells", type=int, default=6)
    p.add_argument("--min_seed_distance", type=float, default=14.)
    p.add_argument("--membrane_thickness", type=float, default=2.0)
    p.add_argument("--membrane_intensity", type=float, default=0.25)
    p.add_argument("--cytosol_intensity", type=float, default=0.7)
    p.add_argument("--num_organelles", type=int, default=8)
    p.add_argument("--organelle_radius", type=float, default=5.)
    p.add_argument("--organelle_intensity", type=float, default=0.05)
    p.add_argument("--safety_margin", type=float, default=1.0)
    p.add_argument("--blur_sigma", type=float, default=0.5)
    p.add_argument("--noise_level", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_threads", type=int, default=os.cpu_count(),
                   help="CPU threads for Numba; default = all logical cores")
    return p.parse_args()

def main():
    args = parse()
    set_num_threads(args.num_threads)
    vol, meta = generate_volume(args)
    tag = (f"voronoi_{args.img_size}_cells{args.num_cells}_"
           f"thick{args.membrane_thickness}_org{args.num_organelles}_seed{args.seed}")
    print("✅ done", tag, meta)

if __name__ == "__main__":
    main()
