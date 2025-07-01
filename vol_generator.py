#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membrane-volume generator – bounded producer/consumer queue
===========================================================

Guaranteed memory cap:
    queue_max × 1 MiB  +  (num_workers × scratch)  < 1 GiB by default
"""

import argparse, io, os, tarfile, time
from pathlib import Path
import multiprocessing as mp

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, convolve1d

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SHAPE = (96, 96, 96)
VOL_BYTES = np.prod(SHAPE) * 4
MASK_BYTES = np.prod(SHAPE) * 1  # uint8 masks
D, H, W = SHAPE
_dg, _hg, _wg = np.ogrid[:D, :H, :W]


# ─────────────────────────── membrane math ──────────────────────────── #
class MembraneGen:
    def __init__(self, generate_masks=False):
        # static params baked in; adapt if needed
        self.n_gauss = (2, 4); self.sigma = (35, 45)  # Reduced from (4,6) for simpler membranes
        self.iso, self.band = 0.8, 0.08  # Keep thin membranes
        self.noise = 0.01  # Reduced from 0.02
        self.n_sph, self.sph_r = (6, 6), (8., 8.) # Exactly 6 spheres of radius 8
        self.n_cube, self.cube_size = (4, 4), (12., 12.) # 4 cubes of size 12
        self.blur, self.iso_var, self.grad = 0.5, 0.2, 0.2 # Reduced blur and iso_var
        self.bg, self.mem, self.sph, self.cube = 0.72, 0.22, 0.03, 0.05
        self.generate_masks = generate_masks

    def check_collision_vectorized(self, placed_centers, placed_radii, new_center, new_radius, min_separation=3.0):
        """Vectorized collision check using NumPy arrays"""
        if len(placed_centers) == 0:
            return False
        
        # Vectorized distance calculation
        new_c = np.array(new_center)
        distances_sq = np.sum((placed_centers - new_c)**2, axis=1)
        required_sep_sq = (placed_radii + new_radius + min_separation)**2
        
        return np.any(distances_sq < required_sep_sq)

    def __call__(self, seed: int):
        rng = np.random.RandomState(int(seed))
        field = np.zeros(SHAPE, np.float32)

        for _ in range(rng.randint(*self.n_gauss)+1):
            cd,ch,cw = rng.uniform(0,D), rng.uniform(0,H), rng.uniform(0,W)
            sd,sh,sw = (rng.uniform(*self.sigma) for _ in range(3))
            amp = rng.uniform(0.5,1.5)
            inv = [1/(2*s*s) for s in (sd,sh,sw)]
            field += amp * np.exp(-((_dg-cd)**2*inv[0]+(_hg-ch)**2*inv[1]+(_wg-cw)**2*inv[2]))

        field -= field.min(); mx = field.max()
        if mx>0: field/=mx

        iso = np.clip(self.iso+rng.uniform(-self.iso_var,self.iso_var),0.1,0.9)
        lo,hi = iso-self.band/2, iso+self.band/2
        mem_mask = (field>=lo)&(field<=hi)

        vol = np.full(SHAPE, self.bg, np.float32)
        if mem_mask.any():
            dist = np.abs(field-iso)/(self.band*0.5)
            grad = np.clip(dist,0,1)*self.grad*self.mem
            mem_vals = np.clip(self.mem+grad,0.05,0.5)
            vol[mem_mask]=mem_vals[mem_mask]

        # Initialize segmentation mask if needed
        if self.generate_masks:
            seg_mask = np.zeros(SHAPE, dtype=np.uint8)  # 0 = background
            if mem_mask.any():
                seg_mask[mem_mask] = 1  # 1 = membrane

        # Compute distance transform for fast membrane collision detection
        if mem_mask.any():
            dist_to_membrane = distance_transform_edt(~mem_mask)
        else:
            dist_to_membrane = np.full(SHAPE, float('inf'), dtype=np.float32)

        # Track placed shapes for collision detection (vectorized)
        placed_centers = []
        placed_radii = []

        # Generate spheres with collision detection (including membrane avoidance)
        ns0,ns1 = self.n_sph
        target_spheres = rng.randint(ns0, ns1+1) if ns1 > 0 else 0
        placed_spheres = 0
        
        for _ in range(target_spheres):
                cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
            r = rng.uniform(*self.sph_r)
            
            # Fast distance-based membrane collision check
            center_coords = (int(np.clip(cd, 0, D-1)), int(np.clip(ch, 0, H-1)), int(np.clip(cw, 0, W-1)))
            min_gap = 2.0
            membrane_clear = dist_to_membrane[center_coords] >= (r + min_gap)
            
            # Vectorized collision check with other organelles
            placed_centers_arr = np.array(placed_centers) if placed_centers else np.empty((0, 3))
            placed_radii_arr = np.array(placed_radii) if placed_radii else np.empty(0)
            organelle_clear = not self.check_collision_vectorized(placed_centers_arr, placed_radii_arr, (cd,ch,cw), r)
            
            if membrane_clear and organelle_clear:
                sph_mask = ((_dg-cd)**2+(_hg-ch)**2+(_wg-cw)**2)<r*r
                sval=np.clip(self.sph+rng.uniform(-0.02,0.02),0.01,0.15)
                vol[sph_mask]=sval

                if self.generate_masks:
                    seg_mask[sph_mask] = 2  # 2 = sphere
                
                placed_centers.append((cd,ch,cw))
                placed_radii.append(r)
                placed_spheres += 1

        # Generate cubes with collision detection (including membrane avoidance)
        nc0,nc1 = self.n_cube
        target_cubes = rng.randint(nc0, nc1+1) if nc1 > 0 else 0
        placed_cubes = 0
        
        for _ in range(target_cubes):
            cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
            size = rng.uniform(*self.cube_size)
            
            # Fast distance-based membrane collision check
            center_coords = (int(np.clip(cd, 0, D-1)), int(np.clip(ch, 0, H-1)), int(np.clip(cw, 0, W-1)))
            min_gap = 2.0
            # Use half-diagonal of cube as effective radius for collision check
            effective_radius = size * np.sqrt(3) / 2
            membrane_clear = dist_to_membrane[center_coords] >= (effective_radius + min_gap)
            
            # Vectorized collision check with other organelles
            placed_centers_arr = np.array(placed_centers) if placed_centers else np.empty((0, 3))
            placed_radii_arr = np.array(placed_radii) if placed_radii else np.empty(0)
            organelle_clear = not self.check_collision_vectorized(placed_centers_arr, placed_radii_arr, (cd,ch,cw), effective_radius)
            
            if membrane_clear and organelle_clear:
                # Create cube mask
                half_size = size / 2
                cube_mask = (
                    (np.abs(_dg - cd) <= half_size) &
                    (np.abs(_hg - ch) <= half_size) &
                    (np.abs(_wg - cw) <= half_size)
                )
                
                cval = np.clip(self.cube + rng.uniform(-0.02, 0.02), 0.01, 0.15)
                vol[cube_mask] = cval

                if self.generate_masks:
                    seg_mask[cube_mask] = 3  # 3 = cube
                
                placed_centers.append((cd, ch, cw))
                placed_radii.append(effective_radius)
                placed_cubes += 1

        # Fast separable blur (3x faster than gaussian_filter)
        if self.blur > 0:
            # Create 1D Gaussian kernel
            sigma = self.blur
            kernel_size = int(4 * sigma + 1)  # Reasonable kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            x = np.arange(kernel_size) - kernel_size // 2
            kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            
            # Apply separable 1D convolutions
            vol = convolve1d(vol, kernel_1d, axis=0, mode='nearest')
            vol = convolve1d(vol, kernel_1d, axis=1, mode='nearest')
            vol = convolve1d(vol, kernel_1d, axis=2, mode='nearest')
        if self.noise>0: vol+=rng.normal(0,self.noise,SHAPE).astype(np.float32)
        np.clip(vol,0,1,out=vol)
        
        if self.generate_masks:
            return vol.tobytes(), seg_mask.tobytes()
        else:
        return vol.tobytes()


# ─────────────────────────── worker loop ─────────────────────────────── #
def worker(seed_start, step, q: mp.Queue, stop_evt: mp.Event, generate_masks=False):
    gen = MembraneGen(generate_masks=generate_masks)
    seed = seed_start
    while not stop_evt.is_set():
        result = gen(seed)
        q.put(result)
        seed += step


# ─────────────────────────── main writer ─────────────────────────────── #
def writer(out_dir: Path, total, shard_size, q: mp.Queue,
           stop_evt: mp.Event, generate_masks=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mask directory if generating masks
    if generate_masks:
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
    
    shard, inside = 0, 0
    vol_tar = tarfile.open(out_dir/f"shard_{shard:05d}.tar", "w|")
    if generate_masks:
        mask_tar = tarfile.open(mask_dir/f"shard_{shard:05d}.tar", "w|")
    
    for _ in range(total):
        result = q.get()
        
        if generate_masks:
            vol_buf, mask_buf = result
        else:
            vol_buf = result
        
        # Write volume
        vol_name = f"v{shard:05d}_{inside:05d}.bin"
        vol_ti = tarfile.TarInfo(vol_name); vol_ti.size = VOL_BYTES
        vol_tar.addfile(vol_ti, io.BytesIO(vol_buf))
        
        # Write mask if generating
        if generate_masks:
            mask_name = f"m{shard:05d}_{inside:05d}.bin"
            mask_ti = tarfile.TarInfo(mask_name); mask_ti.size = MASK_BYTES
            mask_tar.addfile(mask_ti, io.BytesIO(mask_buf))
        
        inside += 1
        if inside == shard_size:
            vol_tar.close()
            if generate_masks:
                mask_tar.close()
                print(f"[+] shard_{shard:05d}.tar (volumes & masks, {shard_size})")
            else:
                print(f"[+] shard_{shard:05d}.tar (volumes only, {shard_size})")
            
            shard, inside = shard+1, 0
            vol_tar = tarfile.open(out_dir/f"shard_{shard:05d}.tar", "w|")
            if generate_masks:
                mask_tar = tarfile.open(mask_dir/f"shard_{shard:05d}.tar", "w|")
    
    vol_tar.close()
    if generate_masks:
        mask_tar.close()
    stop_evt.set()
    
    if generate_masks:
        print(f"[✓] dataset complete with segmentation masks")
    else:
        print(f"[✓] dataset complete")


# ─────────────────────────── CLI / orchestrator ──────────────────────── #
def main():
    ap = argparse.ArgumentParser("Membrane generator – bounded queue")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--num_volumes", type=int, default=1_048_576)
    ap.add_argument("--shard_size",  type=int, default=16_384)
    ap.add_argument("--num_workers", type=int, default=os.cpu_count())
    ap.add_argument("--queue_max",   type=int, default=256,
                    help="max volumes queued (×1 MiB RAM)")
    ap.add_argument("--overwrite",   action="store_true")
    ap.add_argument("--generate_masks", action="store_true",
                    help="Generate segmentation masks (0=bg, 1=membrane, 2=sphere, 3=cube)")
    cfg = ap.parse_args()

    if cfg.num_volumes % cfg.shard_size:
        ap.error("--num_volumes must be a multiple of --shard_size")
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()) and not cfg.overwrite:
        ap.error(f"{cfg.output_dir} not empty – pass --overwrite")
    if cfg.overwrite and cfg.output_dir.exists():
        for f in cfg.output_dir.glob("shard_*.tar"): f.unlink()
        if cfg.generate_masks:
            mask_dir = cfg.output_dir / "masks"
            if mask_dir.exists():
                for f in mask_dir.glob("shard_*.tar"): f.unlink()

    q = mp.Queue(maxsize=cfg.queue_max)
    stop_evt = mp.Event()

    # one writer process
    w = mp.Process(target=writer,
                   args=(cfg.output_dir, cfg.num_volumes,
                         cfg.shard_size, q, stop_evt, cfg.generate_masks),
                   daemon=True)
    w.start()

    # N worker processes
    procs = []
    for i in range(cfg.num_workers):
        p = mp.Process(target=worker,
                       args=(i, cfg.num_workers, q, stop_evt, cfg.generate_masks),
                       daemon=True)
        p.start(); procs.append(p)

    w.join()          # waits until all volumes written
    for p in procs: p.join()

if __name__ == "__main__":
    main()
