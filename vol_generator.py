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
    def __init__(self, generate_masks=False, equal_combinations=True):
        # static params baked in; adapt if needed
        self.n_gauss = (2, 4); self.sigma = (35, 45)  # Reduced from (4,6) for simpler membranes
        self.iso, self.band = 0.8, 0.08  # Keep thin membranes
        self.noise = 0.01  # Reduced from 0.02
        
        # IMPROVED: Varied sphere parameters
        self.n_sph = (2, 16)  # Variable count: 4-8 spheres when present
        self.sph_r = (3.0, 12.0)  # Variable radius: 4-12 pixels
        
        # IMPROVED: Varied cube parameters  
        self.n_cube = (2, 16)  # Variable count: 2-6 cubes when present
        self.cube_size = (8.0, 16.0)  # Variable size: 8-16 pixels
        
        self.blur, self.iso_var, self.grad = 0.5, 0.2, 0.2 # Reduced blur and iso_var
        
        # IMPROVED: Better intensity separation for interpretability
        self.bg_base = 0.75      # Higher background for better contrast
        self.mem_base = 0.25     # Clear membrane signal  
        self.sph_base = 0.05     # Low but detectable spheres
        self.cube_base = 0.45    # High cube signal for clear separation
        self.intensity_variation = 0.1  # Reduced variation for more stable features
        
        self.generate_masks = generate_masks
        self.equal_combinations = equal_combinations
        
        # Define all 8 possible combinations
        self.combinations = [
            {'membrane': True,  'spheres': False, 'cubes': False},  # Membranes only
            {'membrane': False, 'spheres': True,  'cubes': False},  # Spheres only
            {'membrane': False, 'spheres': False, 'cubes': True},   # Cubes only
            {'membrane': True,  'spheres': True,  'cubes': False},  # Membranes + Spheres
            {'membrane': True,  'spheres': False, 'cubes': True},   # Membranes + Cubes
            {'membrane': False, 'spheres': True,  'cubes': True},   # Spheres + Cubes
            {'membrane': True,  'spheres': True,  'cubes': True},   # All structures
        ]
        
        # Add validation tracking
        self.debug_mode = False
        self.feature_stats = {'bg_only': 0, 'mem_only': 0, 'sph_only': 0, 'cube_only': 0, 
                             'mem_sph': 0, 'mem_cube': 0, 'sph_cube': 0, 'all': 0}

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
        
        # IMPROVED: Choose combination with equal probability
        if self.equal_combinations:
            combo = rng.choice(self.combinations)
        else:
            # Fallback to old behavior (all structures always present)
            combo = {'membrane': True, 'spheres': True, 'cubes': True}
        
        # Generate intensity values with variation
        bg_intensity = self.bg_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        mem_intensity = self.mem_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        sph_intensity = self.sph_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        cube_intensity = self.cube_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        
        # Clamp intensities to well-separated ranges for interpretability
        bg_intensity = np.clip(bg_intensity, 0.65, 0.85)     # Narrow background range
        mem_intensity = np.clip(mem_intensity, 0.20, 0.30)   # Clear membrane signal  
        sph_intensity = np.clip(sph_intensity, 0.03, 0.08)   # Low but consistent spheres
        cube_intensity = np.clip(cube_intensity, 0.35, 0.55) # High, distinct cube signal
        
        field = np.zeros(SHAPE, np.float32)

        # Generate membrane structures only if chosen
        if combo['membrane']:
            for _ in range(rng.randint(*self.n_gauss)+1):
                cd,ch,cw = rng.uniform(0,D), rng.uniform(0,H), rng.uniform(0,W)
                sd,sh,sw = (rng.uniform(*self.sigma) for _ in range(3))
                amp = rng.uniform(0.5,1.5)
                inv = [1/(2*s*s) for s in (sd,sh,sw)]
                field += amp * np.exp(-((_dg-cd)**2*inv[0]+(_hg-ch)**2*inv[1]+(_wg-cw)**2*inv[2]))

        # Initialize volume with background
        vol = np.full(SHAPE, bg_intensity, np.float32)
        
        # Process membrane field only if membranes are chosen
        if combo['membrane']:
            field -= field.min(); mx = field.max()
            if mx>0: field/=mx

            iso = np.clip(self.iso+rng.uniform(-self.iso_var,self.iso_var),0.1,0.9)
            lo,hi = iso-self.band/2, iso+self.band/2
            mem_mask = (field>=lo)&(field<=hi)

            if mem_mask.any():
                dist = np.abs(field-iso)/(self.band*0.5)
                grad = np.clip(dist,0,1)*self.grad*mem_intensity
                mem_vals = np.clip(mem_intensity+grad,0.05,0.34) # Clip below cube intensity
                vol[mem_mask]=mem_vals[mem_mask]

        # Initialize segmentation mask if needed
        if self.generate_masks:
            seg_mask = np.zeros(SHAPE, dtype=np.uint8)  # 0 = background
            if combo['membrane'] and 'mem_mask' in locals() and mem_mask.any():
                seg_mask[mem_mask] = 1  # 1 = membrane

        # Compute distance transform for membrane collision detection only if membranes exist
        if combo['membrane'] and 'mem_mask' in locals() and mem_mask.any():
            dist_to_membrane = distance_transform_edt(~mem_mask)
        else:
            dist_to_membrane = np.full(SHAPE, float('inf'), dtype=np.float32)

        # Track placed shapes for collision detection (vectorized)
        placed_centers = []
        placed_radii = []

        # Generate spheres only if chosen
        if combo['spheres']:
            ns0,ns1 = self.n_sph
            target_spheres = rng.randint(ns0, ns1+1)
            placed_spheres = 0
            
            for _ in range(target_spheres):
                cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
                r = rng.uniform(*self.sph_r)  # Variable radius
                
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
                    # Each sphere gets its own intensity
                    sval = self.sph_base + rng.uniform(-self.intensity_variation*0.5, self.intensity_variation*0.5)
                    sval = np.clip(sval, 0.001, 0.2)
                    vol[sph_mask]=sval

                    if self.generate_masks:
                        seg_mask[sph_mask] = 2  # 2 = sphere
                    
                    placed_centers.append((cd,ch,cw))
                    placed_radii.append(r)
                    placed_spheres += 1

        # Generate cubes only if chosen
        if combo['cubes']:
            nc0,nc1 = self.n_cube
            target_cubes = rng.randint(nc0, nc1+1)
            placed_cubes = 0
            
            for _ in range(target_cubes):
                cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
                size = rng.uniform(*self.cube_size)  # Variable size
                
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
                    
                    # Each cube gets its own intensity
                    cval = self.cube_base + rng.uniform(-self.intensity_variation*0.7, self.intensity_variation*0.7)
                    cval = np.clip(cval, 0.05, 0.5)
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
        
        # Track combination statistics for validation
        if self.debug_mode:
            combo_key = self._get_combo_key(combo)
            self.feature_stats[combo_key] += 1
        
        if self.generate_masks:
            return vol.tobytes(), seg_mask.tobytes()
        else:
            return vol.tobytes()

    def _get_combo_key(self, combo):
        """Convert combination dict to stats key"""
        if not combo['membrane'] and not combo['spheres'] and not combo['cubes']:
            return 'bg_only'
        elif combo['membrane'] and not combo['spheres'] and not combo['cubes']:
            return 'mem_only'
        elif not combo['membrane'] and combo['spheres'] and not combo['cubes']:
            return 'sph_only'
        elif not combo['membrane'] and not combo['spheres'] and combo['cubes']:
            return 'cube_only'
        elif combo['membrane'] and combo['spheres'] and not combo['cubes']:
            return 'mem_sph'
        elif combo['membrane'] and not combo['spheres'] and combo['cubes']:
            return 'mem_cube'
        elif not combo['membrane'] and combo['spheres'] and combo['cubes']:
            return 'sph_cube'
        else:  # all True
            return 'all'

    def print_stats(self):
        """Print feature combination statistics"""
        total = sum(self.feature_stats.values())
        if total == 0:
            print("No statistics collected (debug_mode not enabled)")
            return
        print(f"\nFeature combination statistics ({total} volumes):")
        for key, count in self.feature_stats.items():
            pct = 100 * count / total
            print(f"  {key:12s}: {count:6d} ({pct:5.1f}%)")
    
    def validate_intensity_separation(self, vol):
        """Quick validation that intensity ranges are well-separated"""
        unique_vals = np.unique(vol)
        bg_vals = unique_vals[(unique_vals >= 0.65) & (unique_vals <= 0.85)]
        mem_vals = unique_vals[(unique_vals >= 0.20) & (unique_vals <= 0.30)]
        sph_vals = unique_vals[(unique_vals >= 0.03) & (unique_vals <= 0.08)]
        cube_vals = unique_vals[(unique_vals >= 0.35) & (unique_vals <= 0.55)]
        
        return {
            'bg_range': (bg_vals.min(), bg_vals.max()) if len(bg_vals) > 0 else None,
            'mem_range': (mem_vals.min(), mem_vals.max()) if len(mem_vals) > 0 else None,
            'sph_range': (sph_vals.min(), sph_vals.max()) if len(sph_vals) > 0 else None,
            'cube_range': (cube_vals.min(), cube_vals.max()) if len(cube_vals) > 0 else None,
        }


# ─────────────────────────── worker loop ─────────────────────────────── #
def worker(seed_start, step, q: mp.Queue, stop_evt: mp.Event, generate_masks=False, equal_combinations=True):
    gen = MembraneGen(generate_masks=generate_masks, equal_combinations=equal_combinations)
    seed = seed_start
    while not stop_evt.is_set():
        result = gen(seed)
        q.put(result)
        seed += step

# ─────────────────────────── shard writer ─────────────────────────────── #
def shard_writer(q: mp.Queue, out_dir: Path, shard_size: int, num_shards: int, 
           stop_evt: mp.Event, generate_masks=False):
    out_dir.mkdir(exist_ok=True)
    if generate_masks:
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
    
    for s in range(num_shards):
        vol_path = out_dir / f"shard_{s:05d}.tar"
        mask_path = mask_dir / f"shard_{s:05d}.tar" if generate_masks else None
        
        print(f"Writing shard {s+1}/{num_shards}...")
        with tarfile.open(vol_path, "w") as vol_tar:
            mask_tar = tarfile.open(mask_path, "w") if generate_masks else None
            try:
                for v in range(shard_size):
                    result = q.get(timeout=30)
        
                    if generate_masks:
                        vol_bytes, mask_bytes = result
        
                        # Add volume
                        vol_info = tarfile.TarInfo(f"vol_{s:05d}_{v:05d}.bin")
                        vol_info.size = len(vol_bytes)
                        vol_tar.addfile(vol_info, io.BytesIO(vol_bytes))
        
                        # Add mask
                        mask_info = tarfile.TarInfo(f"mask_{s:05d}_{v:05d}.bin")
                        mask_info.size = len(mask_bytes)
                        mask_tar.addfile(mask_info, io.BytesIO(mask_bytes))
                    else:
                        vol_bytes = result
                        vol_info = tarfile.TarInfo(f"vol_{s:05d}_{v:05d}.bin")
                        vol_info.size = len(vol_bytes)
                        vol_tar.addfile(vol_info, io.BytesIO(vol_bytes))
            finally:
                if mask_tar:
                    mask_tar.close()


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
    ap.add_argument("--equal_combinations", action="store_true", default=True,
                    help="Generate all 8 structure combinations with equal probability (default: True)")
    ap.add_argument("--no_equal_combinations", dest="equal_combinations", action="store_false",
                    help="Use old behavior: always generate all structures together")
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

    num_shards = cfg.num_volumes // cfg.shard_size
    workers = [mp.Process(target=worker, args=(i, cfg.num_workers, q, stop_evt, cfg.generate_masks, cfg.equal_combinations))
               for i in range(cfg.num_workers)]
    writer = mp.Process(target=shard_writer, args=(q, cfg.output_dir, cfg.shard_size, num_shards, stop_evt, cfg.generate_masks))

    try:
        for w in workers: w.start()
        writer.start()
        writer.join()  # Wait for writing to finish
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        stop_evt.set()
        for w in workers: w.join()

if __name__ == "__main__":
    main()
