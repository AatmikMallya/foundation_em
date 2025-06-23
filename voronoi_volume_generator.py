#!/usr/bin/env python3
"""
Fast Voronoi-membrane generator  →  Zarr (DEPTH = 16)

* parent RSS stays < 2 GiB no matter how many volumes
* same bit-identical output as v4
"""

from __future__ import annotations
import argparse, os, shutil, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import zarr
from zarr.storage import LRUStoreCache
from numcodecs import Blosc

DEPTH   = 16          # volumes per chunk
RETRIES = 20          # organelle placement retries


# ═══════════════════════  grid cache  ═══════════════════════
_GRIDS: dict[tuple[int, int, int], tuple[np.ndarray, ...]] = {}
def grid(shape):
    if shape not in _GRIDS:
        _GRIDS[shape] = np.indices(shape, dtype=np.float32)
    return _GRIDS[shape]


# ═══════════════════  geometry helper functions  ═════════════════════════════
def place_seeds(shape, n, *, buffer, mindist, seed):
    rng = np.random.default_rng(seed)
    D, H, W = shape
    pts, att = [], 0
    while len(pts) < n and att < n * 40:
        cand = rng.uniform(buffer, [D-buffer, H-buffer, W-buffer])
        if not pts or np.min(np.linalg.norm(np.asarray(pts) - cand, 1)) >= mindist:
            pts.append(cand.astype(np.float32))
        att += 1
    if len(pts) < n:
        raise RuntimeError("seed placement failed")
    return np.asarray(pts, np.float32)


def nearest_second(shape, seeds):
    z, y, x = grid(shape)
    near = np.full(shape, np.inf, np.float32)
    sec  = np.full(shape, np.inf, np.float32)
    for sz, sy, sx in seeds:
        d = np.sqrt((z-sz)**2 + (y-sy)**2 + (x-sx)**2, dtype=np.float32)
        m = d < near
        sec  = np.where(m, near, np.minimum(sec, d))
        near = np.where(m, d, near)
    return near, sec


def make_volume(ns, shape, seeds):
    near, sec = nearest_second(shape, seeds)
    bdist = (sec - near).astype(np.float32)

    vol = np.full(shape, ns.cytosol_intensity, np.float32)
    vol[ndimage.distance_transform_edt(bdist > 0.5) <= ns.membrane_thickness] = ns.membrane_intensity

    if ns.num_organelles:
        clear = bdist >= (ns.membrane_thickness +
                          ns.safety_margin + ns.organelle_radius)
        pts = np.argwhere(clear)
        if pts.shape[0] < ns.num_organelles:
            raise RuntimeError("organelle placement failed")
        rng = np.random.default_rng(ns.seed + 99)
        for cz, cy, cx in pts[rng.choice(len(pts), ns.num_organelles, False)]:
            r, r2 = ns.organelle_radius, ns.organelle_radius**2
            rint = int(np.ceil(r))
            z0, z1 = max(0, cz-rint), min(shape[0], cz+rint+1)
            y0, y1 = max(0, cy-rint), min(shape[1], cy+rint+1)
            x0, x1 = max(0, cx-rint), min(shape[2], cx+rint+1)
            zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
            vol[z0:z1, y0:y1, x0:x1][(zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 <= r2] = ns.organelle_intensity

    if ns.blur_sigma:
        vol = gaussian_filter(vol, ns.blur_sigma)
    if ns.noise_level:
        rng = np.random.default_rng(ns.seed + 999)
        vol += rng.normal(0, ns.noise_level, shape).astype(np.float32)
        np.clip(vol, 0, 1, out=vol)
    return vol


# ═══════════════════════  worker (DEPTH-sized batch)  ════════════════════════
def worker(task):
    start_idx, ns_d = task
    ns = argparse.Namespace(**ns_d)
    shape = (ns.img_size,) * 3
    end   = min(start_idx + DEPTH, ns.num_volumes)
    pack  = []

    for idx in range(start_idx, end):
        tries = 0
        while True:
            try:
                ns.seed = ns.base_seed + idx + tries
                seeds = place_seeds(shape, ns.num_cells,
                                    buffer=5, mindist=ns.min_seed_distance,
                                    seed=ns.seed)
                pack.append((idx, make_volume(ns, shape, seeds)))
                break
            except RuntimeError:
                tries += 1
                if tries >= RETRIES:    # give up on organelles
                    ns.num_organelles = 0
    return pack


# ═════════════════════  argument parser  ════════════════════════════════════
def get_args():
    P = argparse.ArgumentParser("Voronoi → Zarr (memory-safe)")
    # geometry
    P.add_argument("--img_size", type=int, default=64)
    P.add_argument("--num_cells", type=int, default=6)
    P.add_argument("--min_seed_distance", type=float, default=14.0)
    P.add_argument("--membrane_thickness", type=float, default=2.0)
    P.add_argument("--membrane_intensity", type=float, default=0.25)
    P.add_argument("--cytosol_intensity", type=float, default=0.7)
    P.add_argument("--num_organelles", type=int, default=8)
    P.add_argument("--organelle_radius", type=float, default=5.0)
    P.add_argument("--organelle_intensity", type=float, default=0.05)
    P.add_argument("--safety_margin", type=float, default=1.0)
    P.add_argument("--blur_sigma", type=float, default=0.5)
    P.add_argument("--noise_level", type=float, default=0.05)
    # batch / output
    P.add_argument("--num_volumes", type=int, default=1_000_000)
    P.add_argument("--base_seed", type=int, default=42)
    P.add_argument("--num_workers", type=int, default=0)
    P.add_argument("--output_dir", required=True)
    P.add_argument("--overwrite", action="store_true")
    P.add_argument("--cache_mb", type=int, default=16,
                   help="compressed-chunk LRU size for parent (MB)")
    return P.parse_args()


# ═══════════════════════════════════════  main  ═════════════════════════════
def main():
    ns = get_args()
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    n_workers = ns.num_workers or cpu_count()

    out = Path(ns.output_dir).expanduser()
    store_dir = out / "dataset.zarr"
    if store_dir.exists():
        if ns.overwrite:
            shutil.rmtree(store_dir)
        else:
            raise RuntimeError("dataset exists – use --overwrite")

    # --- create metadata ----------------------------------------------------
    store = zarr.DirectoryStore(store_dir, dimension_separator=".")
    zarr.open(
        store, mode="w",
        shape=(ns.num_volumes, ns.img_size, ns.img_size, ns.img_size),
        chunks=(DEPTH, ns.img_size, ns.img_size, ns.img_size),
        dtype="float32",
        compressor=Blosc("zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )

    root = zarr.open(LRUStoreCache(store,
                                   max_size=ns.cache_mb * 1024 * 1024), "r+")
    start = time.time()

    tasks = [(i, vars(ns)) for i in range(0, ns.num_volumes, DEPTH)]
    with Pool(processes=n_workers,
              maxtasksperchild=32) as pool:              # recycle scratch
        for pack in pool.imap_unordered(worker, tasks,
                                        chunksize=max(1, DEPTH // 2)):
            for idx, vol in pack:
                root.oindex[idx] = vol           # write single chunk
            # --- flush python objects out of the LRU every 10 000 volumes
            if (pack[-1][0] + 1) % 10_000 == 0:
                done = pack[-1][0] + 1
                rate = done / (time.time() - start)
                root.chunk_store.cache.clear()   # ✱ keeps parent RSS flat
                print(f"{done:,}/{ns.num_volumes:,}  ({rate:.1f} vol/s)")

    dur = time.time() - start
    print(f"✅  {ns.num_volumes:,} volumes in {dur/3600:.2f} h  "
          f"({ns.num_volumes/dur:.1f} vol/s)")


if __name__ == "__main__":
    main()
