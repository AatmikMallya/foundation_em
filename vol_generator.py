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
from scipy.ndimage import gaussian_filter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SHAPE = (96, 96, 96)
VOL_BYTES = np.prod(SHAPE) * 4
D, H, W = SHAPE
_dg, _hg, _wg = np.ogrid[:D, :H, :W]


# ─────────────────────────── membrane math ──────────────────────────── #
class MembraneGen:
    def __init__(self):
        # static params baked in; adapt if needed
        self.n_gauss = (4, 6); self.sigma = (20, 25)
        self.iso, self.band = 0.8, 0.1
        self.noise = 0.02
        self.n_sph, self.sph_r = (6, 6), (8., 8.)
        self.blur, self.iso_var, self.grad = 1.0, 0.3, 0.2
        self.bg, self.mem, self.sph = 0.72, 0.22, 0.03

    def __call__(self, seed: int) -> bytes:
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

        ns0,ns1 = self.n_sph
        if ns1>0:
            for _ in range(rng.randint(ns0,ns1+1)):
                cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
                r=rng.uniform(*self.sph_r)
                mask=((_dg-cd)**2+(_hg-ch)**2+(_wg-cw)**2)<r*r
                sval=np.clip(self.sph+rng.uniform(-0.02,0.02),0.01,0.15)
                vol[mask]=sval

        if self.blur>0: vol=gaussian_filter(vol,self.blur)
        if self.noise>0: vol+=rng.normal(0,self.noise,SHAPE).astype(np.float32)
        np.clip(vol,0,1,out=vol)
        return vol.tobytes()


# ─────────────────────────── worker loop ─────────────────────────────── #
def worker(seed_start, step, q: mp.Queue, stop_evt: mp.Event):
    gen = MembraneGen()
    seed = seed_start
    while not stop_evt.is_set():
        q.put(gen(seed))
        seed += step


# ─────────────────────────── main writer ─────────────────────────────── #
def writer(out_dir: Path, total, shard_size, q: mp.Queue,
           stop_evt: mp.Event):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard, inside = 0, 0
    tar = tarfile.open(out_dir/f"shard_{shard:05d}.tar", "w|")
    for _ in range(total):
        buf = q.get()
        name = f"v{shard:05d}_{inside:05d}.bin"
        ti = tarfile.TarInfo(name); ti.size = VOL_BYTES
        tar.addfile(ti, io.BytesIO(buf))
        inside += 1
        if inside == shard_size:
            tar.close(); print(f"[+] shard_{shard:05d}.tar ({shard_size})")
            shard, inside = shard+1, 0
            tar = tarfile.open(out_dir/f"shard_{shard:05d}.tar", "w|")
    tar.close(); stop_evt.set(); print(f"[✓] dataset complete")


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
    cfg = ap.parse_args()

    if cfg.num_volumes % cfg.shard_size:
        ap.error("--num_volumes must be a multiple of --shard_size")
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()) and not cfg.overwrite:
        ap.error(f"{cfg.output_dir} not empty – pass --overwrite")
    if cfg.overwrite and cfg.output_dir.exists():
        for f in cfg.output_dir.glob("shard_*.tar"): f.unlink()

    q = mp.Queue(maxsize=cfg.queue_max)
    stop_evt = mp.Event()

    # one writer process
    w = mp.Process(target=writer,
                   args=(cfg.output_dir, cfg.num_volumes,
                         cfg.shard_size, q, stop_evt),
                   daemon=True)
    w.start()

    # N worker processes
    procs = []
    for i in range(cfg.num_workers):
        p = mp.Process(target=worker,
                       args=(i, cfg.num_workers, q, stop_evt),
                       daemon=True)
        p.start(); procs.append(p)

    w.join()          # waits until all volumes written
    for p in procs: p.join()

if __name__ == "__main__":
    main()
