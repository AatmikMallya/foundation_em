#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAE layer probes – PE diagnostics
=================================

Three switches expose how much positional information dominates the token
sub-space and whether that hurts object-level semantics:

  1.  *Coordinate-loss sweep*        (--coord_weights 10 1 0 …)
  2.  *Per-class threshold calibration* (--calibrate)
  3.  *No-position-embedding control*  (--noposition)

Logged per (layer, coord_w, calibrated?, use_pe):

    • val_acc, obj_acc
    • macro_F1 & macro_F1_obj
    • AUROC_membrane / sphere / cube
    • xyz_R²                      (positional leakage)
    • obj_vote_acc                (instance-level)
    • cls_* counterparts if --cls_probe
2025-08-02
"""
from __future__ import annotations

import argparse, os, random, tarfile, gc, math, itertools, signal
from pathlib import Path
from collections import deque, defaultdict
from typing import List

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_auc_score
import wandb

# ───── MAE helpers from your repo ────────────────────────────────────────────
from train_simple_sae import ARCHS, get_patch_xyz

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ─────────────────────────────────────────────────────────────────────────────


# ╭──────────────────────────── Data ─────────────────────────────────╮
class TarShardDataset(IterableDataset):
    def __init__(
        self,
        vols: List[Path],
        masks: List[Path],
        side: int,
        vols_per_shard: int,
        shuffle: bool = False,
        synthetic_only: bool = False,
    ):
        if synthetic_only:
            vols  = [p for p in vols  if "synthetic" in p.name]
            masks = [p for p in masks if "synthetic" in p.name]
            assert vols, "No '*synthetic*' shards found"

        self.vols, self.masks = list(vols), list(masks)
        self.side, self.vps   = side, vols_per_shard
        self.shuffle = shuffle
        assert len(self.vols) == len(self.masks)

    def __len__(self):                        # nominal epoch size
        return len(self.vols) * self.vps

    # iterate one tar pair ---------------------------------------------------
    def _iter_pair(self, v_t: Path, m_t: Path):
        vt = tarfile.open(v_t, "r|", bufsize=32 * 1024 * 1024)
        mt = tarfile.open(m_t, "r|", bufsize=32 * 1024 * 1024)
        vit = (m for m in vt if m.isfile())
        mit = (m for m in mt if m.isfile())
        try:
            for mv in vit:
                vol = np.frombuffer(vt.extractfile(mv).read(), np.float32).reshape(
                    self.side, self.side, self.side
                )
                msk = np.frombuffer(mt.extractfile(next(mit)).read(), np.uint8).reshape(
                    self.side, self.side, self.side
                )
                yield torch.from_numpy(vol).unsqueeze(0), torch.from_numpy(msk)
        finally:
            vt.close(), mt.close()

    # iterable API -----------------------------------------------------------
    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        idx = list(range(len(self.vols)))
        if self.shuffle:
            random.shuffle(idx)
        if w:
            idx = idx[w.id :: w.num_workers]
        for i in idx:
            yield from self._iter_pair(self.vols[i], self.masks[i])


class CUDAPrefetcher:
    """2-batch pinned-mem → GPU stream prefetcher."""
    def __init__(self, loader: DataLoader, dev: torch.device):
        self.it     = iter(loader)
        self.dev    = dev
        self.stream = torch.cuda.Stream(dev, priority=-1)
        self.q, self.ev = [], []
        for _ in range(2):
            self._prefetch()

    def _prefetch(self):
        try:
            vol, msk = next(self.it)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            if not vol.is_pinned():
                vol = vol.pin_memory()
            if not msk.is_pinned():
                msk = msk.pin_memory()
            vol = vol.to(self.dev, non_blocking=True, memory_format=torch.channels_last_3d)
            msk = msk.to(self.dev, non_blocking=True)
            e = torch.cuda.Event()
            e.record(self.stream)
            self.q.append((vol, msk))
            self.ev.append(e)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.q:
            raise StopIteration
        self.ev.pop(0).wait()
        out = self.q.pop(0)
        self._prefetch()
        return out


# ╭────────────────────── Helpers ───────────────────────╮
def majority_labels(mask: torch.Tensor, p: int, K: int) -> torch.Tensor:
    """Per-patch majority label.  mask: (B,D,H,W) uint8  →  (B·L,) long"""
    B, D, H, W = mask.shape
    pd, ph, pw = D // p, H // p, W // p
    x = mask.view(B, pd, p, ph, p, pw, p).permute(0, 1, 3, 5, 2, 4, 6)
    x = x.flatten(0, 3).flatten(-3)                                # (B·pd·ph·pw, p³)
    counts = torch.zeros(x.size(0), K, device=mask.device, dtype=torch.int32)
    for c in range(K):
        counts[:, c] = (x == c).sum(-1)
    return counts.argmax(-1)


def bincount2d(t: torch.Tensor, K: int) -> torch.Tensor:
    """Fast confusion-matrix."""
    idx = t[:, 0] * K + t[:, 1]
    return torch.bincount(idx, minlength=K * K).view(K, K)


def per_class_f1(conf: torch.Tensor) -> torch.Tensor:
    tp = conf.diag()
    fn = conf.sum(1) - tp
    fp = conf.sum(0) - tp
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    return 2 * prec * rec / (prec + rec + 1e-12)


# ╭──────────────────────── Probe ───────────────────────╮
class Probe(torch.nn.Module):
    def __init__(self, dim: int, K: int, probe_type: str = "linear"):
        super().__init__()
        if probe_type == "mlp":
            self.clf = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 2), torch.nn.GELU(),
                torch.nn.Linear(dim * 2, K),
            )
            self.reg = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 2), torch.nn.GELU(),
                torch.nn.Linear(dim * 2, 3),
            )
        else:  # linear
            self.clf = torch.nn.Linear(dim, K)
            self.reg = torch.nn.Linear(dim, 3)

    def forward(self, x):
        return self.clf(x), self.reg(x)


def make_patch_hook(mae, L: int, use_pe: bool):
    """Return callable(vol) -> (B,L,C) tokens for layer `L`."""
    def h(vol):
        x = mae.encoder.patch_embed(vol)                     # B,L,C
        if L == -1:
            return x if use_pe else x                        # -1 never adds PE
        pe = mae.encoder.pos_embed[:, 1:, :] if use_pe else 0.0
        x = x + pe
        if L == -2:
            return x
        cls = mae.encoder.cls_token + mae.encoder.pos_embed[:, :1, :]
        x_full = torch.cat([cls.expand(vol.size(0), -1, -1), x], 1)
        for i, blk in enumerate(mae.encoder.blocks):
            x_full = blk(x_full)
            if i == L:
                return x_full[:, 1:, :]
        return mae.encoder.norm(x_full)[:, 1:, :]

    return h


# ╭──────────────────── Training 1 layer ───────────────────╮
def train_layer(
    L: int,
    mae,
    cfg,
    device: torch.device,
    table: wandb.Table,
    coord_w: float,
    use_pe: bool,
):
    dim, K = cfg.dim, cfg.K

    # -------- data ---------------------------------------------------------
    vols = sorted(Path(cfg.shard_dir).glob("shard_*.tar"))
    masks = sorted(Path(cfg.mask_dir).glob("shard_*.tar"))
    val_n = max(1, int(len(vols) * cfg.val_split))
    tr_ds = TarShardDataset(
        vols[val_n:], masks[val_n:], cfg.img, cfg.vps, True, cfg.synthetic
    )
    va_ds = TarShardDataset(
        vols[:val_n], masks[:val_n], cfg.img, cfg.vps, False, cfg.synthetic
    )

    loader  = DataLoader(
        tr_ds, batch_size=cfg.bs, num_workers=cfg.workers,
        pin_memory=True, persistent_workers=True, drop_last=True
    )
    vloader = DataLoader(va_ds, batch_size=cfg.bs, num_workers=2, pin_memory=True)

    probe  = Probe(dim, K, cfg.probe).to(device)
    opt    = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg.steps, eta_min=cfg.lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    hook   = make_patch_hook(mae, L, use_pe)

    grid   = (cfg.img // cfg.ps,) * 3
    coords = get_patch_xyz(math.prod(grid), grid).to(device)

    ema = deque(maxlen=100)
    step = 0
    key_prefix = f"L{L}/cw{coord_w}_pe{int(use_pe)}"          # unique per setting

    while step < cfg.steps:
        for vol, mask in CUDAPrefetcher(loader, device):
            labs = majority_labels(mask, cfg.ps, K)
            xyz  = coords.repeat(vol.size(0), 1)
            toks = hook(vol).float().reshape(-1, dim)

            with torch.autocast("cuda", torch.bfloat16, enabled=cfg.amp):
                logit, xyz_pred = probe(toks)
                loss = F.cross_entropy(logit, labs) + coord_w * F.mse_loss(
                    xyz_pred, xyz
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()

            ema.append(loss.item())
            step += 1

            if step % cfg.log_int == 0:
                with torch.no_grad():
                    preds = logit.argmax(-1)
                    wandb.log(
                        {
                            f"{key_prefix}/train_loss": sum(ema) / len(ema),
                            "step": step,
                            f"{key_prefix}/train_acc": (preds == labs)
                            .float()
                            .mean()
                            .item(),
                        }
                    )

            if step >= cfg.steps:
                break

    # -------- validation ---------------------------------------------------
    probe.eval()
    conf = torch.zeros(K, K, dtype=torch.int64, device=device)
    obj_conf = conf.clone()
    auc_tot = torch.zeros(K, device=device)
    auc_cnt = torch.zeros(K, device=device)
    xyz_mse = 0.0
    n_tok   = 0

    # storage for threshold search
    if cfg.calibrate:
        store_prob = [[] for _ in range(K)]
        store_lbl  = [[] for _ in range(K)]

    with torch.no_grad():
        for vol, mask in vloader:
            labs = majority_labels(mask.to(device), cfg.ps, K)
            toks = hook(vol.to(device)).float().reshape(-1, dim)
            logit, xyz_pred = probe(toks)
            probs = F.softmax(logit, -1).detach()
            preds = probs.argmax(-1)

            conf += bincount2d(torch.stack([labs, preds], 1), K)
            obj_conf += bincount2d(torch.stack([labs[labs != 0], preds[labs != 0]], 1), K)

            xyz_mse += F.mse_loss(
                xyz_pred, coords.repeat(vol.size(0), 1), reduction="sum"
            ).item()
            n_tok += labs.numel()

            # AUROC
            for c in (1, 2, 3):
                mask_c = labs == c
                if mask_c.any():
                    auc = roc_auc_score(
                        mask_c.cpu().numpy(),
                        probs[:, c].cpu().numpy(),
                    )
                    auc_tot[c] += auc
                    auc_cnt[c] += 1

            if cfg.calibrate:
                for c in range(K):
                    store_prob[c].append(probs[:, c].cpu())
                    store_lbl[c].append((labs == c).cpu())

    # ---- metrics ----------------------------------------------------------
    f1 = per_class_f1(conf.float())
    macro_F1 = f1.mean().item()
    macro_obj_F1 = f1[1:].mean().item()
    val_acc = conf.diag().sum().item() / conf.sum().item()
    val_obj_acc = obj_conf.diag()[1:].sum().item() / obj_conf[1:].sum().item()
    xyz_R2 = 1.0 - xyz_mse / (
        n_tok * (coords.var(0).mean().item() + 1e-12)
    )
    auc = [(auc_tot[c] / max(auc_cnt[c], 1)).item() for c in range(K)]

    # ---- optional calibration --------------------------------------------
    macro_F1_cal = None
    if cfg.calibrate:
        thresholds = []
        for c in range(K):
            p = torch.cat(store_prob[c])
            y = torch.cat(store_lbl[c])
            best_f1, best_t = 0.0, 0.5
            for t in torch.linspace(0.05, 0.95, 19):
                tp = ((p >= t) & (y == 1)).sum()
                fp = ((p >= t) & (y == 0)).sum()
                fn = ((p < t) & (y == 1)).sum()
                f1_tmp = 2 * tp / (2 * tp + fp + fn + 1e-12)
                if f1_tmp > best_f1:
                    best_f1, best_t = f1_tmp, t.item()
            thresholds.append(best_t)

        # apply
        preds_cal = []
        for vol, _ in vloader:
            toks = hook(vol.to(device)).float().reshape(-1, dim)
            probs = F.softmax(probe(toks)[0], -1).cpu()
            logits_thr = probs - torch.tensor(thresholds).view(1, -1)
            preds_cal.append(logits_thr.argmax(1))
        preds_cal = torch.cat(preds_cal).to(device)
        labs_all = torch.cat(
            [majority_labels(m.to(device), cfg.ps, K) for _, m in vloader]
        )
        conf_cal = bincount2d(torch.stack([labs_all, preds_cal], 1), K)
        macro_F1_cal = per_class_f1(conf_cal.float()).mean().item()

    # ---- record -----------------------------------------------------------
    row = [
        L,
        coord_w,
        int(cfg.calibrate and macro_F1_cal is not None),
        int(use_pe),
        val_acc,
        val_obj_acc,
        macro_F1,
        macro_obj_F1,
        xyz_R2,
        auc[1],
        auc[2],
        auc[3],
        macro_F1_cal,
    ]
    table.add_data(*row)

    wandb.log(
        {
            f"{key_prefix}/val_macro_F1": macro_F1,
            f"{key_prefix}/val_xyz_R2": xyz_R2,
            f"{key_prefix}/val_acc": val_acc,
        }
    )


# ╭─────────────────────────── Main ─────────────────────────────────╮
def main(cfg):
    # env -------------------------------------------------------------------
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dev = torch.device("cuda")
    mae = ARCHS[cfg.arch](
        volume_size=(cfg.img,) * 3, patch_size=(cfg.ps,) * 3, in_chans=1, mask_ratio=0.0
    )
    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    mae.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    mae.to(dev).eval().requires_grad_(False)
    cfg.dim = mae.encoder.embed_dim   # attach to cfg for convenience

    run = wandb.init(project=cfg.project, name=cfg.run, config=vars(cfg))
    cols = [
        "layer",
        "coord_w",
        "calibrated",
        "use_pe",
        "val_acc",
        "val_obj_acc",
        "macro_F1",
        "macro_F1_obj",
        "xyz_R2",
        "AUROC_membrane",
        "AUROC_sphere",
        "AUROC_cube",
        "macro_F1_cal",
    ]
    tbl = wandb.Table(columns=cols)

    # sweep over (layer, coord_w, PE flag)
    for L, cw, pe_flag in itertools.product(
        cfg.layers, cfg.coord_weights, [True, not cfg.noposition]
    ):
        train_layer(L, mae, cfg, dev, tbl, cw, pe_flag)

    wandb.log({"summary_table": tbl})
    run.finish()


# ╭───────────────────────── CLI ─────────────────────────╮
if __name__ == "__main__":
    P = argparse.ArgumentParser("PE-diagnostic layer probes")
    # data
    P.add_argument("--shard_dir", required=True)
    P.add_argument("--mask_dir", required=True)
    P.add_argument("--img", type=int, default=96)
    P.add_argument("--ps", type=int, default=8, help="patch size")
    P.add_argument("--vps", type=int, default=16384, help="vols per shard (nominal)")
    P.add_argument("--synthetic", action="store_true", help="use only '*synthetic*' shards")
    # model
    P.add_argument("--ckpt", required=True)
    P.add_argument("--arch", default="base_patch_conv", choices=list(ARCHS.keys()))
    # probe / training
    P.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    P.add_argument("--coord_weights", type=float, nargs="+", default=[10, 1, 0])
    P.add_argument("--calibrate", action="store_true", help="search per-class thresholds")
    P.add_argument("--noposition", action="store_true", help="run a no-PE control")
    P.add_argument("--bs", type=int, default=8)
    P.add_argument("--steps", type=int, default=4000)
    P.add_argument("--lr", type=float, default=1e-2)
    P.add_argument("--wd", type=float, default=1e-4)
    P.add_argument("--clip", type=float, default=1.0)
    P.add_argument("--amp", action="store_true")
    P.add_argument("--workers", type=int, default=4)
    P.add_argument("--val_split", type=float, default=0.02)
    P.add_argument("--K", type=int, default=4)
    P.add_argument("--layers", type=int, nargs="+", default=[-1, -2] + list(range(12)))
    # misc / logging
    P.add_argument("--seed", type=int, default=42)
    P.add_argument("--project", default="mae_pe_probe")
    P.add_argument("--run", default="diagnostics")
    P.add_argument("--log_int", type=int, default=20)

    cfg = P.parse_args()
    main(cfg)
