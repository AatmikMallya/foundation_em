#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage‑1 SAE (reconstruction sanity) for 3‑D MAE tokens
-----------------------------------------------------
• SAE masking == MAE masking (reuse MAE mask/ids; never recompute new ones).
• Best-model saving on lowest pixel_mse_sae_gt.
• Rich monosemanticity metrics (purity/IoU/AUROC/MI/entropy/selectivity/IFF + positional invariance).
• Patch coloring by dominant neuron (2‑D overlays & compact 3‑D GIFs; only color kept patches by default).
• Throughput tweaks (single MAE fwd, chunked SAE, TF32, pinned mem, compile).
• Positional pass‑through (Option B) configurable by encoder layer.
• Graph consistency loss to push neighboring tokens toward the same neuron.

2025‑07‑27
"""

import argparse, random, signal, tarfile, math, gc
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import wandb
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# perf flags
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ─────────── MAE builders ─────────── #
from vit_3d import (
    mae_vit_3d_small, mae_vit_3d_base, mae_vit_3d_large,
    mae_vit_3d_hemibrain_optimal,
    mae_vit_3d_small_conv, mae_vit_3d_base_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv, mae_vit_3d_base_patch_conv
)

ARCHS = {
    "small": mae_vit_3d_small,
    "base":  mae_vit_3d_base,
    "large": mae_vit_3d_large,
    "hemibrain_optimal": mae_vit_3d_hemibrain_optimal,
    "small_conv": mae_vit_3d_small_conv,
    "base_conv":  mae_vit_3d_base_conv,
    "large_conv": mae_vit_3d_large_conv,
    "hemibrain_optimal_conv": mae_vit_3d_hemibrain_optimal_conv,
    "base_patch_conv": mae_vit_3d_base_patch_conv,
}

# ====================== utils ======================
def load_model_checkpoint(path, model, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.","",1): v for k,v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:  print("Missing keys:", miss[:10])
    if unexp: print("Unexpected:",   unexp[:10])
    md = ckpt.get("model_dtype", "")
    if md == "torch.bfloat16": model = model.to(dtype=torch.bfloat16)
    if md == "torch.float16":  model = model.to(dtype=torch.float16)
    return model.to(device)

def orthogonality_loss(W):
    WT = W @ W.t()
    I  = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    return F.mse_loss(WT, I)

def topk_st(z, k):
    if k <= 0 or k >= z.shape[1]:
        return z
    vals, idx = torch.topk(z, k, dim=1)
    z_hard = torch.zeros_like(z).scatter_(1, idx, vals)
    return z + (z_hard - z).detach()

@torch.no_grad()
def build_keep_restore(mask: torch.Tensor):
    # mask: (B,L) 0=keep,1=mask
    ids_sorted = torch.argsort(mask, dim=1, stable=True)
    len_keep = (mask==0).sum(1)
    assert len_keep.unique().numel()==1, "varying keep counts per batch not supported"
    len_keep = int(len_keep[0])
    ids_keep = ids_sorted[:, :len_keep]
    ids_restore = torch.argsort(ids_sorted, dim=1)
    return ids_keep, ids_restore, len_keep

# ====================== data ======================
class TarShardDataset(IterableDataset):
    def __init__(self, vol_shards, vs, shuffle=False, vps=16_384, mask_shards=None):
        self.vols=list(vol_shards)
        self.masks=list(mask_shards) if mask_shards else None
        self.vs=vs; self.shuf=shuffle; self.vps=vps
        print(f"Dataset: {len(self.vols)} shards × {vps} vols" + (" + masks" if self.masks else ""))

    def __len__(self): return len(self.vols)*self.vps

    def _iter_pair(self, v_path, m_path):
        vol_tf  = tarfile.open(v_path,  "r|", bufsize=32*1024*1024)
        mask_tf = tarfile.open(m_path, "r|", bufsize=32*1024*1024) if m_path else None

        vit = (m for m in vol_tf  if m.isfile())
        mit = (m for m in mask_tf if m.isfile()) if mask_tf else None
        for mv in vit:
            vol = np.frombuffer(vol_tf.extractfile(mv).read(), np.float32).reshape(self.vs,self.vs,self.vs)
            vol_t = torch.from_numpy(vol.copy()).unsqueeze(0).pin_memory()
            if mit:
                mm = next(mit)
                seg = np.frombuffer(mask_tf.extractfile(mm).read(), np.uint8).reshape(self.vs,self.vs,self.vs)
                seg_t = torch.from_numpy(seg.copy()).pin_memory()
            else:
                seg_t = torch.zeros((self.vs, self.vs, self.vs), dtype=torch.uint8).pin_memory()
            yield vol_t, seg_t

        vol_tf.close()
        if mask_tf: mask_tf.close()

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        idxs = list(range(len(self.vols)))
        if self.shuf: random.shuffle(idxs)
        if w: idxs = idxs[w.id::w.num_workers]
        for i in idxs:
            m = self.masks[i] if self.masks else None
            yield from self._iter_pair(self.vols[i], m)

class CUDAPrefetcher:
    def __init__(self, loader, dev):
        self.it = iter(loader); self.dev=dev
        self.stream=torch.cuda.Stream(device=dev,priority=-1)
        self.qb,self.qe=[],[]
        for _ in range(2): self._prefetch()

    def _prefetch(self):
        try: vol, seg = next(self.it)
        except StopIteration: return
        with torch.cuda.stream(self.stream):
            vol = vol.pin_memory() if not vol.is_pinned() else vol
            vol_g = vol.to(self.dev, memory_format=torch.channels_last_3d, non_blocking=True)
            if seg is not None:
                seg = seg.pin_memory() if not seg.is_pinned() else seg
                seg_g = seg.to(self.dev, non_blocking=True)
            else: seg_g=None
            ev=torch.cuda.Event(); ev.record(self.stream)
            self.qb.append((vol_g, seg_g)); self.qe.append(ev)

    def __iter__(self): return self
    def __next__(self):
        if not self.qb: raise StopIteration
        b=self.qb.pop(0); self.qe.pop(0).wait(); self._prefetch(); return b

# ====================== pos proj ======================
@torch.no_grad()
def build_pos_projector_from_pe(mae, pos_var: float, pos_rank: int):
    """Original PE SVD (for patch/postpos)."""
    pe = mae.encoder.pos_embed[:,1:,:].squeeze(0).float()
    _, S, Vh = torch.linalg.svd(pe, full_matrices=False)
    if pos_rank>0:
        r = min(pos_rank, Vh.shape[0])
    else:
        expl = (S**2).cumsum(0)/(S**2).sum()
        r = max(int((expl<=pos_var).sum().item()), 1)
    return Vh[:r].contiguous()

@torch.no_grad()
def build_pos_projector_from_samples(tokens_sample, coords, pos_rank:int, pos_var:float):
    """
    tokens_sample: (N,C) float32
    coords: (T,3) in [0,1]; repeated outside
    Returns U_pos (r x C), orthonormal rows spanning positional subspace.
    """
    X = tokens_sample.float()
    Y = coords.repeat(X.shape[0]//coords.shape[0]+1, 1)[:X.shape[0]].to(X.device)
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    W = torch.linalg.lstsq(Xc, Yc).solution  # C x 3
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    if pos_rank>0:
        r = min(pos_rank, U.shape[1])
    else:
        expl = (S**2).cumsum(0)/(S**2).sum()
        r = max(int((expl<=pos_var).sum().item()), 1)
    return U[:, :r].T.contiguous()

def split_pos_subspace(x_flat, U):
    if U is None:
        return x_flat, torch.zeros_like(x_flat)
    proj = (x_flat @ U.t()) @ U
    return x_flat - proj, proj

def pos_corr(enc_w, U):
    num = torch.norm(enc_w @ U.t(), dim=1)
    den = torch.norm(enc_w, dim=1)+1e-12
    c = num/den
    return float(c.mean().clamp(0,1)), float(c.max().clamp(0,1))

# -------- Graph consistency loss -------- #
def graph_consistency_loss(p_grid):
    """
    p_grid: (B,Pd,Ph,Pw,H) probabilities over neurons.
    Returns scalar tensor.
    """
    loss = 0.0
    cnt = 0
    if p_grid.size(1) > 1:  # depth
        diff = p_grid[:,1:,:,:,:] - p_grid[:,:-1,:,:,:]
        loss += (diff.pow(2).mean())
        cnt += 1
    if p_grid.size(2) > 1:  # height
        diff = p_grid[:,:,1:,:,:] - p_grid[:,:,:-1,:,:]
        loss += (diff.pow(2).mean())
        cnt += 1
    if p_grid.size(3) > 1:  # width
        diff = p_grid[:,:,:,1:,:] - p_grid[:,:,:,:-1,:]
        loss += (diff.pow(2).mean())
        cnt += 1
    return loss / max(cnt,1)

# ====================== SAE ======================
class ConvTopKSAE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, k=0, relu=True, momentum=0.01):
        super().__init__()
        self.k=k; self.relu=relu; self.mom=momentum
        self.register_buffer("mu",    torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))
        self.enc_w = torch.nn.Parameter(torch.empty(hidden_dim, in_dim))
        self.enc_b = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.dec_w = torch.nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.dec_b = torch.nn.Parameter(torch.zeros(in_dim))
        torch.nn.init.kaiming_uniform_(self.enc_w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.dec_w, a=math.sqrt(5))

    @torch.no_grad()
    def update_stats(self, mu_b, std_b):
        self.mu    = (1-self.mom)*self.mu    + self.mom*mu_b
        self.sigma = (1-self.mom)*self.sigma + self.mom*std_b

    def forward(self, x_fp32):
        x_hat  = (x_fp32 - self.mu) / self.sigma
        z_lin  = F.linear(x_hat, self.enc_w, self.enc_b)
        z_relu = F.relu(z_lin, inplace=False) if self.relu else z_lin
        z_topk = topk_st(z_relu, self.k) if self.k>0 else z_relu
        rec_hat = F.linear(z_topk, self.dec_w, self.dec_b)
        x_rec   = rec_hat * self.sigma + self.mu
        return x_rec, z_topk, z_relu

# ====================== mono metrics ======================
@torch.no_grad()
def patch_labels_from_seg(seg_mask, p, num_classes=4):
    B,D,H,W = seg_mask.shape
    pd,ph,pw = D//p, H//p, W//p
    seg = seg_mask.view(B,pd,p,ph,p,pw,p).permute(0,1,3,5,2,4,6).contiguous()
    seg = seg.view(B,pd,ph,pw,p*p*p)
    counts = torch.zeros(B,pd,ph,pw,num_classes, device=seg.device, dtype=torch.int32)
    for c in range(num_classes):
        counts[...,c]=(seg==c).sum(-1)
    return counts.argmax(-1).view(-1)

@torch.no_grad()
def get_patch_xyz(L, grid_shape):
    d,h,w = grid_shape
    z,y,x = torch.meshgrid(
        torch.linspace(0,1,d),
        torch.linspace(0,1,h),
        torch.linspace(0,1,w),
        indexing="ij")
    return torch.stack([z,y,x], -1).reshape(-1,3)

@torch.no_grad()
def auc_trap(fpr, tpr):
    idx = torch.argsort(fpr)
    fpr = fpr[idx]; tpr = tpr[idx]
    df = fpr[1:] - fpr[:-1]
    return float((df * (tpr[1:]+tpr[:-1]) / 2).sum().clamp(0,1))

@torch.no_grad()
def binary_mi(tp, fp, fn, tn, eps: float = 1e-12, device=None):
    p = torch.tensor([tp, fp, fn, tn], device=device, dtype=torch.float32) + eps
    p = p / p.sum()
    p11, p10, p01, p00 = p
    px = torch.stack([p11+p10, p01+p00])
    py = torch.stack([p11+p01, p10+p00])
    Hx  = -(px * torch.log2(px)).sum()
    Hy  = -(py * torch.log2(py)).sum()
    Hxy = -(p  * torch.log2(p )).sum()
    return float((Hx + Hy - Hxy).clamp_min(0))

@torch.no_grad()
def monosemantic_metrics_full(z_abs, labels, num_classes, thr, coords=None):
    N,L = z_abs.shape
    device = z_abs.device
    labs = labels.to(device)

    means = torch.zeros(L, num_classes, device=device)
    for c in range(num_classes):
        idx = labs==c
        if idx.any(): means[:,c]=z_abs[idx].mean(0)

    sum_means = means.sum(1)+1e-12
    max_means, argmax = means.max(1)
    purity = (max_means/sum_means).cpu()

    second_means,_ = torch.topk(means, k=2, dim=1)
    sel = ((second_means[:,0]-second_means[:,1]) /
           (second_means[:,0]+second_means[:,1]+1e-12)).cpu()

    probs = (means / sum_means.unsqueeze(1)).clamp_min(1e-12)
    ent = -(probs*torch.log2(probs)).sum(1).cpu()

    top2_share = second_means.sum(1)/sum_means
    top2_share = top2_share.cpu()

    act = (z_abs > thr)
    iou = torch.zeros(L, device=device)
    for n in range(L):
        c = argmax[n]
        pred = act[:,n]; gt = (labs==c)
        inter = (pred & gt).sum().float()
        union = (pred | gt).sum().float() + 1e-12
        iou[n]=inter/union
    iou=iou.cpu()

    aurocs=[]; mis=[]; iff_scores=[]
    thr_grid = torch.linspace(0,1,40,device=device)
    for n in range(L):
        c = argmax[n]
        y = (labs==c).float()
        scores = z_abs[:,n]
        qs = torch.quantile(scores, thr_grid)
        tpr=[]; fpr=[]
        for t in qs:
            pred = (scores>=t).float()
            tp = (pred*y).sum(); fp = (pred*(1-y)).sum()
            fn = ((1-pred)*y).sum(); tn = ((1-pred)*(1-y)).sum()
            tpr.append(tp/(tp+fn+1e-12))
            fpr.append(fp/(fp+tn+1e-12))
        tpr = torch.stack(tpr); fpr=torch.stack(fpr)
        aurocs.append(auc_trap(fpr,tpr))
        pred_bin = (scores>thr).float()
        tp = (pred_bin*y).mean(); fp = (pred_bin*(1-y)).mean()
        fn = ((1-pred_bin)*y).mean(); tn = ((1-pred_bin)*(1-y)).mean()
        mis.append(binary_mi(tp, fp, fn, tn, device=device))
        precision = tp/(tp+fp+1e-12); recall = tp/(tp+fn+1e-12)
        iff_scores.append((precision*recall).item())

    aurocs=torch.tensor(aurocs); mis=torch.tensor(mis); iff_scores=torch.tensor(iff_scores)

    if coords is not None:
        zc = z_abs - z_abs.mean(0,keepdim=True)
        inv_scores=[]
        for axis in range(3):
            a = coords[:,axis].to(device)-coords[:,axis].mean()
            num = (zc * a.unsqueeze(1)).sum(0)**2
            den = (zc.pow(2).sum(0) * a.pow(2).sum())
            r2 = (num/(den+1e-12)).cpu()
            inv_scores.append(r2)
        pos_r2 = torch.stack(inv_scores,1).max(1).values
    else:
        pos_r2 = torch.zeros(L)

    agg = {
        "mono_purity_mean": purity.mean().item(),
        "mono_purity_90th": purity.quantile(0.9).item(),
        "mono_iou_mean":    iou.mean().item(),
        "mono_iou_90th":    iou.quantile(0.9).item(),
        "mono_frac_purity_gt_0.8": (purity>0.8).float().mean().item(),
        "mono_frac_iou_gt_0.3":    (iou>0.3).float().mean().item(),
        "sel_mean": sel.mean().item(),
        "sel_90th": sel.quantile(0.9).item(),
        "entropy_mean": ent.mean().item(),
        "entropy_10th": ent.quantile(0.1).item(),
        "mi_bits_mean": mis.mean().item(),
        "mi_bits_90th": mis.quantile(0.9).item(),
        "auroc_mean": aurocs.mean().item(),
        "auroc_90th": aurocs.quantile(0.9).item(),
        "pos_r2_mean": pos_r2.mean().item(),
        "pos_r2_90th": pos_r2.quantile(0.9).item(),
        "top2_share_mean": top2_share.mean().item(),
        "iff_mean": iff_scores.mean().item(),
        "iff_90th": iff_scores.quantile(0.9).item(),
    }

    rows=[]
    topk_idx = torch.topk(purity, k=min(40, L)).indices.tolist()
    for rank, idx in enumerate(topk_idx,1):
        rows.append({
            "rank":rank,"neuron":idx,"class":int(argmax[idx]),
            "purity":float(purity[idx]),"iou":float(iou[idx]),
            "selectivity":float(sel[idx]),"entropy":float(ent[idx]),
            "mi_bits":float(mis[idx]),"auroc":float(aurocs[idx]),
            "pos_r2":float(pos_r2[idx]),"top2_share":float(top2_share[idx]),
            "iff":float(iff_scores[idx]),
            "mean_act_best":float(max_means[idx]),"mean_act_total":float(sum_means[idx]),
        })
    return agg, rows, purity, iou, sel, ent, mis, aurocs, pos_r2, iff_scores, argmax.cpu()

# ====================== viz helpers ======================
@torch.no_grad()
def make_slice_mosaic(vols, mae_rec, sae_rec, step, outdir="vis"):
    Path(outdir).mkdir(exist_ok=True)
    B,_,D,H,W = vols.shape
    mids=(D//2,H//2,W//2)
    gmin=min(vols.min().item(), mae_rec.min().item(), sae_rec.min().item())
    gmax=max(vols.max().item(), mae_rec.max().item(), sae_rec.max().item())
    fig,ax=plt.subplots(3*B,3,figsize=(7,2.1*B),gridspec_kw={"wspace":.01,"hspace":.01})
    for b in range(B):
        for r,(mid,lbl) in enumerate(zip(mids,["Z","Y","X"])):
            row=3*b+r
            for c,img in enumerate([vols[b,0], mae_rec[b,0], sae_rec[b,0]]):
                sl=[slice(None)]*3; sl[r]=mid
                ax[row,c].imshow(img[tuple(sl)], cmap="gray", vmin=gmin, vmax=gmax)
                ax[row,c].axis("off")
                if r==0: ax[row,c].set_title(["GT","MAE(masked)","SAE(masked)"][c],fontsize=8)
            ax[row,0].set_ylabel(lbl,fontsize=8,rotation=0,labelpad=14)
    fig.suptitle(f"Step {step}",fontsize=10)
    fn=Path(outdir)/f"vis_{step}.png"
    fig.savefig(fn,dpi=120,bbox_inches="tight"); plt.close(fig)
    return fn

def panel_frame(gt_slice, mae_slice, sae_slice_color):
    gt = np.stack([gt_slice]*3, -1) if gt_slice.ndim==2 else gt_slice
    mae= np.stack([mae_slice]*3,-1) if mae_slice.ndim==2 else mae_slice
    return np.concatenate([gt, mae, sae_slice_color], axis=1)

@torch.no_grad()
def make_3d_gif_triplet(gt, mae, sae, owner_vol, colors, alpha=0.35,
                        axis='z', stride=2, max_frames=80, upscale=1):
    """
    Returns ndarray uint8 (T,H,W,3) instead of writing to disk.
    """
    gt = gt.numpy(); mae = mae.numpy(); sae = sae.numpy(); own = owner_vol.numpy()

    if axis=='z':
        num = gt.shape[0]; slicer = lambda i: (i, slice(None), slice(None))
    elif axis=='y':
        num = gt.shape[1]; slicer = lambda i: (slice(None), i, slice(None))
    else:
        num = gt.shape[2]; slicer = lambda i: (slice(None), slice(None), i)

    frames=[]
    step_idx = list(range(0,num,stride))[:max_frames]
    vmin = min(gt.min(), mae.min(), sae.min())
    vmax = max(gt.max(), mae.max(), sae.max())

    for i in step_idx:
        g = gt[slicer(i)]
        m = mae[slicer(i)]
        s = sae[slicer(i)]
        own_s = own[slicer(i)]

        g_img = (g - vmin) / (vmax - vmin + 1e-12)
        m_img = (m - vmin) / (vmax - vmin + 1e-12)
        s_img = (s - vmin) / (vmax - vmin + 1e-12)

        h,w = s_img.shape
        col = np.zeros((h,w,3), dtype=np.float32)
        mask = own_s>=0
        if mask.any():
            nid = own_s[mask]
            col[mask] = colors[(nid % len(colors))]
        sae_col = (1-alpha)*np.stack([s_img]*3,-1) + alpha*col

        frame = (panel_frame(g_img, m_img, sae_col)*255).astype(np.uint8)
        frames.append(frame)

    frames = np.stack(frames,0)  # (T,H,W,3)
    if upscale > 1:
        frames = np.repeat(np.repeat(frames, upscale, axis=1), upscale, axis=2)
    return frames

@torch.no_grad()
def color_patch_overlay_batch(vols, owner_full, mask_keep, grid_shape, patch_size, step, outdir="vis_color"):
    Path(outdir).mkdir(exist_ok=True)
    B,_,D,H,W = vols.shape
    Pd,Ph,Pw = grid_shape
    pd=ph=pw=patch_size

    colors = plt.cm.tab20(np.linspace(0,1,20))
    paths=[]
    for b in range(B):
        vol = vols[b,0]
        mids=(D//2,H//2,W//2)
        fig,ax=plt.subplots(3,1,figsize=(3,6),gridspec_kw={"wspace":.01,"hspace":.01})
        for r,(mid,lbl) in enumerate(zip(mids,["Z","Y","X"])):
            sl=[slice(None)]*3; sl[r]=mid
            img = vol[tuple(sl)].cpu().numpy()
            ax[r].imshow(img, cmap="gray")
            for zz in range(Pd):
                for yy in range(Ph):
                    for xx in range(Pw):
                        idx = zz*Ph*Pw + yy*Pw + xx
                        if not mask_keep[b, idx]:  # only color kept patches
                            continue
                        nid = int(owner_full[b, idx])
                        if nid < 0: continue
                        c = colors[nid % 20]
                        z0,y0,x0 = zz*pd, yy*ph, xx*pw
                        z1,y1,x1 = z0+pd, y0+ph, x0+pw
                        if r==0 and not (z0<=mid<z1): continue
                        if r==1 and not (y0<=mid<y1): continue
                        if r==2 and not (x0<=mid<x1): continue
                        if r==0: y0p,y1p,x0p,x1p = y0,y1,x0,x1
                        elif r==1: y0p,y1p,x0p,x1p = z0,z1,x0,x1
                        else: y0p,y1p,x0p,x1p = z0,z1,y0,y1
                        rect = plt.Rectangle((x0p, y0p),
                                             x1p-x0p, y1p-y0p,
                                             linewidth=1, edgecolor=c,
                                             facecolor=(c[0],c[1],c[2],0.25))
                        ax[r].add_patch(rect)
            ax[r].axis("off"); ax[r].set_title(lbl,fontsize=8)
        fn=Path(outdir)/f"color_{step}_b{b}.png"
        fig.savefig(fn,dpi=120,bbox_inches="tight"); plt.close(fig)
        paths.append(fn)
    return paths

def build_owner_volume(owner_L: torch.Tensor, grid_shape, patch_size, vol_shape):
    Pd,Ph,Pw = grid_shape
    pd=ph=pw=patch_size
    D,H,W = vol_shape
    vol = torch.full((D,H,W), -1, dtype=torch.int32)
    idx = 0
    for zz in range(Pd):
        for yy in range(Ph):
            for xx in range(Pw):
                nid = int(owner_L[idx])
                z0,y0,x0 = zz*pd, yy*ph, xx*pw
                z1,y1,x1 = z0+pd, y0+ph, x0+pw
                if nid >= 0:
                    vol[z0:z1, y0:y1, x0:x1] = nid
                idx += 1
    return vol

def to_wandb_video(frames_uint8, fps, fmt):
    data = frames_uint8.transpose(0,3,1,2)  # T,C,H,W
    return wandb.Video(data, fps=fps, format=fmt)

# ====================== mae helpers ======================
def build_mae(arch,img,p,dev):
    return ARCHS[arch](volume_size=(img,)*3, patch_size=(p,)*3, in_chans=1, mask_ratio=0.0).to(dev)

@torch.no_grad()
def get_tokens_layer(mae, vols, hook_layer:int):
    """
    hook_layer:
      -1 = patch_embed (before adding pos)
      -2 = postpos     (after adding pos, before blocks)
      >=0 = after encoder.blocks[hook_layer] (post-block, pre-norm except last)
    returns (B,L,C)
    """
    x = mae.encoder.patch_embed(vols)                 # B,L,C
    pe = mae.encoder.pos_embed[:,1:,:]
    if hook_layer == -1:
        return x
    x = x + pe
    if hook_layer == -2:
        return x
    cls_tok = mae.encoder.cls_token + mae.encoder.pos_embed[:,:1,:]
    x_full = torch.cat([cls_tok.expand(vols.size(0),-1,-1), x], dim=1)
    for i, blk in enumerate(mae.encoder.blocks):
        x_full = blk(x_full)
        if i == hook_layer:
            return x_full[:,1:,:]
    x_full = mae.encoder.norm(x_full)
    return x_full[:,1:,:]

# ====================== train ======================
def train(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark=True

    vol_shards=sorted(Path(cfg.shard_dir).glob("shard_*.tar"))
    assert vol_shards, "no shards"
    mask_shards=sorted(Path(cfg.mask_dir).glob("shard_*.tar")) if cfg.mask_dir else None
    if mask_shards: assert len(mask_shards)==len(vol_shards)

    val_n=max(1,int(len(vol_shards)*cfg.val_split))
    tr_vol, val_vol = vol_shards[val_n:], vol_shards[:val_n]
    tr_mask = mask_shards[val_n:] if mask_shards else None
    val_mask= mask_shards[:val_n] if mask_shards else None

    mae=load_model_checkpoint(cfg.checkpoint, build_mae(cfg.arch,cfg.img_size,cfg.patch_size,dev))
    mae.eval().requires_grad_(False)
    mae_dtype=next(mae.parameters()).dtype

    dummy=torch.zeros(1,1,cfg.img_size,cfg.img_size,cfg.img_size,device=dev,dtype=mae_dtype)
    t0=get_tokens_layer(mae,dummy,cfg.hook_layer)
    C=t0.shape[-1]; L=t0.shape[1]
    del dummy,t0

    patch_grid = (cfg.img_size//cfg.patch_size,)*3
    coords = get_patch_xyz(L, patch_grid).to(dev) if cfg.pos_metrics or cfg.project_pos else None

    # >>> Build positional projector (Option B)
    U_pos=None
    if cfg.project_pos:
        if cfg.hook_layer < 0:
            U_pos = build_pos_projector_from_pe(mae,cfg.pos_var,cfg.pos_rank).to(dev)
        else:
            sample_loader=DataLoader(TarShardDataset(val_vol,cfg.img_size,vps=min(cfg.vols_per_shard,1024)),
                                     batch_size=cfg.pos_proj_batch,num_workers=0)
            s_vols,_ = next(iter(sample_loader))
            s_vols = s_vols.to(dev,dtype=mae_dtype)
            toks_s = get_tokens_layer(mae,s_vols,cfg.hook_layer).float().reshape(-1,C)
            U_pos = build_pos_projector_from_samples(toks_s, coords, cfg.pos_rank, cfg.pos_var).to(dev)
        print(f"[POS] rank={U_pos.shape[0]} projected.")

    H=int(cfg.latent_mul*C)
    sae=ConvTopKSAE(C,H,k=cfg.k_sparse,relu=not cfg.no_relu,momentum=cfg.whiten_mom).to(dev,dtype=torch.float32)
    if cfg.compile_sae: sae=torch.compile(sae)

    opt=torch.optim.AdamW(sae.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.steps,eta_min=cfg.lr*0.1)

    wandb.init(project=cfg.project,name=cfg.run,config=vars(cfg))
    wandb.define_metric("step"); wandb.define_metric("*",step_metric="step")

    hist=deque(maxlen=100); step=0; stop={"s":False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("s",True))

    best_val = float("inf")
    best_path = Path(cfg.out).with_suffix(".best.pt")

    Pd,Ph,Pw = patch_grid

    ds=TarShardDataset(tr_vol,cfg.img_size,shuffle=(cfg.epoch_shuffle and step>0),
                           vps=cfg.vols_per_shard,mask_shards=tr_mask)
    loader=DataLoader(ds,batch_size=cfg.batch_size,num_workers=cfg.workers,
                        pin_memory=True,pin_memory_device='cuda',drop_last=True,persistent_workers=True,
                        multiprocessing_context="spawn",timeout=cfg.timeout,prefetch_factor=2)
    pf=CUDAPrefetcher(loader,dev)

    while step<cfg.steps and not stop["s"]:
        for vols, seg in pf:
            if stop["s"] or step>=cfg.steps: break

            vols=vols.to(dev,dtype=mae_dtype,non_blocking=True)

            # ----- MAE fwd to get mask/ids -----
            with torch.inference_mode():
                out = mae(vols, mask_ratio=cfg.train_mask_ratio)
                if len(out)==4:
                    _, pred_mae_tr, mask_mae_tr, ids_restore_tr = out
                else:
                    _, pred_mae_tr, mask_mae_tr = out
                    ids_restore_tr = None
                if ids_restore_tr is None:
                    ids_keep_tr, ids_restore_tr, _ = build_keep_restore(mask_mae_tr)
                else:
                    ids_keep_tr, _, _ = build_keep_restore(mask_mae_tr)

            toks=get_tokens_layer(mae,vols,cfg.hook_layer).float()
            toks_flat = toks.reshape(-1,C)
            toks_sem, toks_pos = split_pos_subspace(toks_flat, U_pos)

            sae.update_stats(toks_sem.mean(0), toks_sem.std(0).clamp_min(1e-6))

            opt.zero_grad(set_to_none=True)
            rec_sem_flat, z_topk, z_relu = sae(toks_sem)
            mse_proj = F.mse_loss(rec_sem_flat,toks_sem)

            with torch.no_grad():
                rec_dense_sem = (F.linear(z_relu, sae.dec_w, sae.dec_b) * sae.sigma + sae.mu)
                mse_dense_no_topk = F.mse_loss(rec_dense_sem, toks_sem).item()
                energy_ratio = (z_topk.abs().sum() / (z_relu.abs().sum()+1e-12)).item()

            rec_raw_flat = rec_sem_flat + toks_pos
            mse_raw = F.mse_loss(rec_raw_flat, toks_flat).item()

            l1 = z_topk.abs().mean() if cfg.l1>0 else toks_sem.new_tensor(0.)
            ort= orthogonality_loss(sae.dec_w) if cfg.ortho>0 else toks_sem.new_tensor(0.)

            # ----- Graph consistency loss -----
            if cfg.gc_lambda > 0:
                B = toks.shape[0]
                Hhid = z_relu.shape[1]
                probs = F.softmax(z_relu / cfg.gc_tau, dim=1)  # (B*L, H)
                probs_grid = probs.view(B, Pd, Ph, Pw, Hhid)
                gc_loss = graph_consistency_loss(probs_grid)
            else:
                gc_loss = toks_sem.new_tensor(0.)

            loss = mse_proj + cfg.l1*l1 + cfg.ortho*ort + cfg.gc_lambda*gc_loss
            loss.backward()
            gnorm=torch.nn.utils.clip_grad_norm_(sae.parameters(),cfg.grad_clip).item()
            opt.step(); sched.step()
            step+=1; hist.append(float(loss))

            if step%cfg.log_int==0:
                var=toks_sem.var().item()
                r2=1-mse_proj.item()/(var+1e-12)
                dead=(sae.enc_w.norm(dim=1)<1e-5).float().mean().item()
                k_eff=(z_topk.abs()>0).sum(1).float().mean().item()
                frac_act = k_eff / z_topk.shape[1]
                raw_std=toks.std().item()
                pos_m=pos_M=0.
                if U_pos is not None: pos_m,pos_M=pos_corr(sae.enc_w,U_pos)
                if cfg.k_sparse>0:
                    mx = (z_topk.abs()>0).sum(1).max().item()
                    assert mx <= cfg.k_sparse+5, f"Top-k broken: saw {mx}"

                wandb.log({
                    "step":step,
                    "train_loss":sum(hist)/len(hist),
                    "train_mse":mse_proj.item(),
                    "mse_dense_no_topk":mse_dense_no_topk,
                    "mse_raw_tokens":mse_raw,
                    "energy_ratio_topk":energy_ratio,
                    "train_l1":l1.item(),
                    "ortho_loss":ort.item(),
                    "graph_cons_loss":gc_loss.item(),
                    "lr":sched.get_last_lr()[0],
                    "token_mean":toks_sem.mean().item(),
                    "token_std":toks_sem.std().item(),
                    "raw_token_std":raw_std,
                    "token_R2":r2,
                    "sigma_mean":sae.sigma.mean().item(),
                    "sigma_std":sae.sigma.std().item(),
                    "sae_grad_norm":gnorm,
                    "dead_frac":dead,
                    "frac_active":frac_act,
                    "k_eff":k_eff,
                    "latent_dim":H,
                    "pos_corr_mean":pos_m,
                    "pos_corr_max":pos_M,
                })

            # ---------- viz ----------
            try:
                if step%cfg.vis_int==0 or step==1:
                    sae.eval(); torch.cuda.empty_cache()
                    with torch.inference_mode():
                        v_loader=DataLoader(TarShardDataset(val_vol,cfg.img_size,vps=cfg.vols_per_shard,mask_shards=val_mask),
                                            batch_size=cfg.vis_n,num_workers=0, pin_memory=False)
                        v_vols,v_mask = next(iter(v_loader))
                        v_vols=v_vols.to(dev,dtype=mae_dtype)
                        v_mask=v_mask.to(dev) if v_mask is not None else None

                        out = mae(v_vols, mask_ratio=cfg.vis_mask_ratio)
                        if len(out)==4:
                            _, pred_mae, mask_mae, ids_restore_mae = out
                        else:
                            _, pred_mae, mask_mae = out
                            ids_restore_mae = None
                        if ids_restore_mae is None:
                            ids_keep_v, ids_restore_mae, _ = build_keep_restore(mask_mae)
                        else:
                            ids_keep_v, _, _ = build_keep_restore(mask_mae)

                        toks_full=get_tokens_layer(mae,v_vols,cfg.hook_layer).float()
                        toks_full_flat = toks_full.reshape(-1,C)
                        toks_sem_flat, toks_pos_flat = split_pos_subspace(toks_full_flat, U_pos)
                        toks_sem = toks_sem_flat.view_as(toks_full)

                        kept_v = torch.gather(toks_sem,1,ids_keep_v.unsqueeze(-1).expand(-1,-1,C))
                        Bv,Kv,_ = kept_v.shape
                        flat_kept = kept_v.reshape(-1,C)

                        rec_chunks=[]; z_chunks=[]
                        if cfg.vis_chunk_tokens and flat_kept.shape[0]>cfg.vis_chunk_tokens:
                            for s in range(0,flat_kept.shape[0],cfg.vis_chunk_tokens):
                                r,z_t,_ = sae(flat_kept[s:s+cfg.vis_chunk_tokens])
                                rec_chunks.append(r); z_chunks.append(z_t)
                            rec_kept_sem=torch.cat(rec_chunks,0).view(Bv,Kv,C)
                            z_kept_fp32=torch.cat(z_chunks,0).view(Bv,Kv,-1)
                        else:
                            rec_kept_sem, z_kept_fp32, _ = sae(flat_kept)
                            rec_kept_sem=rec_kept_sem.view(Bv,Kv,C)
                            z_kept_fp32  = z_kept_fp32.view(Bv, Kv, -1)

                        hook_mse=F.mse_loss(rec_kept_sem, kept_v).item()
                        num=((rec_kept_sem-rec_kept_sem.mean(1,True))*(kept_v-kept_v.mean(1,True))).sum((1,2))
                        den=(rec_kept_sem.var(1,False).sum(1).sqrt()*kept_v.var(1,False).sum(1).sqrt()+1e-12)
                        kept_corr=(num/den).mean().item()

                        # ---- Decoder path ----
                        pos_full = toks_pos_flat.view_as(toks_full)
                        pos_keep = torch.gather(pos_full,1,ids_keep_v.unsqueeze(-1).expand(-1,-1,C))
                        rec_kept_tokens = (rec_kept_sem + pos_keep).to(mae_dtype)

                        pos_all = mae.encoder.pos_embed[:,1:,:].expand(Bv,-1,-1)
                        pos_keep_pe = torch.gather(pos_all,1,ids_keep_v.unsqueeze(-1).expand(-1,-1,C))

                        cls_tok = mae.encoder.cls_token + mae.encoder.pos_embed[:,:1,:]
                        x_enc = torch.cat([cls_tok.expand(Bv,-1,-1),
                                        rec_kept_tokens + pos_keep_pe.to(mae_dtype)], dim=1)
                        for blk in mae.encoder.blocks:
                            x_enc=blk(x_enc)
                        x_enc=mae.encoder.norm(x_enc)

                        dec_in = mae.decoder_embed(x_enc)
                        dec_no_cls = dec_in[:,1:,:]
                        mask_tokens_dec = mae.mask_token.repeat(Bv, L - Kv, 1)
                        dec_cat = torch.cat([dec_no_cls, mask_tokens_dec], dim=1)
                        dec_cat = torch.gather(dec_cat, 1, ids_restore_mae.unsqueeze(-1).expand(-1,-1,dec_in.shape[2]))
                        dec_full = torch.cat([dec_in[:,:1,:], dec_cat], dim=1)
                        dec_full = dec_full + mae.decoder_pos_embed
                        for blk in mae.decoder_blocks:
                            dec_full=blk(dec_full)
                        dec_full=mae.decoder_norm(dec_full)
                        dec_full=dec_full[:,1:,:]
                        if getattr(mae,"decoder_neck","linear")=="conv":
                            pred_sae=mae.decoder_pred(dec_full, mae.patch_grid_shape)
                        else:
                            pred_sae=mae.decoder_pred(dec_full)

                        # ---- pixel metrics & viz ----
                        gt_vol = v_vols.float().cpu()
                        pm = mae.unpatchify(pred_mae).float().cpu()
                        ps = mae.unpatchify(pred_sae).float().cpu()
                        pix_mse_sae_gt = F.mse_loss(ps, gt_vol).item()
                        pix_mse_mae_gt = F.mse_loss(pm, gt_vol).item()
                        pix_mse_mae_sae = F.mse_loss(ps, pm).item()

                        vis_path=make_slice_mosaic(gt_vol, pm, ps, step)

                        logv={
                            "step":step,
                            "pixel_mse_diff":pix_mse_mae_sae,
                            "pixel_mse_sae_gt": pix_mse_sae_gt,
                            "pixel_mse_mae_gt": pix_mse_mae_gt,
                            "hook_token_mse":hook_mse,
                            "kept_token_corr":kept_corr,
                            "k_eff_kept": (z_kept_fp32.abs()>0).sum(2).float().mean().item(),
                            "slice_grid":wandb.Image(str(vis_path)),
                        }

                        if v_mask is not None and cfg.mono_eval:
                            p=cfg.patch_size
                            labels=patch_labels_from_seg(v_mask,p,cfg.num_classes)
                            _, z_all_topk, _ = sae(toks_sem_flat)
                            coords_rep = coords.repeat(v_vols.shape[0],1) if coords is not None else None
                            agg, rows, purity, iou, sel, ent, mis, auroc, pos_r2, iff, argmax = monosemantic_metrics_full(
                                z_all_topk.abs(), labels, cfg.num_classes, cfg.mono_thr, coords_rep
                            )
                            logv.update(agg)
                            if cfg.log_hists_every and step%cfg.log_hists_every==0:
                                logv["purity_hist"]=wandb.Histogram(purity.numpy())
                                logv["iou_hist"]=wandb.Histogram(iou.numpy())
                                logv["sel_hist"]=wandb.Histogram(sel.numpy())
                                logv["entropy_hist"]=wandb.Histogram(ent.numpy())
                                logv["mi_hist"]=wandb.Histogram(mis.numpy())
                                logv["auroc_hist"]=wandb.Histogram(auroc.numpy())
                                logv["pos_r2_hist"]=wandb.Histogram(pos_r2.numpy())
                                logv["iff_hist"]=wandb.Histogram(iff.numpy())
                            table = wandb.Table(columns=list(rows[0].keys())) if rows else wandb.Table(["empty"])
                            for r in rows: table.add_data(*r.values())
                            logv["top_neurons"]=table

                            if cfg.color_neurons:
                                z_full = z_all_topk.abs().view(v_vols.shape[0], -1, z_all_topk.shape[1])  # (B,L,H)
                                owner_full = z_full.argmax(-1)
                                below = (z_full.max(-1).values < cfg.mono_thr)
                                owner_full[below] = -1
                                keep_bool = (mask_mae==0)
                                mask_bool = ~keep_bool
                                owner_masked = owner_full.clone()
                                owner_masked[keep_bool] = -1

                                paths2d = color_patch_overlay_batch(
                                    gt_vol, owner_masked.cpu(), mask_bool.cpu(), patch_grid, cfg.patch_size, step
                                )
                                logv["color_overlays"] = [wandb.Image(str(p)) for p in paths2d]

                                if cfg.vis_3d:
                                    colors = plt.cm.tab20(np.linspace(0,1,20))[:,:3]
                                    vids=[]
                                    for b in range(min(cfg.vis_3d_n, gt_vol.shape[0])):
                                        ov = build_owner_volume(owner_masked[b], patch_grid,
                                                                cfg.patch_size, gt_vol.shape[2:])
                                        frames = make_3d_gif_triplet(gt_vol[b,0], pm[b,0], ps[b,0],
                                                                    ov, colors,
                                                                    alpha=cfg.gif_alpha,
                                                                    axis='z',
                                                                    stride=cfg.gif_stride,
                                                                    max_frames=cfg.gif_max_frames,
                                                                    upscale=cfg.vid_upscale)
                                        vids.append(frames)
                                    if cfg.vid_format == "mp4":
                                        logv["videos_3d"] = [to_wandb_video(v, cfg.gif_fps, "mp4") for v in vids]
                                    else:
                                        logv["gifs_3d"]   = [to_wandb_video(v, cfg.gif_fps, "gif") for v in vids]

                        wandb.log(logv)

                        if pix_mse_sae_gt < best_val:
                            best_val = pix_mse_sae_gt
                            torch.save({
                                "enc_w":sae.enc_w.cpu(),"enc_b":sae.enc_b.cpu(),
                                "dec_w":sae.dec_w.cpu(),"dec_b":sae.dec_b.cpu(),
                                "mu":sae.mu.cpu(),"sigma":sae.sigma.cpu(),
                                "input_dim":C,"hidden_dim":H,"k":cfg.k_sparse,
                                "hook_point":cfg.hook_point,
                                "hook_layer":cfg.hook_layer,
                                "project_pos":cfg.project_pos,
                            }, best_path)
                            wandb.run.summary["best_pix_mse_sae_gt"]=best_val
                            wandb.run.summary["best_ckpt_path"]=str(best_path)

                        del (pred_mae, pred_sae, pm, ps, gt_vol, v_vols, v_mask,
                            toks_full, toks_full_flat, toks_sem_flat, toks_pos_flat,
                            kept_v, rec_kept_sem, z_kept_fp32,
                            flat_kept, x_enc, dec_in, dec_no_cls, dec_cat, dec_full,
                            mask_mae, ids_restore_mae, ids_keep_v, mask_tokens_dec,
                            pos_all, labels, z_all_topk, coords_rep, agg, rows, purity,
                            iou, sel, ent, mis, auroc, pos_r2, iff, argmax,
                            paths2d, vids)
                        torch.cuda.empty_cache(); gc.collect()
                    sae.train()
            except Exception as e:
                print(f"Error in viz: {e}")
                wandb.log({"error": str(e)})

            del vols, seg, toks, toks_flat, toks_sem, toks_pos, rec_sem_flat, z_topk, z_relu, rec_dense_sem, rec_raw_flat
            torch.cuda.empty_cache()

        del pf, loader, ds
        torch.cuda.empty_cache(); gc.collect()

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc_w":sae.enc_w.cpu(),"enc_b":sae.enc_b.cpu(),
        "dec_w":sae.dec_w.cpu(),"dec_b":sae.dec_b.cpu(),
        "mu":sae.mu.cpu(),"sigma":sae.sigma.cpu(),
        "input_dim":C,"hidden_dim":H,"k":cfg.k_sparse,
        "hook_point":cfg.hook_point,
        "hook_layer":cfg.hook_layer,
        "project_pos":cfg.project_pos,
    }, cfg.out)
    wandb.finish()
    print("Saved SAE to", cfg.out, "Best:", best_path if best_val<1e9 else "N/A")

# ====================== CLI ======================
def cli():
    P=argparse.ArgumentParser("Stage‑1 SAE recon sanity + mono metrics")
    # data
    P.add_argument("--shard_dir",required=True)
    P.add_argument("--mask_dir",default=None)
    P.add_argument("--vols_per_shard",type=int,default=16_384)
    # mae
    P.add_argument("--checkpoint",required=True)
    P.add_argument("--arch",choices=list(ARCHS.keys()),default="base_patch_conv")
    P.add_argument("--img_size",type=int,default=96)
    P.add_argument("--patch_size",type=int,default=8)
    # >>> hook configuration
    P.add_argument("--hook_point",choices=["patch","postpos"],default="patch")
    P.add_argument("--hook_layer",type=int,default=-1,
                   help="-1 patch_embed, -2 postpos, >=0 index of encoder block")
    # sae
    P.add_argument("--latent_mul",type=float,default=2.0)
    P.add_argument("--k_sparse",type=int,default=0)
    P.add_argument("--no_relu",action="store_true")
    P.add_argument("--l1",type=float,default=0.0)
    P.add_argument("--wd",type=float,default=1e-4)
    P.add_argument("--ortho",type=float,default=0.0)
    P.add_argument("--whiten_mom",type=float,default=0.01)
    # pos
    P.add_argument("--project_pos",action="store_true",default=False)
    P.add_argument("--pos_var",type=float,default=0.99)
    P.add_argument("--pos_rank",type=int,default=3)
    P.add_argument("--pos_metrics",action="store_true",default=True)
    P.add_argument("--pos_proj_batch",type=int,default=4, help="batches to estimate U_pos when hook_layer>=0")
    # graph consistency
    P.add_argument("--gc_lambda",type=float,default=0.0, help="weight of graph consistency loss")
    P.add_argument("--gc_tau",type=float,default=0.2, help="softmax temperature for neuron probs")
    # train
    P.add_argument("--compile_sae",action="store_true",default=True)
    P.add_argument("--batch_size",type=int,default=512)
    P.add_argument("--lr",type=float,default=1.5e-3)
    P.add_argument("--steps",type=int,default=10_000)
    P.add_argument("--workers",type=int,default=8)
    P.add_argument("--timeout",type=int,default=300)
    P.add_argument("--val_split",type=float,default=0.02)
    P.add_argument("--epoch_shuffle",action="store_true")
    P.add_argument("--grad_clip",type=float,default=1e9)
    P.add_argument("--train_mask_ratio",type=float,default=0.85)
    # viz
    P.add_argument("--vis_mask_ratio",type=float,default=0.85)
    P.add_argument("--vis_int",type=int,default=1000)
    P.add_argument("--vis_n",type=int,default=3)
    P.add_argument("--vis_chunk_tokens",type=int,default=200_000)
    P.add_argument("--color_neurons",action="store_true",default=True)
    # 3D video opts
    P.add_argument("--vis_3d",action="store_true",default=True)
    P.add_argument("--vis_3d_n",type=int,default=6)
    P.add_argument("--gif_stride",type=int,default=2)
    P.add_argument("--gif_max_frames",type=int,default=80)
    P.add_argument("--gif_alpha",type=float,default=0.35)
    P.add_argument("--gif_fps",type=int,default=16)
    P.add_argument("--vid_upscale",type=int,default=3)
    P.add_argument("--vid_format",choices=["mp4","gif"],default="mp4")
    # mono
    P.add_argument("--mono_eval",action="store_true",default=True)
    P.add_argument("--num_classes",type=int,default=4)
    P.add_argument("--mono_thr",type=float,default=0.1)
    P.add_argument("--log_hists_every",type=int,default=0)
    # logging/out
    P.add_argument("--project",default="sae_stage1")
    P.add_argument("--run",default="stage1_test")
    P.add_argument("--out",default="checkpoints/sae_stage1.pt")
    P.add_argument("--log_int",type=int,default=20)
    return P.parse_args()

if __name__=="__main__":
    train(cli())
