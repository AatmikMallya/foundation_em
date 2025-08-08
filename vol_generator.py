#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse (L1) Whitened SAE trainer with monosemantic metrics.
Mask-synced viz, CPU-offload, memory safe. 2025‑07‑25
"""

import argparse, random, signal, tarfile, math, gc, io
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import wandb, matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ------------------------- MAE builders -------------------------
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

# ------------------------- Checkpoint loader -------------------------
def load_model_checkpoint(path, model, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.","",1): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    if ckpt.get("model_dtype") == "torch.bfloat16":
        model = model.to(dtype=torch.bfloat16)
    if ckpt.get("model_dtype") == "torch.float16":
        model = model.to(dtype=torch.float16)
    return model.to(device)

# ------------------------- Dataset / Prefetcher -------------------------
class TarShardDataset(IterableDataset):
    """If mask_dir is provided, returns (vol, seg_mask). Else returns (vol, None)."""
    def __init__(self, vol_shards, vs, shuffle=False, vps=16_384, mask_shards=None):
        self.vols = list(vol_shards)
        self.masks = list(mask_shards) if mask_shards else None
        self.vs = vs; self.sh = shuffle; self.vps = vps
        print(f"Dataset: {len(self.vols)} vol shards × {vps} vols"
              + (f" with masks" if self.masks else ""))
    def __len__(self): return len(self.vols) * self.vps
    def _iter_shard_pair(self, vol_tar_path, mask_tar_path):
        vol_tf  = tarfile.open(vol_tar_path,  "r|", bufsize=32*1024*1024)
        mask_tf = tarfile.open(mask_tar_path, "r|", bufsize=32*1024*1024) if mask_tar_path else None
        try:
            vol_it  = (m for m in vol_tf if m.isfile())
            mask_it = (m for m in mask_tf if m.isfile()) if mask_tf else None
            for m_vol in vol_it:
                vol = np.frombuffer(vol_tf.extractfile(m_vol).read(), np.float32).reshape(self.vs, self.vs, self.vs)
                vol_t = torch.from_numpy(vol.copy()).unsqueeze(0).pin_memory()
                if mask_it:
                    # assume same order
                    m_mask = next(mask_it)
                    seg = np.frombuffer(mask_tf.extractfile(m_mask).read(), np.uint8).reshape(self.vs, self.vs, self.vs)
                    seg_t = torch.from_numpy(seg.copy()).pin_memory()
                else:
                    seg_t = None
                yield vol_t, seg_t
        finally:
            vol_tf.close()
            if mask_tf: mask_tf.close()

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        idxs = list(range(len(self.vols)))
        if self.sh: random.shuffle(idxs)
        if w: idxs = idxs[w.id::w.num_workers]
        for i in idxs:
            mask_path = self.masks[i] if self.masks else None
            yield from self._iter_shard_pair(self.vols[i], mask_path)

class CUDAPrefetcher:
    def __init__(self, loader, dev):
        self.it = iter(loader); self.dev = dev
        self.stream = torch.cuda.Stream(device=dev, priority=-1)
        self.qb, self.qe = [], []
        for _ in range(2): self._prefetch()
    def _prefetch(self):
        try: bc = next(self.it)
        except StopIteration: return
        with torch.cuda.stream(self.stream):
            # bc is (vol, mask) or (vol, None)
            vol = bc[0]
            if not vol.is_pinned(): vol = vol.pin_memory()
            vol_g = vol.to(self.dev, memory_format=torch.channels_last_3d, non_blocking=True)
            if bc[1] is not None:
                mask = bc[1]
                if not mask.is_pinned(): mask = mask.pin_memory()
                mask_g = mask.to(self.dev, non_blocking=True)
            else:
                mask_g = None
            ev = torch.cuda.Event(); ev.record(self.stream)
            self.qb.append((vol_g, mask_g)); self.qe.append(ev)
    def __iter__(self): return self
    def __next__(self):
        if not self.qb: raise StopIteration
        b = self.qb.pop(0); self.qe.pop(0).wait(); self._prefetch(); return b

# ------------------------- Sparse Whitened SAE -------------------------
class SparseWhitenedSAE(torch.nn.Module):
    def __init__(self, dim, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mu",    torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))
        self.enc_w = torch.nn.Parameter(torch.empty(dim, dim))
        self.enc_b = torch.nn.Parameter(torch.zeros(dim))
        self.dec_w = torch.nn.Parameter(torch.empty(dim, dim))
        self.dec_b = torch.nn.Parameter(torch.zeros(dim))
        torch.nn.init.kaiming_uniform_(self.enc_w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.dec_w, a=math.sqrt(5))
    @torch.no_grad()
    def update_stats(self, b_mu, b_std):
        self.mu    = (1-self.momentum)*self.mu    + self.momentum*b_mu
        self.sigma = (1-self.momentum)*self.sigma + self.momentum*b_std
    def forward(self, x):  # x fp32
        x_hat = (x - self.mu) / self.sigma
        z     = F.linear(x_hat, self.enc_w, self.enc_b)
        rec_w = F.linear(z,     self.dec_w, self.dec_b)
        x_rec = rec_w * self.sigma + self.mu
        return x_rec, z

# ------------------------- Monosemantic metrics -------------------------
@torch.no_grad()
def patch_labels_from_seg(seg_mask, mae, num_classes=4):
    """
    seg_mask: (B,D,H,W) uint8 labels {0..num_classes-1}
    returns patch_labels (B*L) long: majority label per patch
    """
    B,D,H,W = seg_mask.shape
    p = mae.patch_size[0]
    pd, ph, pw = D//p, H//p, W//p
    # reshape to grid of patches then flatten
    seg = seg_mask.view(B, pd, p, ph, p, pw, p).permute(0,1,3,5,2,4,6).contiguous() # (B, pd, ph, pw, p, p, p)
    seg = seg.view(B, pd, ph, pw, p*p*p)  # gather voxels per patch
    # majority vote
    counts = torch.zeros(B, pd, ph, pw, num_classes, device=seg.device, dtype=torch.int32)
    for cls in range(num_classes):
        counts[..., cls] = (seg==cls).sum(-1)
    patch_lab = counts.argmax(-1)  # (B,pd,ph,pw)
    return patch_lab.view(-1)      # B*L

@torch.no_grad()
def monosemantic_metrics(z, patch_labels, num_classes, thr):
    """
    z: (N, latent) activations (float32)
    patch_labels: (N,) long
    thr: threshold on |z| to call "active"
    Returns dict of aggregates + table rows
    """
    N, L = z.shape
    labs = patch_labels.to(z.device)
    # mean activation per class per neuron
    means = torch.zeros(L, num_classes, device=z.device)
    for c in range(num_classes):
        idx = (labs==c)
        if idx.any():
            means[:,c] = z[idx].abs().mean(0)
    sum_means = means.sum(1) + 1e-12
    max_means, argmax = means.max(1)
    purity = (max_means / sum_means).cpu()
    # IoU / F1
    active = (z.abs() > thr)  # (N,L)
    ious   = torch.zeros(L, device=z.device)
    f1s    = torch.zeros(L, device=z.device)
    for n in range(L):
        c = argmax[n]
        pred = active[:,n]
        gt   = (labs==c)
        inter = (pred & gt).sum().float()
        union = (pred | gt).sum().float() + 1e-12
        iou = (inter / union).item()
        ious[n] = iou
        tp = inter
        fp = (pred & ~gt).sum().float()
        fn = (~pred & gt).sum().float()
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1s[n] = (2*prec*rec/(prec+rec+1e-12)).item()

    agg = {
        "mono_purity_mean": purity.mean().item(),
        "mono_purity_90th": purity.quantile(0.9).item(),
        "mono_iou_mean": ious.mean().item(),
        "mono_iou_90th": ious.quantile(0.9).item(),
        "mono_frac_purity_gt_0.8": (purity>0.8).float().mean().item(),
        "mono_frac_iou_gt_0.3": (ious>0.3).float().mean().item(),
    }

    # top neurons
    topk = torch.topk(purity, k=min(20, L))
    rows = []
    for rank, idx in enumerate(topk.indices.tolist(), 1):
        rows.append({
            "rank": rank,
            "neuron": idx,
            "class": int(argmax[idx]),
            "purity": purity[idx].item(),
            "iou": ious[idx].item(),
            "f1": f1s[idx].item(),
            "mean_act_best": max_means[idx].item(),
            "mean_act_total": sum_means[idx].item()
        })
    return agg, rows, purity.cpu(), ious.cpu()

# ------------------------- Visual helper -------------------------
@torch.no_grad()
def make_slice_mosaic(vols, m0, m1, step, outdir="vis"):
    Path(outdir).mkdir(exist_ok=True)
    B,_,D,H,W = vols.shape
    mids = (D//2, H//2, W//2)

    global_min = min(vols.min().item(), m0.min().item(), m1.min().item())
    global_max = max(vols.max().item(), m0.max().item(), m1.max().item())

    fig, ax = plt.subplots(3*B, 3, figsize=(7, 2.1*B),
                           gridspec_kw={"wspace": .01, "hspace": .01})
    for b in range(B):
        for r,(mid,lbl) in enumerate(zip(mids,["Z","Y","X"])):
            row = 3*b + r
            for c,img in enumerate([vols[b,0], m0[b,0], m1[b,0]]):
                sl = [slice(None)]*3; sl[r]=mid
                ax[row,c].imshow(img[tuple(sl)], cmap="gray", vmin=global_min, vmax=global_max)
                ax[row,c].axis("off")
                if r==0: ax[row,c].set_title(["GT","MAE","SAE"][c], fontsize=8)
            ax[row,0].set_ylabel(lbl, fontsize=8, rotation=0, labelpad=14)

    fig.suptitle(f"Step {step}", fontsize=10)
    fn = Path(outdir)/f"vis_{step}.png"
    fig.savefig(fn, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fn

# ------------------------- Build MAE -------------------------
def build_mae(arch, img, p, dev):
    return ARCHS[arch](volume_size=(img,)*3, patch_size=(p,)*3, in_chans=1, mask_ratio=0.0).to(dev)

# ------------------------- Training -------------------------
def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vol_shards = sorted(Path(cfg.shard_dir).glob("shard_*.tar"))
    assert vol_shards, "No shard_*.tar in shard_dir"
    if cfg.mask_dir:
        mask_shards = sorted(Path(cfg.mask_dir).glob("shard_*.tar"))
        assert len(mask_shards)==len(vol_shards), "mask shards mismatch"
    else:
        mask_shards = None

    val_n = max(1, int(len(vol_shards)*cfg.val_split))
    tr_vol, val_vol = vol_shards[val_n:], vol_shards[:val_n]
    tr_mask = mask_shards[val_n:] if mask_shards else None
    val_mask = mask_shards[:val_n] if mask_shards else None

    mae  = load_model_checkpoint(cfg.checkpoint, build_mae(cfg.arch, cfg.img_size, cfg.patch_size, dev))
    mae.eval().requires_grad_(False)
    dtype = next(mae.parameters()).dtype

    # token dim
    dummy = torch.zeros(1,1,cfg.img_size,cfg.img_size,cfg.img_size, device=dev, dtype=dtype)
    acts  = []
    hk = mae.encoder.blocks[cfg.layer].register_forward_hook(lambda m,i,o: acts.append(o))
    mae.forward_encoder(dummy, mask_ratio=0.0); hk.remove()
    C = acts[0][:,1:].shape[-1]
    del acts, dummy; torch.cuda.empty_cache()

    sae = SparseWhitenedSAE(C).to(dev, dtype=torch.float32)
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr*0.1)

    wandb.init(project=cfg.project, name=cfg.run, config=vars(cfg))
    wandb.define_metric("step"); wandb.define_metric("*", step_metric="step")

    hist = deque(maxlen=100); step = 0; stop = {"s": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("s", True))

    # main loop
    while step < cfg.steps and not stop["s"]:
        ds = TarShardDataset(tr_vol, cfg.img_size, shuffle=(cfg.epoch_shuffle and step>0),
                             vps=cfg.vols_per_shard, mask_shards=tr_mask)
        loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.workers,
                            pin_memory=False, drop_last=True, persistent_workers=True,
                            multiprocessing_context="spawn", timeout=cfg.timeout)
        pf = CUDAPrefetcher(loader, dev)

        for vols, _ in pf:  # training doesn't need seg masks
            if stop["s"] or step >= cfg.steps: break
            vols = vols.to(dev, dtype=dtype, non_blocking=True)

            # hook tokens (no mask)
            acts=[]
            hk = mae.encoder.blocks[cfg.layer].register_forward_hook(lambda m,i,o: acts.append(o.detach()))
            mae.forward_encoder(vols, mask_ratio=0.0); hk.remove()
            toks = acts[0][:,1:].reshape(-1, C).float()
            del acts; torch.cuda.empty_cache()

            sae.update_stats(toks.mean(0), toks.std(0).clamp_min(1e-6))

            opt.zero_grad(set_to_none=True)
            rec, z = sae(toks)
            mse = F.mse_loss(rec, toks)
            l1  = z.abs().mean()
            loss = mse + cfg.l1 * l1
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e9).item()
            upd_ratio = sae.enc_w.grad.std().item() / (sae.enc_w.std().item() + 1e-12)
            opt.step(); sched.step()
            hist.append(float(loss)); step += 1

            if step % cfg.log_int == 0:
                var = toks.var().item()
                dead_frac = (sae.enc_w.norm(dim=1) < 1e-4).float().mean().item()
                frac_active = (z.abs()>1e-6).float().mean().item()
                wandb.log({
                    "step": step,
                    "train_loss": sum(hist)/len(hist),
                    "train_mse": mse.item(),
                    "train_l1": l1.item(),
                    "lr": sched.get_last_lr()[0],
                    "token_mean": toks.mean().item(),
                    "token_std": toks.std().item(),
                    "token_R2": 1.0 - mse.item()/var,
                    "sigma_mean": sae.sigma.mean().item(),
                    "sigma_std": sae.sigma.std().item(),
                    "sae_grad_norm": g_norm,
                    "update_ratio": upd_ratio,
                    "dead_frac": dead_frac,
                    "frac_active": frac_active,
                })
            del rec, z, toks; torch.cuda.empty_cache()

            # -------- VIS + MONO METRICS --------
            if step % cfg.vis_int == 0:
                sae.eval()
                torch.cuda.empty_cache()

                with torch.no_grad():
                    v_loader = DataLoader(TarShardDataset(val_vol, cfg.img_size, vps=cfg.vols_per_shard,
                                                          mask_shards=val_mask),
                                          batch_size=cfg.vis_n, num_workers=0)
                    v_vols, v_mask = next(iter(v_loader))
                    v_vols = v_vols.to(dev, dtype=dtype)
                    v_mask = v_mask.to(dev) if v_mask is not None else None

                    cpu_state  = torch.random.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state(dev)

                    kept_list=[]
                    def cap_hook(_m,_i,out): kept_list.append(out.detach())

                    hk_cap = mae.encoder.blocks[cfg.layer].register_forward_hook(cap_hook)
                    base_loss, pred_mae, mask, _ = mae(v_vols, mask_ratio=cfg.vis_mask_ratio)
                    hk_cap.remove()
                    kept_tokens = kept_list[0]
                    kept_orig   = kept_tokens[:,1:,:].float()  # (B,K,C)
                    flat = kept_orig.view(-1, C)

                    # SAE on kept tokens
                    if cfg.vis_chunk_tokens and flat.shape[0] > cfg.vis_chunk_tokens:
                        rec_chunks=[]
                        for s in range(0, flat.shape[0], cfg.vis_chunk_tokens):
                            rec_chunks.append(sae(flat[s:s+cfg.vis_chunk_tokens])[0])  # only rec
                        kept_rec = torch.cat(rec_chunks, 0).to(dtype).view_as(kept_orig)
                        del rec_chunks
                    else:
                        kept_rec = sae(flat)[0].to(dtype).view_as(kept_orig)

                    # restore RNG for identical masks
                    torch.random.set_rng_state(cpu_state)
                    torch.cuda.set_rng_state(cuda_state, dev)

                    def inj(_m,_i,out):
                        out[:,1:,:] = kept_rec
                        return out

                    hk_inj = mae.encoder.blocks[cfg.layer].register_forward_hook(inj)
                    _, pred_sae, mask2, _ = mae(v_vols, mask_ratio=cfg.vis_mask_ratio)
                    hk_inj.remove()
                    assert torch.equal(mask, mask2), "Mask mismatch."

                    # token-level metrics
                    hook_token_mse = F.mse_loss(kept_rec, kept_orig).item()
                    num = ((kept_rec - kept_rec.mean(dim=1,keepdim=True)) *
                           (kept_orig - kept_orig.mean(dim=1,keepdim=True))).sum(dim=(1,2))
                    den = (kept_rec.var(dim=1, unbiased=False).sum(dim=1).sqrt() *
                           kept_orig.var(dim=1, unbiased=False).sum(dim=1).sqrt() + 1e-12)
                    kept_token_corr = (num/den).mean().item()

                    # Pixel-level diff
                    if cfg.vis_offload_cpu:
                        pred_mae_cpu = pred_mae.cpu()
                        pred_sae_cpu = pred_sae.cpu()
                        v_cpu        = v_vols.float().cpu()
                        pm = mae.unpatchify(pred_mae_cpu).float()
                        ps = mae.unpatchify(pred_sae_cpu).float()
                    else:
                        pm = mae.unpatchify(pred_mae).float().cpu()
                        ps = mae.unpatchify(pred_sae).float().cpu()
                        v_cpu = v_vols.float().cpu()

                    pix_mse = F.mse_loss(ps, pm).item()
                    vis_img = make_slice_mosaic(v_cpu, pm, ps, step)

                    logv = {
                        "step": step,
                        "pixel_mse_diff": pix_mse,
                        "hook_token_mse": hook_token_mse,
                        "kept_token_corr": kept_token_corr,
                        "slice_grid": wandb.Image(str(vis_img)),
                    }

                    # ---- monosemantic metrics (if masks available) ----
                    if v_mask is not None and cfg.mono_eval:
                        patch_lab = patch_labels_from_seg(v_mask, mae, num_classes=cfg.num_classes)  # (B*L)
                        # run SAE on ALL tokens (not only kept) for mono eval (mask_ratio=0)
                        acts_all=[]
                        hk_all = mae.encoder.blocks[cfg.layer].register_forward_hook(lambda m,i,o: acts_all.append(o.detach()))
                        mae.forward_encoder(v_vols, mask_ratio=0.0); hk_all.remove()
                        toks_all = acts_all[0][:,1:,:].reshape(-1, C).float()  # (B*L,C)
                        rec_all, z_all = sae(toks_all)
                        agg, rows, purity, iouv = monosemantic_metrics(z_all, patch_lab, cfg.num_classes, cfg.mono_thr)
                        logv.update(agg)
                        # histograms (optional)
                        if cfg.log_hists_every and step % cfg.log_hists_every == 0:
                            logv["purity_hist"] = wandb.Histogram(purity.numpy())
                            logv["iou_hist"]    = wandb.Histogram(iouv.numpy())
                        # table
                        table = wandb.Table(columns=list(rows[0].keys()))
                        for r in rows: table.add_data(*r.values())
                        logv["top_neurons"] = table

                        del acts_all, toks_all, rec_all, z_all, patch_lab, purity, iouv, rows, agg, table

                    wandb.log(logv)

                    # cleanup
                    try: del pred_mae_cpu
                    except NameError: pass
                    try: del pred_sae_cpu
                    except NameError: pass
                    del (pred_mae, pred_sae, pm, ps, v_vols, v_cpu,
                         kept_tokens, kept_orig, kept_rec, mask, mask2, flat, vis_img, v_mask)
                    torch.cuda.empty_cache(); gc.collect()

                sae.train()

            if step >= cfg.steps: break

        del pf, loader, ds
        torch.cuda.empty_cache(); gc.collect()

    # save SAE
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc_w":  sae.enc_w.cpu(),
        "enc_b":  sae.enc_b.cpu(),
        "dec_w":  sae.dec_w.cpu(),
        "dec_b":  sae.dec_b.cpu(),
        "mu":     sae.mu.cpu(),
        "sigma":  sae.sigma.cpu(),
        "input_dim": C,
        "layer":     cfg.layer,
        "l1":        cfg.l1,
    }, cfg.out)
    wandb.finish()
    print("Saved SAE to", cfg.out)

# ------------------------- CLI -------------------------
def cli():
    P = argparse.ArgumentParser("Sparse Whitened SAE trainer (monosemantic metrics)")
    # data
    P.add_argument("--shard_dir", required=True, help="dir with shard_*.tar volumes")
    P.add_argument("--mask_dir",  default=None,  help="dir with shard_*.tar masks (optional)")
    P.add_argument("--vols_per_shard", type=int, default=16_384)
    # model / mae
    P.add_argument("--checkpoint", required=True)
    P.add_argument("--arch", choices=list(ARCHS.keys()), default="base_patch_conv")
    P.add_argument("--img_size", type=int, default=96)
    P.add_argument("--patch_size", type=int, default=8)
    P.add_argument("--layer", type=int, default=2)
    # train
    P.add_argument("--batch_size", type=int, default=128)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--steps", type=int, default=10_000)
    P.add_argument("--workers", type=int, default=8)
    P.add_argument("--timeout", type=int, default=300)
    P.add_argument("--val_split", type=float, default=0.02)
    P.add_argument("--epoch_shuffle", action="store_true")
    # sparsity
    P.add_argument("--l1", type=float, default=1e-3)
    # viz
    P.add_argument("--vis_mask_ratio", type=float, default=0.85)
    P.add_argument("--vis_int", type=int, default=1000)
    P.add_argument("--vis_n", type=int, default=3)
    P.add_argument("--vis_offload_cpu", action="store_true", default=True)
    P.add_argument("--vis_chunk_tokens", type=int, default=200_000,
                   help="Chunk size for SAE on kept tokens during viz")
    # mono metrics
    P.add_argument("--mono_eval", action="store_true", default=True,
                   help="Compute monosemantic metrics if masks exist")
    P.add_argument("--num_classes", type=int, default=4,
                   help="0=bg,1=mem,2=sphere,3=cube by default")
    P.add_argument("--mono_thr", type=float, default=0.05,
                   help="|z| threshold for binary activation when computing IoU/F1")
    # logging
    P.add_argument("--project", default="sae_sparse_mono")
    P.add_argument("--run", default="sparse_layer2")
    P.add_argument("--out", default="checkpoints/sae_sparse_whitened.pt")
    P.add_argument("--log_int", type=int, default=20)
    P.add_argument("--log_hists_every", type=int, default=0)
    return P.parse_args()

if __name__ == "__main__":
    train(cli())
