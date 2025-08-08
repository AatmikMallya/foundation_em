#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 SAE (Anthropic-parity version, 2025-07-31)
=================================================
• Unit-norm decoder dictionary with Adam-compatible gradient projection.
• Pre-encoder bias = − decoder bias via running µ (no separate dec_b).
• Optional top-k straight-through sparsity and L1 penalty.
• SAE masking == MAE masking (reuse MAE mask/ids).
• Rich monosemanticity metrics, 2-D overlays, compact 3-D GIFs.
• Throughput tweaks (single MAE fwd, chunked SAE, TF32, compile, etc.).
"""

# ─────────────────────────────────────────────────────────────── import
import argparse, random, signal, tarfile, math, gc
from pathlib import Path
from collections import deque
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import wandb
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ─────────────────────────────────────────────────────── MAE builders
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

# ───────────────────────────────────────────────────────────── utilities
def load_model_checkpoint(path, model, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.","",1): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    md = ckpt.get("model_dtype", "")
    if md == "torch.bfloat16": model = model.to(dtype=torch.bfloat16)
    if md == "torch.float16":  model = model.to(dtype=torch.float16)
    return model.to(device)

def topk_st(z, k):
    if k <= 0 or k >= z.shape[1]:
        return z
    vals, idx = torch.topk(z, k, dim=1)
    z_hard = torch.zeros_like(z).scatter_(1, idx, vals)
    return z + (z_hard - z).detach()

@torch.no_grad()
def build_keep_restore(mask: torch.Tensor):
    ids_sorted = torch.argsort(mask, dim=1, stable=True)
    len_keep = (mask==0).sum(1)
    assert len_keep.unique().numel()==1
    len_keep = int(len_keep[0])
    ids_keep = ids_sorted[:, :len_keep]
    ids_restore = torch.argsort(ids_sorted, dim=1)
    return ids_keep, ids_restore, len_keep

# ───────────────────────────────────────────────────── dataset + loader
class TarShardDataset(IterableDataset):
    def __init__(self, vol_shards: List[Path], vs:int,
                 shuffle=False, vps:int=16_384,
                 mask_shards: Optional[List[Path]]=None):
        self.vols=list(vol_shards)
        self.masks=list(mask_shards) if mask_shards else None
        self.vs=vs; self.shuf=shuffle; self.vps=vps
        print(f"Dataset: {len(self.vols)} shards × {vps} vols"
              + (" + masks" if self.masks else ""))

    def __len__(self): return len(self.vols)*self.vps

    def _iter_pair(self, v_path:Path, m_path:Optional[Path]):
        vol_tf  = tarfile.open(v_path,  "r|", bufsize=32*1024*1024)
        mask_tf = tarfile.open(m_path, "r|", bufsize=32*1024*1024) if m_path else None

        vit = (m for m in vol_tf  if m.isfile())
        mit = (m for m in mask_tf if m.isfile()) if mask_tf else None
        for mv in vit:
            vol = np.frombuffer(vol_tf.extractfile(mv).read(), np.float32)\
                     .reshape(self.vs,self.vs,self.vs)
            vol_t = torch.from_numpy(vol.copy()).unsqueeze(0).pin_memory()
            if mit:
                mm = next(mit)
                seg = np.frombuffer(mask_tf.extractfile(mm).read(), np.uint8)\
                         .reshape(self.vs,self.vs,self.vs)
                seg_t = torch.from_numpy(seg.copy()).pin_memory()
            else:
                seg_t = torch.zeros((self.vs,self.vs,self.vs), dtype=torch.uint8).pin_memory()
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
    def __init__(self, loader:DataLoader, dev:torch.device):
        self.it = iter(loader); self.dev=dev
        self.stream=torch.cuda.Stream(device=dev,priority=-1)
        self.qb,self.qe=[],[]
        for _ in range(2): self._prefetch()

    def _prefetch(self):
        try: vol, seg = next(self.it)
        except StopIteration: return
        with torch.cuda.stream(self.stream):
            vol = vol.pin_memory() if not vol.is_pinned() else vol
            vol_g = vol.to(self.dev, memory_format=torch.channels_last_3d,
                           non_blocking=True)
            if seg is not None:
                seg = seg.pin_memory() if not seg.is_pinned() else seg
                seg_g = seg.to(self.dev, non_blocking=True)
            else: seg_g=None
            ev=torch.cuda.Event(); ev.record(self.stream)
            self.qb.append((vol_g, seg_g)); self.qe.append(ev)

    def __iter__(self): return self
    def __next__(self):
        if not self.qb: raise StopIteration
        b=self.qb.pop(0); self.qe.pop(0).wait()
        self._prefetch()
        return b

# ───────────────────────────────────────────────── positional projector
@torch.no_grad()
def build_pos_projector_from_pe(mae, pos_var:float, pos_rank:int):
    pe = mae.encoder.pos_embed[:,1:,:].squeeze(0).float()  # (L,C)
    _, S, Vh = torch.linalg.svd(pe, full_matrices=False)
    if pos_rank>0:
        r = min(pos_rank, Vh.shape[0])
    else:
        expl = (S**2).cumsum(0)/(S**2).sum()
        r = max(int((expl<=pos_var).sum().item()), 1)
    return Vh[:r].contiguous()  # (r,C)

@torch.no_grad()
def build_pos_projector_from_samples(tokens_sample, coords, pos_rank:int, pos_var:float):
    """
    tokens_sample: (N,C)
    coords      : (T,3) grid coords in [0,1]
    """
    X = tokens_sample.float()
    Y = coords.repeat(X.shape[0]//coords.shape[0]+1, 1)[:X.shape[0]].to(X.device)
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    W = torch.linalg.lstsq(Xc, Yc).solution            # C×3
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    if pos_rank>0:
        r = min(pos_rank, U.shape[1])
    else:
        expl = (S**2).cumsum(0)/(S**2).sum()
        r = max(int((expl<=pos_var).sum().item()), 1)
    return U[:, :r].T.contiguous()  # (r,C)

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

# ───────────────────────────────────────────────── graph consistency
def graph_consistency_loss(p_grid):
    loss = 0.0; cnt = 0
    if p_grid.size(1) > 1:
        diff = p_grid[:,1:,:,:,:] - p_grid[:,:-1,:,:,:]
        loss += diff.pow(2).mean(); cnt += 1
    if p_grid.size(2) > 1:
        diff = p_grid[:,:,1:,:,:] - p_grid[:,:,:-1,:,:]
        loss += diff.pow(2).mean(); cnt += 1
    if p_grid.size(3) > 1:
        diff = p_grid[:,:,:,1:,:] - p_grid[:,:,:,:-1,:]
        loss += diff.pow(2).mean(); cnt += 1
    return loss / max(cnt,1)

# ────────────────────────────────────────────────────── SAE module
class ConvTopKSAE(torch.nn.Module):
    """
    One-hidden-layer over-complete SAE with:
      • running µ/σ for whitening
      • ReLU + optional top-k ST sparsity
      • unit-norm decoder dictionary (+ tangent-space grad projection)
    """
    def __init__(self, in_dim, hidden_dim, k=0, relu=True, momentum=0.01):
        super().__init__()
        self.k=k; self.relu=relu; self.mom=momentum
        self.register_buffer("mu",    torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))

        self.enc_w = torch.nn.Parameter(torch.empty(hidden_dim, in_dim))
        self.enc_b = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.dec_w = torch.nn.Parameter(torch.empty(in_dim, hidden_dim))

        torch.nn.init.kaiming_uniform_(self.enc_w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.dec_w, a=math.sqrt(5))
        self._renorm_dec_w()                                              # unit norm

        # project gradient component parallel to each dec column
        def proj_grad(g):
            w=self.dec_w.data
            col_dot=(g*w).sum(0,keepdim=True)     # 1×H
            return g - w*col_dot
        self.dec_w.register_hook(proj_grad)

    @torch.no_grad()
    def _renorm_dec_w(self):
        self.dec_w.div_(self.dec_w.norm(dim=0, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def update_stats(self, mu_b, std_b):
        self.mu    = (1-self.mom)*self.mu    + self.mom*mu_b
        self.sigma = (1-self.mom)*self.sigma + self.mom*std_b

    def forward(self, x_fp32):
        x_hat = (x_fp32 - self.mu) / self.sigma
        z_lin = F.linear(x_hat, self.enc_w, self.enc_b)
        z_relu= F.relu(z_lin, inplace=False) if self.relu else z_lin
        z_topk= topk_st(z_relu, self.k) if self.k>0 else z_relu
        rec_hat= F.linear(z_topk, self.dec_w)            # no bias
        x_rec = rec_hat * self.sigma + self.mu           # add µ back
        return x_rec, z_topk, z_relu

# ────────────────────────────────────────────────────── mono metrics
# … identical functions: patch_labels_from_seg, get_patch_xyz,
#                         auc_trap, binary_mi, monosemantic_metrics_full …
# (unchanged – paste original definitions here)

# ────────────────────────────────────────────────────── viz helpers
# … identical helpers: make_slice_mosaic, panel_frame,
#   make_3d_gif_triplet, color_patch_overlay_batch,
#   build_owner_volume, to_wandb_video …
# (unchanged – paste original definitions here)

# ────────────────────────────────────────────────────── MAE helpers
def build_mae(arch,img,p,dev):
    return ARCHS[arch](volume_size=(img,)*3,
                       patch_size=(p,)*3,
                       in_chans=1, mask_ratio=0.0).to(dev)

@torch.no_grad()
def get_tokens_layer(mae, vols, hook_layer:int):
    x = mae.encoder.patch_embed(vols)                 # B,L,C
    pe = mae.encoder.pos_embed[:,1:,:]
    if hook_layer == -1:  return x
    x = x + pe
    if hook_layer == -2:  return x
    cls_tok = mae.encoder.cls_token + mae.encoder.pos_embed[:,:1,:]
    x_full  = torch.cat([cls_tok.expand(vols.size(0),-1,-1), x], dim=1)
    for i, blk in enumerate(mae.encoder.blocks):
        x_full = blk(x_full)
        if i == hook_layer:
            return x_full[:,1:,:]
    x_full = mae.encoder.norm(x_full)
    return x_full[:,1:,:]

# ────────────────────────────────────────────────────── training loop
def train(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark=True

    # data split
    vol_shards=sorted(Path(cfg.shard_dir).glob("shard_*.tar"))
    mask_shards=sorted(Path(cfg.mask_dir).glob("shard_*.tar")) if cfg.mask_dir else None
    val_n=max(1,int(len(vol_shards)*cfg.val_split))
    tr_vol, val_vol = vol_shards[val_n:], vol_shards[:val_n]
    tr_mask = mask_shards[val_n:] if mask_shards else None
    val_mask= mask_shards[:val_n] if mask_shards else None

    # MAE
    mae=load_model_checkpoint(cfg.checkpoint,
                              build_mae(cfg.arch,cfg.img_size,cfg.patch_size,dev))
    mae.eval().requires_grad_(False)
    mae_dtype=next(mae.parameters()).dtype

    dummy=torch.zeros(1,1,cfg.img_size,cfg.img_size,cfg.img_size,
                      device=dev,dtype=mae_dtype)
    C=get_tokens_layer(mae,dummy,cfg.hook_layer).shape[-1]; dummy=None

    # positional projector
    patch_grid = (cfg.img_size//cfg.patch_size,)*3
    coords = (torch.stack(torch.meshgrid(
                torch.linspace(0,1,patch_grid[0]),
                torch.linspace(0,1,patch_grid[1]),
                torch.linspace(0,1,patch_grid[2]),
                indexing="ij"),-1).reshape(-1,3).to(dev)
              if (cfg.pos_metrics or cfg.project_pos) else None)

    U_pos=None
    if cfg.project_pos:
        if cfg.hook_layer<0:
            U_pos=build_pos_projector_from_pe(mae,cfg.pos_var,cfg.pos_rank).to(dev)
        else:
            sample_loader=DataLoader(
                TarShardDataset(val_vol,cfg.img_size,
                                vps=min(cfg.vols_per_shard,1024)),
                batch_size=cfg.pos_proj_batch,num_workers=0)
            s_vols,_=next(iter(sample_loader))
            s_vols=s_vols.to(dev,dtype=mae_dtype)
            toks_s=get_tokens_layer(mae,s_vols,cfg.hook_layer).float().reshape(-1,C)
            U_pos=build_pos_projector_from_samples(toks_s, coords,
                                                   cfg.pos_rank,cfg.pos_var).to(dev)
        print(f"[POS] rank={U_pos.shape[0]} projected.")

    # SAE
    H=int(cfg.latent_mul*C)
    sae=ConvTopKSAE(C,H,k=cfg.k_sparse,relu=not cfg.no_relu,
                    momentum=cfg.whiten_mom).to(dev)
    if cfg.compile_sae: sae=torch.compile(sae)

    opt=torch.optim.AdamW(sae.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.steps,
                                                     eta_min=cfg.lr*0.1)

    wandb.init(project=cfg.project,name=cfg.run,config=vars(cfg))
    wandb.define_metric("step"); wandb.define_metric("*",step_metric="step")

    hist=deque(maxlen=100); step=0; stop={"s":False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("s",True))

    best_val=float("inf"); best_path=Path(cfg.out).with_suffix(".best.pt")
    Pd,Ph,Pw = patch_grid

    # loader & prefetcher
    ds=TarShardDataset(tr_vol,cfg.img_size,shuffle=False,
                       vps=cfg.vols_per_shard,mask_shards=tr_mask)
    loader=DataLoader(ds,batch_size=cfg.batch_size,num_workers=cfg.workers,
                      pin_memory=True,pin_memory_device='cuda',
                      drop_last=True,persistent_workers=True,
                      multiprocessing_context="spawn",
                      timeout=cfg.timeout,prefetch_factor=2)
    pf=CUDAPrefetcher(loader,dev)

    # ───────────────────────────── main loop
    while step<cfg.steps and not stop["s"]:
        for vols, seg in pf:
            if stop["s"] or step>=cfg.steps: break
            vols=vols.to(dev,dtype=mae_dtype,non_blocking=True)

            # MAE forward to get mask / ids
            with torch.inference_mode():
                out=mae(vols,mask_ratio=cfg.train_mask_ratio)
                _, pred_mae_tr, mask_mae_tr, ids_restore_tr = (
                    out if len(out)==4 else (*out, None))
                if ids_restore_tr is None:
                    ids_keep_tr, ids_restore_tr,_=build_keep_restore(mask_mae_tr)
                else:
                    ids_keep_tr,_,_=build_keep_restore(mask_mae_tr)

            # token extraction & split
            toks=get_tokens_layer(mae,vols,cfg.hook_layer).float()
            toks_flat=toks.reshape(-1,C)
            toks_sem,toks_pos=split_pos_subspace(toks_flat,U_pos)
            sae.update_stats(toks_sem.mean(0),toks_sem.std(0).clamp_min(1e-6))

            # SAE forward
            opt.zero_grad(set_to_none=True)
            rec_sem_flat,z_topk,_=sae(toks_sem)
            mse_proj=F.mse_loss(rec_sem_flat,toks_sem)
            l1=z_topk.abs().mean() if cfg.l1>0 else toks_sem.new_tensor(0.)

            # optional graph-consistency loss
            if cfg.gc_lambda>0:
                B=toks.shape[0]; Hhid=z_topk.shape[1]
                probs=F.softmax(z_topk/cfg.gc_tau,dim=1)
                probs_grid=probs.view(B,Pd,Ph,Pw,Hhid)
                gc_loss=graph_consistency_loss(probs_grid)
            else:
                gc_loss=toks_sem.new_tensor(0.)

            loss=mse_proj+cfg.l1*l1+cfg.gc_lambda*gc_loss
            loss.backward()
            gnorm=torch.nn.utils.clip_grad_norm_(sae.parameters(),
                                                 cfg.grad_clip).item()
            opt.step(); sae._renorm_dec_w(); sched.step()
            step+=1; hist.append(float(loss))

            # ───────── logging
            if step%cfg.log_int==0:
                var=toks_sem.var().item()
                r2=1-mse_proj.item()/(var+1e-12)
                dead=(sae.enc_w.norm(dim=1)<1e-5).float().mean().item()
                k_eff=(z_topk.abs()>0).sum(1).float().mean().item()
                frac_act=k_eff/z_topk.shape[1]
                dens=(z_topk.abs()>0).float().mean(0).cpu().numpy()

                wandb.log({
                    "step":step,
                    "train_loss":sum(hist)/len(hist),
                    "train_mse":mse_proj.item(),
                    "train_l1":l1.item(),
                    "graph_cons_loss":gc_loss.item(),
                    "lr":sched.get_last_lr()[0],
                    "token_R2":r2,
                    "dead_frac":dead,
                    "frac_active":frac_act,
                    "k_eff":k_eff,
                    "feature_density_hist":wandb.Histogram(dens),
                    "sae_grad_norm":gnorm,
                })

            # ───────── validation / viz / mono-metrics
            if step%cfg.vis_int==0 or step==1:
                sae.eval(); torch.cuda.empty_cache()
                try:
                    with torch.inference_mode():
                        v_loader=DataLoader(
                            TarShardDataset(val_vol,cfg.img_size,
                                            vps=cfg.vols_per_shard,
                                            mask_shards=val_mask),
                            batch_size=cfg.vis_n,num_workers=0)
                        v_vols,v_mask=next(iter(v_loader))
                        v_vols=v_vols.to(dev,dtype=mae_dtype)
                        v_mask=v_mask.to(dev) if v_mask is not None else None

                        out=mae(v_vols,mask_ratio=cfg.vis_mask_ratio)
                        _, pred_mae, mask_mae, ids_restore_mae = (
                            out if len(out)==4 else (*out, None))
                        if ids_restore_mae is None:
                            ids_keep_v, ids_restore_mae,_=build_keep_restore(mask_mae)
                        else:
                            ids_keep_v,_,_=build_keep_restore(mask_mae)

                        toks_full=get_tokens_layer(mae,v_vols,cfg.hook_layer).float()
                        toks_full_flat=toks_full.reshape(-1,C)
                        toks_sem_flat,toks_pos_flat=split_pos_subspace(toks_full_flat,U_pos)
                        toks_sem=toks_sem_flat.view_as(toks_full)

                        kept_v=torch.gather(toks_sem,1,
                                            ids_keep_v.unsqueeze(-1).expand(-1,-1,C))
                        Bv,Kv,_=kept_v.shape
                        flat_kept=kept_v.reshape(-1,C)

                        # chunked recon
                        rec_kept_sem=[]; z_kept=[]
                        for s in range(0,flat_kept.shape[0],cfg.vis_chunk_tokens or flat_kept.shape[0]):
                            r,z,_=sae(flat_kept[s:s+cfg.vis_chunk_tokens])
                            rec_kept_sem.append(r); z_kept.append(z)
                        rec_kept_sem=torch.cat(rec_kept_sem,0).view(Bv,Kv,C)
                        z_kept=torch.cat(z_kept,0).view(Bv,Kv,-1)

                        # metrics in hook space
                        hook_mse=F.mse_loss(rec_kept_sem,kept_v).item()
                        num=((rec_kept_sem-rec_kept_sem.mean(1,True))*
                             (kept_v-kept_v.mean(1,True))).sum((1,2))
                        den=(rec_kept_sem.var(1,False).sum(1).sqrt()*
                             kept_v.var(1,False).sum(1).sqrt()+1e-12)
                        kept_corr=(num/den).mean().item()

                        # decoder path to pixels
                        pos_full=toks_pos_flat.view_as(toks_full)
                        pos_keep=torch.gather(pos_full,1,
                                              ids_keep_v.unsqueeze(-1).expand(-1,-1,C))
                        rec_kept_tokens=(rec_kept_sem+pos_keep).to(mae_dtype)

                        pos_all=mae.encoder.pos_embed[:,1:,:].expand(Bv,-1,-1)
                        pos_keep_pe=torch.gather(pos_all,1,
                                                 ids_keep_v.unsqueeze(-1).expand(-1,-1,C))

                        cls_tok=mae.encoder.cls_token+mae.encoder.pos_embed[:,:1,:]
                        x_enc=torch.cat([cls_tok.expand(Bv,-1,-1),
                                         rec_kept_tokens+pos_keep_pe.to(mae_dtype)],dim=1)
                        for blk in mae.encoder.blocks: x_enc=blk(x_enc)
                        x_enc=mae.encoder.norm(x_enc)

                        dec_in=mae.decoder_embed(x_enc)
                        dec_no_cls=dec_in[:,1:,:]
                        mask_tokens_dec=mae.mask_token.repeat(Bv,dec_no_cls.size(1)-Kv,1)
                        dec_cat=torch.cat([dec_no_cls,mask_tokens_dec],dim=1)
                        dec_cat=torch.gather(dec_cat,1,
                                             ids_restore_mae.unsqueeze(-1).expand(-1,-1,dec_in.size(2)))
                        dec_full=torch.cat([dec_in[:,:1,:],dec_cat],1)
                        dec_full=dec_full+mae.decoder_pos_embed
                        for blk in mae.decoder_blocks: dec_full=blk(dec_full)
                        dec_full=mae.decoder_norm(dec_full)
                        dec_full=dec_full[:,1:,:]
                        pred_sae=(mae.decoder_pred(dec_full,mae.patch_grid_shape)
                                  if getattr(mae,"decoder_neck","linear")=="conv"
                                  else mae.decoder_pred(dec_full))

                        # pixel metrics
                        gt_vol=v_vols.float().cpu()
                        pm=mae.unpatchify(pred_mae).float().cpu()
                        ps=mae.unpatchify(pred_sae).float().cpu()
                        pix_mse_sae_gt=F.mse_loss(ps,gt_vol).item()
                        pix_mse_mae_gt=F.mse_loss(pm,gt_vol).item()
                        pix_mse_mae_sae=F.mse_loss(ps,pm).item()

                        vis_path=make_slice_mosaic(gt_vol,pm,ps,step)

                        logv={
                            "step":step,
                            "pixel_mse_diff":pix_mse_mae_sae,
                            "pixel_mse_sae_gt":pix_mse_sae_gt,
                            "pixel_mse_mae_gt":pix_mse_mae_gt,
                            "hook_token_mse":hook_mse,
                            "kept_token_corr":kept_corr,
                            "k_eff_kept":(z_kept.abs()>0).sum(2).float().mean().item(),
                            "slice_grid":wandb.Image(str(vis_path)),
                        }

                        # monosemantic metrics
                        if v_mask is not None and cfg.mono_eval:
                            labels=patch_labels_from_seg(v_mask,cfg.patch_size,
                                                         cfg.num_classes)
                            _,z_all_topk,_=sae(toks_sem_flat)
                            coords_rep=(coords.repeat(v_vols.size(0),1)
                                        if coords is not None else None)
                            agg,rows,purity,iou,sel,ent,mis,aurocs,pos_r2,iff,_=\
                                monosemantic_metrics_full(z_all_topk.abs(),labels,
                                                          cfg.num_classes,cfg.mono_thr,
                                                          coords_rep)
                            logv.update(agg)
                            if cfg.log_hists_every and step%cfg.log_hists_every==0:
                                logv.update({
                                    "purity_hist":wandb.Histogram(purity.numpy()),
                                    "iou_hist":wandb.Histogram(iou.numpy()),
                                    "sel_hist":wandb.Histogram(sel.numpy()),
                                    "entropy_hist":wandb.Histogram(ent.numpy()),
                                    "mi_hist":wandb.Histogram(mis.numpy()),
                                    "auroc_hist":wandb.Histogram(aurocs.numpy()),
                                    "pos_r2_hist":wandb.Histogram(pos_r2.numpy()),
                                    "iff_hist":wandb.Histogram(iff.numpy()),
                                })
                            table=wandb.Table(columns=list(rows[0].keys()))
                            for r in rows: table.add_data(*r.values())
                            logv["top_neurons"]=table

                        wandb.log(logv)

                        # checkpoint best
                        if pix_mse_sae_gt<best_val:
                            best_val=pix_mse_sae_gt
                            torch.save({
                                "enc_w":sae.enc_w.cpu(),"enc_b":sae.enc_b.cpu(),
                                "dec_w":sae.dec_w.cpu(),
                                "mu":sae.mu.cpu(),"sigma":sae.sigma.cpu(),
                                "input_dim":C,"hidden_dim":H,"k":cfg.k_sparse,
                                "hook_point":cfg.hook_point,
                                "hook_layer":cfg.hook_layer,
                                "project_pos":cfg.project_pos,
                            },best_path)
                            wandb.run.summary.update({"best_pix_mse_sae_gt":best_val,
                                                      "best_ckpt_path":str(best_path)})
                finally:
                    sae.train()
                    torch.cuda.empty_cache()

            # housekeeping
            del vols, seg, toks, toks_flat, toks_sem, toks_pos, rec_sem_flat, z_topk
            torch.cuda.empty_cache()

        # epoch refresh
        del pf, loader, ds
        torch.cuda.empty_cache(); gc.collect()

    # save final
    Path(cfg.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "enc_w":sae.enc_w.cpu(),"enc_b":sae.enc_b.cpu(),
        "dec_w":sae.dec_w.cpu(),
        "mu":sae.mu.cpu(),"sigma":sae.sigma.cpu(),
        "input_dim":C,"hidden_dim":H,"k":cfg.k_sparse,
        "hook_point":cfg.hook_point,"hook_layer":cfg.hook_layer,
        "project_pos":cfg.project_pos,
    },cfg.out)
    wandb.finish()
    print("Saved SAE to",cfg.out,"Best:",
          best_path if best_val<1e9 else "N/A")

# ─────────────────────────────────────────────────────── CLI parser
def cli():
    P=argparse.ArgumentParser("Stage-1 SAE (Anthropic-parity)")
    # data
    P.add_argument("--shard_dir",required=True)
    P.add_argument("--mask_dir",default=None)
    P.add_argument("--vols_per_shard",type=int,default=16_384)
    # mae
    P.add_argument("--checkpoint",required=True)
    P.add_argument("--arch",choices=list(ARCHS.keys()),
                   default="base_patch_conv")
    P.add_argument("--img_size",type=int,default=96)
    P.add_argument("--patch_size",type=int,default=8)
    # hook
    P.add_argument("--hook_point",choices=["patch","postpos"],default="patch")
    P.add_argument("--hook_layer",type=int,default=-1)
    # sae
    P.add_argument("--latent_mul",type=float,default=2.0)
    P.add_argument("--k_sparse",type=int,default=0)
    P.add_argument("--no_relu",action="store_true")
    P.add_argument("--l1",type=float,default=0.0)
    P.add_argument("--wd",type=float,default=1e-4)
    P.add_argument("--whiten_mom",type=float,default=0.01)
    # pos
    P.add_argument("--project_pos",action="store_true",default=False)
    P.add_argument("--pos_var",type=float,default=0.99)
    P.add_argument("--pos_rank",type=int,default=3)
    P.add_argument("--pos_metrics",action="store_true",default=True)
    P.add_argument("--pos_proj_batch",type=int,default=4)
    # graph consistency
    P.add_argument("--gc_lambda",type=float,default=0.0)
    P.add_argument("--gc_tau",type=float,default=0.2)
    # train
    P.add_argument("--compile_sae",action="store_true",default=True)
    P.add_argument("--batch_size",type=int,default=512)
    P.add_argument("--lr",type=float,default=1e-3)
    P.add_argument("--steps",type=int,default=25_000)
    P.add_argument("--workers",type=int,default=8)
    P.add_argument("--timeout",type=int,default=300)
    P.add_argument("--val_split",type=float,default=0.02)
    P.add_argument("--grad_clip",type=float,default=1e9)
    P.add_argument("--train_mask_ratio",type=float,default=0.85)
    # viz
    P.add_argument("--vis_mask_ratio",type=float,default=0.85)
    P.add_argument("--vis_int",type=int,default=500)
    P.add_argument("--vis_n",type=int,default=3)
    P.add_argument("--vis_chunk_tokens",type=int,default=200_000)
    P.add_argument("--color_neurons",action="store_true",default=True)
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
    P.add_argument("--mono_thr",type=float,default=0.05)
    P.add_argument("--log_hists_every",type=int,default=0)
    # logging/out
    P.add_argument("--project",default="sae_stage1")
    P.add_argument("--run",default="stage1_anthropic")
    P.add_argument("--out",default="checkpoints/sae_stage1.pt")
    P.add_argument("--log_int",type=int,default=20)
    return P.parse_args()

if __name__=="__main__":
    train(cli())
