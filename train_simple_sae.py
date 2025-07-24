#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whitened Identity‑SAE trainer with extensive WandB diagnostics.
"""

# ───────────── stdlib
import argparse, random, signal, tarfile, math
from pathlib import Path
from collections import deque

# ───────────── 3rd‑party
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import wandb, matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ───────────── MAE builders
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

# ═════════════════════ Checkpoint loader ════════════════════════════
def load_model_checkpoint(path, model, device="cuda"):
    ckpt=torch.load(path,map_location=device)
    sd=ckpt.get("model_state_dict",ckpt)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd={k.replace("_orig_mod.","",1):v for k,v in sd.items()}
    model.load_state_dict(sd,strict=False)
    if ckpt.get("model_dtype")=="torch.bfloat16": model=model.to(dtype=torch.bfloat16)
    if ckpt.get("model_dtype")=="torch.float16":  model=model.to(dtype=torch.float16)
    return model.to(device)

# ═════════════════════ Dataset / Prefetcher ═════════════════════════
class TarShardDataset(IterableDataset):
    def __init__(self,shards,vs,shuffle=False,vps=16_384):
        self.s=list(shards); self.vs=vs; self.sh=shuffle; self.vps=vps
        print(f"Dataset: {len(shards)} shards × {vps} vols")
    def __len__(self): return len(self.s)*self.vps
    def _iter_shard(self,p):
        with tarfile.open(p,"r|",bufsize=32*1024*1024) as tf:
            for m in tf:
                if m.isfile():
                    vol=np.frombuffer(tf.extractfile(m).read(),np.float32).reshape(self.vs,self.vs,self.vs)
                    yield torch.from_numpy(vol).unsqueeze(0).pin_memory()
    def __iter__(self):
        w=torch.utils.data.get_worker_info()
        lst=self.s.copy(); random.shuffle(lst) if self.sh else None
        if w: lst=lst[w.id::w.num_workers]
        for p in lst: yield from self._iter_shard(p)

class CUDAPrefetcher:
    def __init__(self,loader,dev):
        self.it=iter(loader); self.dev=dev
        self.stream=torch.cuda.Stream(device=dev,priority=-1)
        self.qb,self.qe=[],[]
        for _ in range(2): self._prefetch()
    def _prefetch(self):
        try: bc=next(self.it)
        except StopIteration: return
        with torch.cuda.stream(self.stream):
            bc=bc.pin_memory() if not bc.is_pinned() else bc
            bg=bc.to(self.dev,memory_format=torch.channels_last_3d,non_blocking=True)
            ev=torch.cuda.Event(); ev.record(self.stream)
            self.qb.append(bg); self.qe.append(ev)
    def __iter__(self): return self
    def __next__(self):
        if not self.qb: raise StopIteration
        b=self.qb.pop(0); self.qe.pop(0).wait(); self._prefetch(); return b

# ═════════════════════ Whitened SAE ═════════════════════════════════
class WhitenedSAE(torch.nn.Module):
    def __init__(self,dim,momentum=0.01):
        super().__init__()
        self.momentum=momentum
        self.register_buffer("mu",torch.zeros(dim))
        self.register_buffer("sigma",torch.ones(dim))
        self.enc_w=torch.nn.Parameter(torch.empty(dim,dim))
        self.enc_b=torch.nn.Parameter(torch.zeros(dim))
        self.dec_w=torch.nn.Parameter(torch.empty(dim,dim))
        self.dec_b=torch.nn.Parameter(torch.zeros(dim))
        torch.nn.init.kaiming_uniform_(self.enc_w,a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.dec_w,a=math.sqrt(5))
    @torch.no_grad()
    def update_stats(self,b_mu,b_std):
        self.mu   = (1-self.momentum)*self.mu   + self.momentum*b_mu
        self.sigma= (1-self.momentum)*self.sigma+ self.momentum*b_std
    def forward(self,x):                       # x fp32
        x_hat=(x-self.mu)/self.sigma
        z=F.linear(x_hat,self.enc_w,self.enc_b)
        rec_w=F.linear(z,self.dec_w,self.dec_b)
        return rec_w*self.sigma + self.mu

# ═════════════════════ Visual helper ═══════════════════════════════
@torch.no_grad()
def make_slice_mosaic(vols,m0,m1,step,outdir="vis"):
    Path(outdir).mkdir(exist_ok=True)
    B,_,D,H,W=vols.shape; mids=(D//2,H//2,W//2)
    fig,ax=plt.subplots(3*B,3,figsize=(7,2.1*B),
                        gridspec_kw={"wspace":.01,"hspace":.01})
    for b in range(B):
        for r,(mid,label) in enumerate(zip(mids,["Z","Y","X"])):
            row=3*b+r
            for c,img in enumerate([vols[b,0],m0[b,0],m1[b,0]]):
                sl=[slice(None)]*3; sl[r]=mid
                ax[row,c].imshow(img[tuple(sl)],cmap="gray"); ax[row,c].axis("off")
                if r==0: ax[row,c].set_title(["GT","MAE","SAE"][c],fontsize=8)
            ax[row,0].set_ylabel(label,fontsize=8,rotation=0,labelpad=14)
    ax[0,0].figure.suptitle(f"Step {step}",fontsize=10)
    fn=Path(outdir)/f"vis_{step}.png"; fig.savefig(fn,dpi=120,bbox_inches="tight"); plt.close(fig)
    return fn

# ═════════════════════ Training loop ════════════════════════════════
def build_mae(arch,img,p,dev):
    return ARCHS[arch](volume_size=(img,)*3,patch_size=(p,)*3,in_chans=1,mask_ratio=0.0).to(dev)

def train(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards=sorted(Path(cfg.shard_dir).glob("shard*.tar")); assert shards
    val_n=max(1,int(len(shards)*cfg.val_split)); tr,val=shards[val_n:],shards[:val_n]

    mae=load_model_checkpoint(cfg.checkpoint,build_mae(cfg.arch,cfg.img_size,cfg.patch_size,dev))
    mae.eval().requires_grad_(False); dtype=next(mae.parameters()).dtype

    dummy=torch.zeros(1,1,cfg.img_size,cfg.img_size,cfg.img_size,device=dev,dtype=dtype)
    acts=[]; hk=mae.encoder.blocks[cfg.layer].register_forward_hook(lambda m,i,o:acts.append(o))
    mae.forward_encoder(dummy,mask_ratio=0.0); hk.remove(); C=acts[0][:,1:].shape[-1]

    sae=WhitenedSAE(C).to(dev,dtype=torch.float32)
    opt=torch.optim.AdamW(sae.parameters(),lr=cfg.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.steps,eta_min=cfg.lr*0.1)

    wandb.init(project=cfg.project,name=cfg.run,config=vars(cfg))
    wandb.define_metric("step"); wandb.define_metric("*",step_metric="step")

    hist=deque(maxlen=100); step=0; stop={"s":False}
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__("s",True))

    while step<cfg.steps and not stop["s"]:
        ds=TarShardDataset(tr,cfg.img_size,shuffle=(cfg.epoch_shuffle and step>0))
        loader=DataLoader(ds,batch_size=cfg.batch_size,num_workers=cfg.workers,
                          pin_memory=False,drop_last=True,persistent_workers=True,
                          multiprocessing_context="spawn",timeout=cfg.timeout)
        pf=CUDAPrefetcher(loader,dev)
        for vols in pf:
            if stop["s"] or step>=cfg.steps: break
            vols=vols.to(dev,dtype=dtype,non_blocking=True)

            acts=[]; hk=mae.encoder.blocks[cfg.layer].register_forward_hook(lambda m,i,o:acts.append(o.detach()))
            mae.forward_encoder(vols,mask_ratio=0.0); hk.remove()
            toks=acts[0][:,1:].reshape(-1,C).float()

            sae.update_stats(toks.mean(0),toks.std(0).clamp_min(1e-6))

            opt.zero_grad(set_to_none=True)
            rec=sae(toks); loss=F.mse_loss(rec,toks)
            loss.backward()
            g_norm=torch.nn.utils.clip_grad_norm_(sae.parameters(),1e9).item()
            # update/weight ratio (encoder)
            upd_ratio=sae.enc_w.grad.std().item()/sae.enc_w.std().item()
            opt.step(); sched.step()
            hist.append(float(loss)); step+=1

            if step%cfg.log_int==0:
                var=toks.var().item()
                dead_frac=(sae.enc_w.norm(dim=1)<1e-4).float().mean().item()
                wandb.log({
                    "step":step,
                    "train_loss":sum(hist)/len(hist),
                    "lr":sched.get_last_lr()[0],
                    "token_mean":toks.mean().item(),
                    "token_std":toks.std().item(),
                    "token_R2":1.0-loss.item()/var,
                    "sigma_mean":sae.sigma.mean().item(),
                    "sigma_std":sae.sigma.std().item(),
                    "sae_grad_norm":g_norm,
                    "update_ratio":upd_ratio,
                    "dead_frac":dead_frac,
                })

            if step%cfg.vis_int==0:
                sae.eval()
                v=next(iter(DataLoader(TarShardDataset(val,cfg.img_size),batch_size=cfg.vis_n,num_workers=0))).to(dev,dtype=dtype)
                _,pred_mae,*_=mae(v,mask_ratio=cfg.vis_mask_ratio)

                acts=[]; hk=mae.encoder.blocks[cfg.layer].register_forward_hook(lambda m,i,o:acts.append(o.detach()))
                mae.forward_encoder(v,mask_ratio=0.0); hk.remove()
                p=acts[0][:,1:].reshape(-1,C).float(); p_rec=sae(p).to(dtype)

                def inj(_,__,out):
                    B,Lp,Cdim=out.shape; needed=Lp-1
                    out_mod=out.clone()
                    out_mod[:,1:,:]=p_rec.reshape(B,-1,Cdim)[:,:needed]
                    return out_mod
                inj_hd=mae.encoder.blocks[cfg.layer].register_forward_hook(inj)
                _,pred_sae,*_=mae(v,mask_ratio=cfg.vis_mask_ratio); inj_hd.remove()

                pix_mse=F.mse_loss(mae.unpatchify(pred_sae).float(),mae.unpatchify(pred_mae).float()).item()
                enc_spec=torch.linalg.svdvals(sae.enc_w).max().item()
                dec_spec=torch.linalg.svdvals(sae.dec_w).max().item()

                img=make_slice_mosaic(v.float().cpu(),mae.unpatchify(pred_mae).float().cpu(),
                                      mae.unpatchify(pred_sae).float().cpu(),step)
                wandb.log({
                    "step":step,
                    "pixel_mse_diff":pix_mse,
                    "enc_spec":enc_spec,
                    "dec_spec":dec_spec,
                    "mu_hist":wandb.Histogram(sae.mu.cpu()),
                    "sigma_hist":wandb.Histogram(sae.sigma.cpu()),
                    "slice_grid":wandb.Image(str(img)),
                })
                sae.train()

            if step>=cfg.steps: break

    Path(cfg.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "enc_w":sae.enc_w.cpu(),"enc_b":sae.enc_b.cpu(),
        "dec_w":sae.dec_w.cpu(),"dec_b":sae.dec_b.cpu(),
        "mu":sae.mu.cpu(),"sigma":sae.sigma.cpu(),
        "input_dim":C,"layer":cfg.layer
    },cfg.out)
    wandb.finish(); print("Saved SAE to",cfg.out)

# ═════════════════════ CLI ══════════════════════════════════════════
def cli():
    P=argparse.ArgumentParser("Whitened Identity‑SAE trainer")
    P.add_argument("--shard_dir",required=True)
    P.add_argument("--checkpoint",required=True)
    P.add_argument("--arch",choices=list(ARCHS.keys()),default="base_patch_conv")
    P.add_argument("--img_size",type=int,default=96)
    P.add_argument("--patch_size",type=int,default=8)
    P.add_argument("--layer",type=int,default=2)
    P.add_argument("--batch_size",type=int,default=128)
    P.add_argument("--lr",type=float,default=1e-3)
    P.add_argument("--steps",type=int,default=10_000)
    P.add_argument("--workers",type=int,default=8)
    P.add_argument("--timeout",type=int,default=300)
    P.add_argument("--val_split",type=float,default=0.02)
    P.add_argument("--epoch_shuffle",action="store_true")
    P.add_argument("--vis_mask_ratio",type=float,default=0.85)
    P.add_argument("--vis_int",type=int,default=1000)
    P.add_argument("--vis_n",type=int,default=3)
    P.add_argument("--project",default="sae_whitened_stream")
    P.add_argument("--run",default="whitened_id_layer2")
    P.add_argument("--out",default="checkpoints/sae_whitened_identity.pt")
    P.add_argument("--log_int",type=int,default=20)
    return P.parse_args()

if __name__=="__main__":
    train(cli())
