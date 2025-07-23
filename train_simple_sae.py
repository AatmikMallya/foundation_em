#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identity‑SAE: minimal auto‑encoder that just copies ViT‑MAE‑3D activations
=========================================================================

  • Tied weights  :  decoder = encoderᵀ
  • Loss          :  MSE(recon, target)
  • No sparsity, no orthogonality, no dead‑latent logic
  • bf16 AMP by default (set --no_amp for fp32)
  • WandB logging + slice‑grid visualisation

Author: 2025‑07‑23  (written completely from scratch for Aatmik)
"""

# ───────────── stdlib
import argparse, math, signal, tarfile, time
from pathlib import Path
from collections import deque

# ───────────── 3rd‑party
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
plt.switch_backend("Agg")                # headless

# ═════════════════════════ Dataset ════════════════════════════════════
class TarShardDataset(Dataset):
    """Reads 96³ float32 volumes packed in *.tar shards (no labels)."""
    def __init__(self, shard_paths, img_size=96):
        self.members = []        # (shard_path, member_name)
        self.img_size = img_size
        for p in shard_paths:
            with tarfile.open(p) as tf:
                for m in tf.getmembers():
                    if m.size == img_size**3 * 4:      # float32 bytes
                        self.members.append((p, m.name))
        if not self.members:
            raise RuntimeError("No suitable volumes found!")
    def __len__(self):  return len(self.members)
    def __getitem__(self, idx):
        tar_path, member = self.members[idx]
        with tarfile.open(tar_path) as tf:
            buf = tf.extractfile(member).read()
        vol = np.frombuffer(buf, dtype=np.float32).reshape(
                    self.img_size, self.img_size, self.img_size)
        return torch.from_numpy(vol).unsqueeze(0)       # (1,D,H,W)

# ═══════════════════════  Identity SAE  ═══════════════════════════════
class IdentitySAE(torch.nn.Module):
    """Encoder  z = Wx + b         (linear)
       Decoder  x̂ = Wᵀ z + c      (tied)"""
    def __init__(self, dim):
        super().__init__()
        self.weight   = torch.nn.Parameter(torch.eye(dim))
        self.enc_bias = torch.nn.Parameter(torch.zeros(dim))
        self.dec_bias = torch.nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        z   = F.linear(x, self.weight,  self.enc_bias)
        x̂   = F.linear(z, self.weight.t(), self.dec_bias)
        return x̂

# ═══════════════════════  MAE helper  ════════════════════════════════
def load_mae(path, device):
    """
    Expects a full torch.nn.Module saved via torch.save(model, ...)
    If you only have a state‑dict, replace the body by your model‑builder
    and load_state_dict().
    """
    mae = torch.load(path, map_location=device)
    mae.eval().requires_grad_(False)
    # sanity‑check we have required helper functions
    for fn in ["patchify", "unpatchify"]:
        if not hasattr(mae, fn):
            raise AttributeError(f"MAE model lacks `{fn}()`")
    return mae

# ═════════════════════  Token extractor  ═════════════════════════════
class Hook:
    """Captures activations from an nn.Module forward hook."""
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.act    = None
    def hook_fn(self, _m, _in, out):
        self.act = out.detach()
    def close(self): self.handle.remove()

def get_token_dim(mae, layer_idx, device, img_size):
    dummy = torch.zeros(1,1,img_size,img_size,img_size, device=device)
    hook  = Hook(mae.encoder.blocks[layer_idx])
    mae.forward_encoder(dummy, mask_ratio=0.0)
    dim = hook.act[:,1:].shape[-1]     # drop CLS
    hook.close()
    return dim

# ═════════════════════ Visualiser (slice grid) ═══════════════════════
@torch.no_grad()
def slice_grid(vols, rec_mae, rec_sae, step, out_dir="vis"):
    """
    vols, rec_* : (B,1,D,H,W)
    Grid: GT | MAE | SAE  (middle Z,Y,X slices)
    """
    Path(out_dir).mkdir(exist_ok=True)
    B, _, D, H, W = vols.shape
    mids = (D//2, H//2, W//2)
    rows, cols = 3, 3*B
    fig, ax = plt.subplots(rows, cols,
                           figsize=(2.2*cols, 2*rows),
                           gridspec_kw={"wspace":0.01,"hspace":0.01})
    for b in range(B):
        ims = [vols[b,0], rec_mae[b,0], rec_sae[b,0]]
        for r,mid in enumerate(mids):
            for c,img in enumerate(ims):
                a = ax[r,3*b+c]
                sl = [slice(None)]*3; sl[r]=mid         # pick correct axis
                a.imshow(img[tuple(sl)], cmap="gray")
                if r==0: a.set_title(["GT","MAE","SAE"][c])
                a.axis("off")
    fig.suptitle(f"Step {step}")
    fn = Path(out_dir)/f"grid_{step}.png"
    fig.savefig(fn, dpi=120, bbox_inches="tight"); plt.close(fig)
    return fn

# ═══════════════════════  Training loop  ═════════════════════════════
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if cfg.amp and torch.cuda.is_available() else torch.float32

    shards = sorted(Path(cfg.shard_dir).glob("shard*.tar"))
    val_n  = max(1,int(len(shards)*cfg.val_split))
    tr_ds  = TarShardDataset(shards[val_n:], cfg.img_size)
    val_ds = TarShardDataset(shards[:val_n], cfg.img_size)
    tr_ld  = DataLoader(tr_ds, cfg.batch_size, shuffle=True,
                        num_workers=cfg.workers, pin_memory=False, drop_last=True)
    val_ld = DataLoader(val_ds, cfg.batch_size, shuffle=False,
                        num_workers=cfg.workers, pin_memory=False, drop_last=False)

    mae = load_mae(cfg.checkpoint, device)
    C   = get_token_dim(mae, cfg.layer, device, cfg.img_size)
    sae = IdentitySAE(C).to(device, dtype=amp_dtype)
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.steps, eta_min=cfg.lr*0.1)

    # capture tokens from encoder block
    tok_hook = Hook(mae.encoder.blocks[cfg.layer])

    wandb.init(project=cfg.project, name=cfg.run, config=vars(cfg))
    wandb.define_metric("step"); wandb.define_metric("*", step_metric="step")

    cancel = {"stop":False}
    signal.signal(signal.SIGTERM, lambda *_: cancel.__setitem__("stop",True))
    loss_hist = deque(maxlen=100)
    step = 0

    tr_iter = iter(tr_ld)
    while step < cfg.steps and not cancel["stop"]:
        try:
            vols = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_ld); vols = next(tr_iter)
        vols = vols.to(device, dtype=amp_dtype, non_blocking=True)

        # --- extract tokens ---------------------------------------
        tok_hook.act = None
        mae.forward_encoder(vols, mask_ratio=0.0)
        toks = tok_hook.act[:,1:].reshape(-1,C)         # drop CLS, flatten

        # --- SAE step --------------------------------------------
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=cfg.amp):
            recon = sae(toks)
            loss  = F.mse_loss(recon, toks)
        loss.backward(); opt.step(); sched.step()

        step += 1; loss_hist.append(loss.item())

        # --- logging ---------------------------------------------
        if step % cfg.log_int == 0:
            wandb.log({"step":step,
                       "train_loss": sum(loss_hist)/len(loss_hist),
                       "lr": sched.get_last_lr()[0],
                       "token_mean": toks.mean().item(),
                       "token_std":  toks.std().item()})

        # --- validation ------------------------------------------
        if step % cfg.val_int == 0:
            sae.eval(); vls=[]
            with torch.no_grad():
                for v in val_ld:
                    v = v.to(device, dtype=amp_dtype)
                    tok_hook.act=None; mae.forward_encoder(v, mask_ratio=0.0)
                    t = tok_hook.act[:,1:].reshape(-1,C)
                    vls.append( F.mse_loss(sae(t), t).item() )
                    if len(vls)==10: break
            wandb.log({"step":step,"val_loss": sum(vls)/len(vls)})
            sae.train()

        # --- visualise -------------------------------------------
        if step % cfg.vis_int == 0:
            sae.eval()
            v = next(iter(val_ld))[:cfg.vis_samples].to(device, dtype=amp_dtype)
            # GT volumes
            tok_hook.act=None; mae.forward_encoder(v, mask_ratio=0.0)
            t = tok_hook.act[:,1:].reshape(-1,C)
            rec_tok = sae(t).to(dtype=amp_dtype)
            # inject: replace encoder output then run full MAE decode
            def inj_fn(_m, _in, out):
                out = out.clone()
                out[:,1:,:] = rec_tok.view_as(out[:,1:,:])
                return out
            inj_handle = mae.encoder.blocks[cfg.layer].register_forward_hook(inj_fn)
            _, pred_sae, *_ = mae(v, mask_ratio=0.0)
            inj_handle.remove()
            _, pred_mae, *_ = mae(v, mask_ratio=0.0)
            img = slice_grid(v.float().cpu(),
                             mae.unpatchify(pred_mae).float().cpu(),
                             mae.unpatchify(pred_sae).float().cpu(),
                             step, out_dir="vis")
            wandb.log({"step":step, "slice_grid": wandb.Image(str(img))})
            sae.train()

    # save SAE
    Path(cfg.out).parent.mkdir(exist_ok=True)
    torch.save({"sae_weight": sae.weight.cpu(),
                "enc_bias": sae.enc_bias.cpu(),
                "dec_bias": sae.dec_bias.cpu(),
                "input_dim": C,
                "layer": cfg.layer}, cfg.out)
    wandb.finish()
    print("Finished. SAE saved to", cfg.out)

# ═══════════════════════════ CLI ══════════════════════════════════════
def parse():
    P = argparse.ArgumentParser("Identity‑SAE trainer from scratch")
    P.add_argument("--shard_dir", required=True)
    P.add_argument("--checkpoint", required=True, help="*.pt file saved with torch.save(model, …)")
    P.add_argument("--img_size", type=int, default=96)
    P.add_argument("--patch_size", type=int, default=8)          # only used by MAE if needed
    P.add_argument("--layer", type=int,  default=2)
    P.add_argument("--batch_size", type=int, default=128)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--steps", type=int, default=5000)
    P.add_argument("--workers", type=int, default=8)
    P.add_argument("--val_split", type=float, default=0.02)
    P.add_argument("--project", default="sae_identity")
    P.add_argument("--run", default="id_layer2")
    P.add_argument("--out", default="checkpoints/sae_identity.pt")
    P.add_argument("--vis_samples", type=int, default=6)
    P.add_argument("--amp", action="store_true", default=True)
    P.add_argument("--log_int",  type=int, default=20)
    P.add_argument("--val_int",  type=int, default=500)
    P.add_argument("--vis_int",  type=int, default=1000)
    return P.parse_args()

if __name__ == "__main__":
    cfg = parse(); train(cfg)
