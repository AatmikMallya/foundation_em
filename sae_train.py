#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse Auto-Encoder (SAE) training for ViT-MAE-3D activations
==============================================================
Trains a linear, L1-sparsity-penalised auto-encoder on the patch-token
activations of a *single* encoder block inside a pretrained MAE model.

Typical usage
-------------
python3 sae_train.py \
    --checkpoint checkpoints/best_model_mask_75.pt \
    --shard_dir /path/to/tar/shards \
    --layer 6               # 0-based index into model.encoder.blocks

The script re-uses TarShardDataset/CUDAPrefetcher from *vol_train.py* to
stream 3-D EM volumes from tar shards and computes activations with the
MAE frozen (mask_ratio = 0 so every patch is visible).  Each patch token
(i.e. each 8³ cube for the default 96³ volume) is treated as an
independent training sample for the SAE.

Loss = MSE(recon, target) + λ ⋅ |latent|₁.

The resulting *.pt* file stores encoder weight `W` (latent × input), the
L1 coefficient and dataclass-style hyper-parameters for downstream use
(e.g. concept attribution / nearest-neighbour visualisation).
"""

# ───────────────────────── stdlib
import argparse, math, random, time
from pathlib import Path

# ───────────────────────── 3rd-party
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ───────────────────────── project
from vol_train import TarShardDataset, CUDAPrefetcher  # re-use optimised loaders
from vit_3d import (
    mae_vit_3d_small_conv, mae_vit_3d_base_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv,
    get_device,
)

# ═════════════════════════ SAE module ═══════════════════════════════════
class LinearSAE(torch.nn.Module):
    """Linear weight-tied sparse auto-encoder.

    recon = z @ Wᵀ  with z = max(0, x @ W)
    where W ∈ ℝ^{latent_dim × input_dim}
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Following Anthropic, no bias, weight-tied decoder
        self.weight = torch.nn.Parameter(torch.empty(latent_dim, input_dim))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Encoder (ReLU for non-negativity)
        z = F.relu(F.linear(x, self.weight))  # (N, latent_dim)
        # Decoder (weight-tied)
        recon = F.linear(z, self.weight.t())  # (N, input_dim)
        return recon, z

# ═════════════════════════ helpers ═════════════════════════════════════
@torch.no_grad()
def extract_patch_tokens(model, volumes, layer_idx: int):
    """Run volumes through *model* and return patch-token activations at *layer_idx*.

    Returns tensor of shape (N_tokens, C).
    """
    captured = {}

    def _hook(_module, _inp, out):
        # out: (B, 1+L, C).  Drop CLS token.
        captured["act"] = out[:, 1:, :].detach()  # (B, L, C)

    handle = model.encoder.blocks[layer_idx].register_forward_hook(_hook)

    # Forward pass (mask_ratio=0 ⇒ no masking ⇒ deterministic activations)
    model.forward_encoder(volumes, mask_ratio=0.0)

    handle.remove()

    act = captured["act"].contiguous()  # (B, L, C)
    B, L, C = act.shape
    return act.view(B * L, C)  # flatten tokens

# ═════════════════════════ training loop ═══════════════════════════════
def train_sae(args):
    device = get_device()
    torch.backends.cudnn.benchmark = True

    # ─── dataset ────────────────────────────────────────────────────
    shards = sorted(Path(args.shard_dir).expanduser().glob("shard*.tar"))
    dataset = TarShardDataset(shards, args.img_size, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # batch of *volumes*
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    # ─── MAE backbone (frozen) ─────────────────────────────────────
    archs = {
        "small": mae_vit_3d_small_conv,
        "base": mae_vit_3d_base_conv,
        "large": mae_vit_3d_large_conv,
        "hemibrain_optimal": mae_vit_3d_hemibrain_optimal_conv,
    }
    mae = archs[args.model_arch](
        volume_size=(args.img_size,) * 3,
        patch_size=args.patch_size,
        norm_pix_loss=False,
        mask_ratio=args.initial_masking_ratio,
    ).to(device)
    mae.eval()
    for p in mae.parameters():
        p.requires_grad = False

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        missing = mae.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded MAE checkpoint – missing keys: {missing.missing_keys}")

    # Determine token dimension C from model
    dummy = torch.zeros(1, 1, args.img_size, args.img_size, args.img_size)
    C = extract_patch_tokens(mae, dummy.to(device), args.layer).shape[1]
    print(f"Token feature dimension: {C}")

    latent_dim = args.latent_dim or C * args.latent_dim_multiplier
    print(f"Latent dim: {latent_dim}")

    sae = LinearSAE(C, latent_dim).to(device)
    optim = torch.optim.AdamW(sae.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    global_step, samples = 0, 0
    pbar = tqdm(range(args.epochs), desc="SAE Epochs")

    for _ in pbar:
        # Use CUDA prefetcher for overlap
        prefetcher = CUDAPrefetcher(loader, device)
        for vols in prefetcher:
            global_step += 1
            tokens = extract_patch_tokens(mae, vols, args.layer)  # (N_tokens, C)
            tokens = tokens.to(device)

            # Mini-batch over tokens to fit memory
            for chunk in tokens.split(args.token_chunk_size):
                recon, z = sae(chunk)
                mse = F.mse_loss(recon, chunk)
                l1 = z.abs().mean()
                loss = mse + args.l1_coeff * l1

                optim.zero_grad()
                loss.backward()
                optim.step()

                samples += chunk.shape[0]

            if global_step % args.log_interval == 0:
                pbar.set_postfix(
                    step=global_step,
                    loss=float(loss),
                    mse=float(mse),
                    sparsity=float((z == 0).float().mean()),
                )

            if global_step >= args.total_steps:
                break
        if global_step >= args.total_steps:
            break

    # ─── save ──────────────────────────────────────────────────────
    out = {
        "weight": sae.weight.detach().cpu(),
        "input_dim": C,
        "latent_dim": latent_dim,
        "layer": args.layer,
        "l1_coeff": args.l1_coeff,
        "args": vars(args),
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved SAE to {out_path}")

# ═════════════════════════ argparse ═══════════════════════════════════
if __name__ == "__main__":
    P = argparse.ArgumentParser("Sparse Auto-Encoder trainer for ViT-MAE-3D activations")
    # Data / MAE model
    P.add_argument("--shard_dir", required=True, help="Directory with shard_XXXXX.tar files")
    P.add_argument("--checkpoint", required=True, help="Pre-trained MAE checkpoint (.pt)")
    P.add_argument("--model_arch", default="base", choices=["small", "base", "large", "hemibrain_optimal"],
                   help="Which MAE architecture to instantiate (must match checkpoint)")
    # EMA config reuse – needed for kwargs
    P.add_argument("--img_size", type=int, default=96)
    P.add_argument("--patch_size", type=int, default=8)
    P.add_argument("--initial_masking_ratio", type=float, default=0.0, help="Keep all patches during SAE training")

    # SAE specifics
    P.add_argument("--layer", type=int, default=6, help="Encoder block index (0-based) to extract activations from")
    P.add_argument("--latent_dim", type=int, default=None, help="Explicit latent size (overrides multiplier)")
    P.add_argument("--latent_dim_multiplier", type=int, default=4,
                   help="If --latent_dim not set, use multiplier × input_dim (default 4×)")
    P.add_argument("--l1_coeff", type=float, default=1e-4, help="L1 coefficient for sparsity")

    # Optimisation
    P.add_argument("--batch_size", type=int, default=16, help="Number of *volumes* per MAE forward pass")
    P.add_argument("--token_chunk_size", type=int, default=4096, help="Token sub-batch size for SAE training")
    P.add_argument("--learning_rate", type=float, default=1e-3)
    P.add_argument("--epochs", type=int, default=1)
    P.add_argument("--total_steps", type=int, default=10_000, help="Stop after this many MAE forward passes")
    P.add_argument("--num_workers", type=int, default=16)
    P.add_argument("--prefetch_factor", type=int, default=4)
    P.add_argument("--log_interval", type=int, default=50)

    # Output
    P.add_argument("--output_path", default="checkpoints/sae_layer6.pt")

    args = P.parse_args()
    train_sae(args) 