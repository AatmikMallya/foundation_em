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
import argparse, math, random, signal, time
from pathlib import Path
from collections import deque
from typing import Optional
from inspect import isclass
from itertools import islice

    # ───────────────────────── 3rd-party
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# AMP dtype mapping (bf16 recommended for H100)
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# ───────────────────────── project
from vol_train import TarShardDataset, CUDAPrefetcher, load_model_checkpoint  # re-use optimised loaders
from vit_3d import (
    mae_vit_3d_small_conv, mae_vit_3d_base_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv, mae_vit_3d_base_patch_conv,
    get_device,
)
from enhanced_visualization import enhanced_visualize_reconstructions

# ═════════════════════════ SAE module ═══════════════════════════════════
class LinearSAE(torch.nn.Module):
    """Linear sparse auto-encoder following Anthropic's architecture.
    
    Key features:
    - Separate encoder and decoder weights (not tied)
    - Both encoder bias (b_e) and decoder bias (b_d)
    - Gradient projection for proper dictionary normalization
    """
    def __init__(self, input_dim: int, latent_dim: int, activation: str = "relu", k_sparse: Optional[int] = None):
        super().__init__()
        # Token-wise LayerNorm replaces external whitening
        self.input_norm = torch.nn.Identity()     # disable for sanity check
        self.activation = activation
        
        # If k_sparse is provided (>0) we will keep only the k strongest (absolute) activations per token
        if k_sparse is not None and k_sparse <= 0:
            raise ValueError("k_sparse must be a positive integer or None")
        self.k_sparse = k_sparse
        
        # Encoder weights and bias
        self.encoder_weight = torch.nn.Parameter(torch.empty(latent_dim, input_dim))
        self.encoder_bias = torch.nn.Parameter(torch.zeros(latent_dim))
        
        # Decoder weights (dictionary) and bias
        self.decoder_weight = torch.nn.Parameter(torch.empty(latent_dim, input_dim))
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_dim))
        
        # Initialize all weights with Kaiming uniform (Anthropic's choice)
        torch.nn.init.kaiming_uniform_(self.encoder_weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.decoder_weight, a=math.sqrt(5))
        
        # Track dead latents for resampling
        self.register_buffer('latent_acts', torch.zeros(latent_dim))
        self.register_buffer('steps_since_active', torch.zeros(latent_dim, dtype=torch.long))

    def encode(self, x):
        """Encode with Anthropic's formula: f = ReLU(W_e * (x - b_d) + b_e)"""
        x_centered = x - self.decoder_bias
        linear_out = F.linear(x_centered, self.encoder_weight, self.encoder_bias)
        if self.activation == "relu":
            f = F.relu(linear_out)
        elif self.activation == "gelu":
            f = F.gelu(linear_out, approximate="tanh")
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return f

    def decode(self, f):
        """Decode with dictionary: x_hat = f @ W_d + b_d"""
        # f: (batch, latent_dim), decoder_weight: (latent_dim, input_dim)
        # f @ decoder_weight = (batch, input_dim)
        return f @ self.decoder_weight + self.decoder_bias

    def forward(self, x):
        f = self.encode(x)
        # ─── k-winners sparsity (top-k) ────────────────────────────
        if self.k_sparse is not None:
            # Keep sign and magnitude of the k strongest activations per sample
            # Compute absolute values to find strongest activations regardless of sign
            with torch.no_grad():
                topk_idx = torch.topk(f.abs(), k=self.k_sparse, dim=1, largest=True, sorted=False).indices
                mask = torch.zeros_like(f, dtype=torch.bool)
                mask.scatter_(1, topk_idx, True)
            f = f * mask.to(f.dtype)

        x_hat = self.decode(f)
        return x_hat, f
    
    @torch.no_grad()
    def normalize_decoder_weights_proper(self):
        """Proper dictionary normalization with gradient projection (Anthropic method)."""
        # Normalize decoder rows to unit norm (each row is a dictionary vector)
        norms = self.decoder_weight.norm(dim=1, keepdim=True).clamp_(min=1e-8)
        self.decoder_weight.data /= norms
    
    def apply_gradient_projection(self):
        """Project gradients orthogonal to dictionary vectors before optimizer step."""
        if self.decoder_weight.grad is not None:
            # For each dictionary vector (row), remove gradient component parallel to it
            with torch.no_grad():
                # Compute dot product of gradient with normalized dictionary vector
                dict_normalized = F.normalize(self.decoder_weight.data, p=2, dim=1)  # (latent_dim, input_dim)
                grad_parallel = torch.sum(self.decoder_weight.grad * dict_normalized, dim=1, keepdim=True)
                # Remove parallel component
                self.decoder_weight.grad -= grad_parallel * dict_normalized
    
    @torch.no_grad()
    def update_dead_latent_stats(self, f):
        """Track which latents are active for dead neuron resampling."""
        if self.activation == "relu":
            active_mask = (f > 0).any(dim=0)  # (latent_dim,)
        else:  # gelu or other signed activations
            active_mask = (f.abs() > 1e-6).any(dim=0)  # (latent_dim,)
        
        self.latent_acts += active_mask.float()
        
        # Increment steps since active for inactive latents
        self.steps_since_active += ~active_mask
        # Reset counter for active latents
        self.steps_since_active[active_mask] = 0
    
    @torch.no_grad()
    def resample_dead_latents_anthropic(self, high_loss_tokens, dead_threshold=12500):
        """Anthropic-style resampling: more sophisticated approach."""
        if high_loss_tokens.numel() == 0:
            return 0
            
        # Find dead latents
        dead_mask = self.steps_since_active > dead_threshold
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        
        if len(dead_indices) == 0:
            return 0
        
        # Sample from high-loss tokens (weighted by squared loss as in Anthropic)
        n_resample = min(len(dead_indices), high_loss_tokens.shape[0])
        if n_resample == 0:
            return 0
            
        # Sample according to loss-squared probability
        sampled_indices = torch.randperm(high_loss_tokens.shape[0])[:n_resample]
        sampled_tokens = high_loss_tokens[sampled_indices]
        
        # Normalize to unit L2 norm (dictionary vectors)
        sampled_tokens_norm = F.normalize(sampled_tokens, p=2, dim=1)
        
        # Assign to dead latents
        dead_to_resample = dead_indices[:n_resample]
        
        # Set dictionary vectors (decoder weights are rows)
        self.decoder_weight.data[dead_to_resample] = sampled_tokens_norm
        
        # Set encoder vectors: same direction but scaled to avg norm * 0.2
        avg_encoder_norm = self.encoder_weight.data.norm(dim=1).mean()
        encoder_scale = avg_encoder_norm * 0.2
        self.encoder_weight.data[dead_to_resample] = sampled_tokens_norm * encoder_scale
        
        # Reset encoder bias to zero for resampled neurons
        self.encoder_bias.data[dead_to_resample] = 0.0
        
        # Reset tracking for resampled latents
        self.steps_since_active[dead_to_resample] = 0
        self.latent_acts[dead_to_resample] = 0
        
        return len(dead_to_resample)

# ═════════════════════════ Gated Top-K SAE ═════════════════════════════
class GatedSAE(torch.nn.Module):
    """Encoder is split into a binary *gate* pathway and a ReLU magnitude pathway.
    L1 penalty is applied only to the gate, avoiding shrinkage.
    Optionally keep only k strongest magnitudes per token (Top-K).
    """
    def __init__(self, input_dim:int, latent_dim:int, k_sparse:Optional[int]=None):
        super().__init__()
        self.activation="relu"  # for compatibility with helper functions
        if k_sparse is not None and k_sparse<=0:
            raise ValueError("k_sparse must be positive or None")
        self.k_sparse = k_sparse
        # magnitude encoder
        self.mag_weight  = torch.nn.Parameter(torch.empty(latent_dim, input_dim))
        self.mag_bias    = torch.nn.Parameter(torch.zeros(latent_dim))
        # gate encoder
        self.gate_weight = torch.nn.Parameter(torch.empty(latent_dim, input_dim))
        self.gate_bias   = torch.nn.Parameter(torch.zeros(latent_dim))
        # decoder
        self.decoder_weight = torch.nn.Parameter(torch.empty(latent_dim, input_dim))
        self.decoder_bias   = torch.nn.Parameter(torch.zeros(input_dim))
        # init
        torch.nn.init.kaiming_uniform_(self.mag_weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.decoder_weight, a=math.sqrt(5))

        # dead latent tracking buffers (re-use code later)
        self.register_buffer('latent_acts', torch.zeros(latent_dim))
        self.register_buffer('steps_since_active', torch.zeros(latent_dim, dtype=torch.long))

    def encode(self,x):
        x_c = x - self.decoder_bias
        mag = F.relu(F.linear(x_c, self.mag_weight,  self.mag_bias))
        gate_pre = F.linear(x_c, self.gate_weight, self.gate_bias)
        gate_prob = torch.sigmoid(gate_pre)
        # straight-through binary gate
        gate_bin = (gate_prob>0.5).float()
        gate = gate_prob + (gate_bin - gate_prob).detach()
        if self.k_sparse is not None:
            with torch.no_grad():
                topk_idx = torch.topk(mag, k=self.k_sparse, dim=1, largest=True, sorted=False).indices
                mask = torch.zeros_like(mag, dtype=torch.bool)
                mask.scatter_(1, topk_idx, True)
            mag = mag*mask.float()
        z = gate * mag
        return z, gate

    def decode(self,f):
        return f @ self.decoder_weight + self.decoder_bias

    def forward(self,x):
        z, gate = self.encode(x)
        recon = self.decode(z)
        return recon, z, gate

    # reuse helper functions from LinearSAE
    normalize_decoder_weights_proper = LinearSAE.normalize_decoder_weights_proper
    apply_gradient_projection       = LinearSAE.apply_gradient_projection
    update_dead_latent_stats        = LinearSAE.update_dead_latent_stats
    resample_dead_latents_anthropic = LinearSAE.resample_dead_latents_anthropic

# ═════════════════════════ helpers ═════════════════════════════════════
class TokenExtractor:
    """Optimized token extractor that can extract from various points in the MAE model."""
    def __init__(self, model, layer_idx: int, extract_from: str = "encoder"):
        self.model = model
        self.layer_idx = layer_idx
        self.extract_from = extract_from.lower()
        self.captured = {}
        
        if self.extract_from == "patchembed":
            # Extract from PatchEmbed output (raw CNN features)
            def _hook(_module, _inp, out):
                # out: (B, L, C) - direct patch embeddings
                self.captured["act"] = out.detach()  # (B, L, C)
            
            self.handle = model.encoder.patch_embed.register_forward_hook(_hook)
            print(f"TokenExtractor: Extracting from PatchEmbed output (raw CNN features)")
            
        elif self.extract_from == "encoder":
            # Extract from encoder block (default behavior)
            def _hook(_module, _inp, out):
                # out: (B, 1+L, C).  Drop CLS token.
                self.captured["act"] = out[:, 1:, :].detach()  # (B, L, C)
            
            self.handle = model.encoder.blocks[layer_idx].register_forward_hook(_hook)
            print(f"TokenExtractor: Extracting from encoder block {layer_idx}")
            
        else:
            raise ValueError(f"Unknown extract_from: {extract_from}. Use 'patchembed' or 'encoder'")
    
    @torch.no_grad()
    def extract_tokens(self, volumes):
        """Extract tokens without re-registering hook."""
        # Clear previous activations
        self.captured.clear()
        
        if self.extract_from == "patchembed":
            # For PatchEmbed, we only need to run through the patch embedding
            # No masking, no encoder blocks
            x = self.model.encoder.patch_embed(volumes)
            # Hook will capture the output automatically
            act = self.captured["act"].contiguous()  # (B, L, C)
        else:
            # For encoder blocks, run full forward_encoder
            self.model.forward_encoder(volumes, mask_ratio=0.0)
            act = self.captured["act"].contiguous()  # (B, L, C)
        
        B, L, C = act.shape
        return act.view(B * L, C).contiguous()  # flatten tokens with optimal memory layout
    
    def cleanup(self):
        """Remove the hook when done."""
        if hasattr(self, 'handle'):
            self.handle.remove()

class ActivationInjector:
    """Inject modified activations back into the MAE model for reconstruction evaluation."""
    def __init__(self, model, layer_idx: int, extract_from: str = "encoder"):
        self.model = model
        self.layer_idx = layer_idx
        self.extract_from = extract_from.lower()
        self.replacement_activations = None
        self.original_hook = None
        self.injection_handle = None
        
    def set_replacement_activations(self, activations):
        """Set the activations to inject. Shape: (B*L, C) -> will be reshaped to (B, L, C)"""
        self.replacement_activations = activations
        
    def _create_injection_hook(self, target_shape):
        """Create a hook that replaces activations with our modified ones."""
        def _inject_hook(module, input, output):
            if self.replacement_activations is not None:
                B, L_total, C = target_shape
                # Only drop the CLS token if we're operating on an encoder block
                if self.extract_from == "encoder":
                    L_tokens = L_total - 1
                else:                       # patchembed → no CLS
                    L_tokens = L_total
                
                # Reshape flat activations back to (B, L, C)
                reshaped_acts = self.replacement_activations.view(B, L_tokens, C)
                
                if self.extract_from == "encoder":
                    # For encoder blocks, we need to preserve the CLS token
                    # output is (B, 1+L, C), so we replace everything except first token
                    modified_output = output.clone()
                    modified_output[:, 1:, :] = reshaped_acts
                    return modified_output
                else:
                    # For PatchEmbed, directly return the reshaped activations
                    return reshaped_acts
            return output
        return _inject_hook
        
    @torch.no_grad()
    def inject_and_reconstruct(self, volumes):
        """Run MAE with injected activations to get modified reconstruction."""
        if self.replacement_activations is None:
            raise ValueError("Must call set_replacement_activations() first")
            
        # First, do a forward pass to get the target shape
        if self.extract_from == "patchembed":
            x = self.model.encoder.patch_embed(volumes)
            target_shape = x.shape  # (B, L, C)
            hook_target = self.model.encoder.patch_embed
        else:
            # For encoder blocks, we need to get the shape after the target layer
            latent, mask, ids_restore = self.model.forward_encoder(volumes, mask_ratio=0.0)
            # latent is (B, 1+L_visible, C) but we want full shape
            B, _, C = latent.shape
            # Calculate original L from volume size and patch size
            volume_size = volumes.shape[-1]  # Assuming cubic volumes
            patch_size = self.model.encoder.patch_embed.patch_size[0]
            L = (volume_size // patch_size) ** 3
            target_shape = (B, 1 + L, C)  # +1 for CLS token
            hook_target = self.model.encoder.blocks[self.layer_idx]
        
        # Register the injection hook
        injection_hook = self._create_injection_hook(target_shape)
        self.injection_handle = hook_target.register_forward_hook(injection_hook)
        
        try:
            # Run full MAE forward pass with injection
            loss, pred, mask, _ = self.model(volumes, mask_ratio=0.0)
            return pred, mask
        finally:
            # Always cleanup the hook
            if self.injection_handle:
                self.injection_handle.remove()
                self.injection_handle = None

    @staticmethod
    @torch.no_grad()
    def _reconstruct_with_mask(mae, vols, mask_ratio, injector=None):
        """
        Run MAE once, optionally with SAE-injected activations, and rebuild a *full*
        volume in which **visible** patches come directly from the GT input while
        **masked** patches are MAE predictions.

        Returns
        -------
        rec  : (B,1,D,H,W)  rebuilt volume
        mask : (B,L)        0=visible 1=masked (same order as patchify)
        """
        amp_dtype = next(mae.parameters()).dtype
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            out  = mae(vols, mask_ratio=mask_ratio)
            loss, pred, mask = out[0], out[1], out[2] if len(out) > 2 else None

        if injector is not None:
            injector.set_replacement_activations(injector.replacement_activations)
            # identical mask pattern – injector only changes activations
            _, pred, mask = injector.inject_and_reconstruct_with_masking(vols, mask_ratio)

        # stitch prediction + visible GT back into a full grid
        patchified = mae.patchify(vols).float()            # (B,L,P³C)
        pred = pred.float()
        patchified[mask.bool()] = pred[mask.bool()]
        rec = mae.unpatchify(patchified)           # (B,1,D,H,W)
        return rec, mask
                
    @torch.no_grad()
    def inject_and_reconstruct_with_masking(self, volumes, mask_ratio):
        """Run MAE with injected activations and masking to get modified reconstruction loss."""
        if self.replacement_activations is None:
            raise ValueError("Must call set_replacement_activations() first")
        
            
        # This is more complex - for now, use a simpler approximation
        # Run the injection without masking first
        pred, _ = self.inject_and_reconstruct(volumes)
        
        # Then compute loss using MAE's loss function with a mask pattern
        # Get original mask pattern
        with torch.cuda.amp.autocast(dtype=next(self.model.parameters()).dtype):
            _, _, mask_orig, _ = self.model(volumes, mask_ratio=mask_ratio)
        
        # Compute loss using original mask pattern
        loss, _ = self.model.forward_loss(volumes, pred, mask_orig)
        return loss, pred, mask_orig

@torch.no_grad()
def compute_masked_mae_loss_with_sae(mae, volumes, layer_idx, sae, device, token_extractor, extract_from="encoder", mask_ratio=0.85):
    """Compute MAE loss with SAE-modified activations, handling masking properly."""
    mae.eval()
    
    if extract_from == "patchembed":
        # -----------------------------------------------------------
        # 1) full PatchEmbed → C tokens               (no CLS token)
        # -----------------------------------------------------------
        tokens_full = mae.encoder.patch_embed(volumes)          # (B, L, C)
        B, L, C = tokens_full.shape
        flat = tokens_full.reshape(B * L, C)

        # -----------------------------------------------------------
        # 2) run SAE on *every* token (not just the visible ones)
        # -----------------------------------------------------------
        if isinstance(sae, GatedSAE):
            sae_recon, _, _ = sae(flat)
        else:
            sae_recon, _ = sae(flat)

        sae_tokens_full = sae_recon

        # -----------------------------------------------------------
        # 3) inject back into MAE and let MAE handle masking
        # -----------------------------------------------------------
        injector = ActivationInjector(mae, layer_idx,
                                      extract_from="patchembed")
        injector.set_replacement_activations(sae_tokens_full)   # (B*L, C)

        loss_sae_masked, _, _ = injector.inject_and_reconstruct_with_masking(
            volumes, mask_ratio
        )
        return float(loss_sae_masked)
        
    else:
        # For encoder layer injection, we need to run up to that layer normally, then inject
        # This is more complex and requires running partial forward passes
        # For now, use a simpler approximation
        
        # Get original masked forward pass
        latent_orig, mask, ids_restore = mae.forward_encoder(volumes, mask_ratio)
        
        # Extract all tokens (not just visible ones) and apply SAE
        all_tokens = token_extractor.extract_tokens(volumes)  # This gets all patches
        
        if isinstance(sae, GatedSAE):
            sae_recon, _, _ = sae(all_tokens)
        else:
            sae_recon, _ = sae(all_tokens)
        
        sae_tokens = sae_recon
        
        # This is an approximation - inject SAE tokens and run full forward
        injector = ActivationInjector(mae, layer_idx, extract_from=extract_from)
        injector.set_replacement_activations(sae_tokens)
        loss_sae, pred_sae, _ = injector.inject_and_reconstruct_with_masking(volumes, mask_ratio)
        return float(loss_sae)
    
    injector = ActivationInjector(mae, layer_idx, extract_from="patchembed")
    injector.set_replacement_activations(x_sae.reshape(-1, x_sae.size(-1)))  # flat (B*L_vis, C)

    loss_sae_masked, _, _ = injector.inject_and_reconstruct_with_masking(
        volumes, mask_ratio
    )
    return float(loss_sae_masked)

@torch.no_grad()
def compute_reconstruction_mse_with_sae(
    mae, volumes, layer_idx, sae,
    device, token_extractor,
    extract_from="encoder", mask_ratio=0.85
):
    """
    MSE between the vanilla-MAE reconstruction and the SAE-patched one
    (plus a few auxiliary metrics).  All heavy ops remain in bf16; we
    cast to fp32 only for the final scalars.
    """
    amp_dtype = next(mae.parameters()).dtype          # bf16 in your run
    volumes   = volumes.to(dtype=amp_dtype)           # safety first
    mae.eval()

    # ---------- 1 · plain MAE recon (no masking) --------------------
    with torch.cuda.amp.autocast(dtype=amp_dtype):
        loss_orig_full, pred_orig_full, *rest = mae(volumes, mask_ratio=0.0)
        mask_orig_full = rest[0] if rest else None

    # ---------- 2 · MAE recon with training mask -------------------
    with torch.cuda.amp.autocast(dtype=amp_dtype):
        loss_orig_masked, pred_orig_masked, *rest = mae(volumes, mask_ratio=mask_ratio)
        mask_orig_masked = rest[0] if rest else None

    # ---------- 3 · SAE-patched activations ------------------------
    orig_tokens    = token_extractor.extract_tokens(volumes)      # bf16
    sae_recon, _   = sae(orig_tokens)                          # bf16
    sae_tokens     = sae_recon

    injector = ActivationInjector(mae, layer_idx, extract_from)
    injector.set_replacement_activations(sae_tokens)
    with torch.cuda.amp.autocast(dtype=amp_dtype):
        pred_sae_full, _ = injector.inject_and_reconstruct(volumes)

    # ---------- 4 · masked MAE loss with SAE (still bf16) ----------
    loss_sae_masked = compute_masked_mae_loss_with_sae(
        mae, volumes, layer_idx, sae,
        device, token_extractor, extract_from, mask_ratio
    )

    # ---------- 5 · error metrics ----------------------------------
    # Keep everything in the same dtype for mse computations
    mse_recon          = F.mse_loss(pred_sae_full, pred_orig_full)
    rec_orig_full_bf16 = mae.unpatchify(pred_orig_full)           # bf16
    rec_sae_full_bf16  = mae.unpatchify(pred_sae_full)            # bf16
    mse_orig_vs_input  = F.mse_loss(rec_orig_full_bf16, volumes)
    mse_sae_vs_input   = F.mse_loss(rec_sae_full_bf16,  volumes)

    # ---------- 6 · cast *only scalars* to Python floats -----------
    return {
        "reconstruction_mse"  : float(mse_recon),
        "orig_vs_input_mse"   : float(mse_orig_vs_input),
        "sae_vs_input_mse"    : float(mse_sae_vs_input),
        "orig_mae_loss"       : float(loss_orig_masked),
        "sae_mae_loss"        : float(loss_sae_masked),
        "mae_loss_diff"       : float(loss_sae_masked - loss_orig_masked),
        # full-resolution volumes stay in bf16 – callers decide when to .float()
        "pred_orig"           : rec_orig_full_bf16,
        "pred_sae"            : rec_sae_full_bf16,
        "mask"                : mask_orig_full,
    }

def visualize_sae_reconstructions(mae, sae, vis_loader, device, step, 
                                layer_idx, token_extractor, model_dtype, extract_from="encoder", 
                                num_examples=3, save_dir="vis"):
    """Create visualization comparing original input, MAE reconstruction, and SAE-modified reconstruction."""
    mae.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    vis_paths = []
    
    try:
        for i, volumes in enumerate(vis_loader):
            if i >= num_examples:
                break
                
            volumes = volumes.to(device, dtype=model_dtype)
            B = volumes.shape[0]
            
            # Get reconstructions
            recon_data = compute_reconstruction_mse_with_sae(
                mae, volumes, layer_idx, sae, device, token_extractor, extract_from
            )
            
            pred_orig = recon_data['pred_orig'] 
            pred_sae = recon_data['pred_sae']
            mask = recon_data['mask']
            
            for b in range(min(B, num_examples - i)):
                vol_orig = volumes[b, 0].to(torch.float32).cpu().numpy()  # Original input
                vol_mae = pred_orig[b, 0].to(torch.float32).cpu().numpy()  # MAE reconstruction  
                vol_sae = pred_sae[b, 0].to(torch.float32).cpu().numpy()   # SAE-modified reconstruction
                
                # Create comparison plot with 3 columns
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle(f'Step {step} - Sample {i*num_examples + b + 1}\n'
                           f'Orig MSE: {recon_data["orig_vs_input_mse"]:.4f}, '
                           f'SAE MSE: {recon_data["sae_vs_input_mse"]:.4f}, '
                           f'Diff MSE: {recon_data["reconstruction_mse"]:.4f}\n'
                           f'Orig MAE Loss: {recon_data["orig_mae_loss"]:.4f}, '
                           f'SAE MAE Loss: {recon_data["sae_mae_loss"]:.4f}, '
                           f'MAE Loss Diff: {recon_data["mae_loss_diff"]:.4f}')
                
                # Show middle slices in each dimension
                D, H, W = vol_orig.shape
                slices = [D//2, H//2, W//2]
                
                volumes_to_show = [vol_orig, vol_mae, vol_sae]
                titles = ['Original Input', 'MAE Reconstruction', 'SAE-Modified Reconstruction']
                
                for col, (vol, title) in enumerate(zip(volumes_to_show, titles)):
                    # Z slice (XY plane)
                    axes[0, col].imshow(vol[slices[0], :, :], cmap='gray')
                    axes[0, col].set_title(f'{title}\nZ-slice {slices[0]}')
                    axes[0, col].axis('off')
                    
                    # Y slice (XZ plane) 
                    axes[1, col].imshow(vol[:, slices[1], :], cmap='gray')
                    axes[1, col].set_title(f'Y-slice {slices[1]}')
                    axes[1, col].axis('off')
                    
                    # X slice (YZ plane)
                    axes[2, col].imshow(vol[:, :, slices[2]], cmap='gray')
                    axes[2, col].set_title(f'X-slice {slices[2]}')
                    axes[2, col].axis('off')
                
                # plt.tight_layout()
                
                # Save the plot
                save_path = save_dir / f"sae_recon_step_{step}_sample_{i*num_examples + b + 1}.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                vis_paths.append(save_path)
                
    except Exception as e:
        print(f"Error in SAE visualization: {e}")
        import traceback
        traceback.print_exc()
    
    return vis_paths

@torch.no_grad()
def extract_patch_tokens(model, volumes, layer_idx: int, extract_from: str = "encoder"):
    """Legacy function for backward compatibility - creates extractor each time."""
    extractor = TokenExtractor(model, layer_idx, extract_from=extract_from)
    try:
        return extractor.extract_tokens(volumes)
    finally:
        extractor.cleanup()

@torch.no_grad()
def run_sae_validation(sae, mae, val_loader, layer_idx, device, token_extractor, extract_from="encoder", sae_variant="linear", max_batches=10):
    """Run SAE validation on a subset of validation data."""
    sae.eval()
    val_losses, val_mses, val_l1s, val_sparsities = [], [], [], []
    recon_mses, orig_vs_input_mses, sae_vs_input_mses = [], [], []
    orig_mae_losses, sae_mae_losses, mae_loss_diffs = [], [], []
    
    # Get model dtype to ensure input consistency
    model_dtype = next(mae.parameters()).dtype

    
    batch_count = 0
    for vols in val_loader:
        if batch_count >= max_batches:
            break
        
        vols = vols.to(device, dtype=model_dtype)
        tokens = token_extractor.extract_tokens(vols)
        
        # Compute reconstruction MSE for this batch (using smaller subset for speed)
        if batch_count < 3:  # Only compute reconstruction MSE for first few batches (expensive)
            recon_data = compute_reconstruction_mse_with_sae(
                mae, vols, layer_idx, sae, device, token_extractor, extract_from
            )
            recon_mses.append(recon_data['reconstruction_mse'])
            orig_vs_input_mses.append(recon_data['orig_vs_input_mse'])
            sae_vs_input_mses.append(recon_data['sae_vs_input_mse'])
            orig_mae_losses.append(recon_data['orig_mae_loss'])
            sae_mae_losses.append(recon_data['sae_mae_loss'])
            mae_loss_diffs.append(recon_data['mae_loss_diff'])

        
        # Process in chunks to match training
        for chunk in tokens.split(4096):  # Use smaller chunks for validation
            if sae_variant=="gated":
                recon, z, gate = sae(chunk)
                l1 = gate.abs().sum(dim=1).mean()
            else:
                recon, z = sae(chunk)
                l1 = z.abs().sum(dim=1).mean()  # For linear SAE, compute L1 of activations
            mse = F.mse_loss(recon, chunk)
            val_mses.append(float(mse.detach()))
            val_l1s.append(float(l1.detach()))
            val_sparsities.append(float((z.detach().abs() < 1e-6).float().mean()))
        
        batch_count += 1
    
    sae.train()
    metrics = {
        'val_mse': sum(val_mses) / len(val_mses) if val_mses else 0.0,
        'val_l1': sum(val_l1s) / len(val_l1s) if val_l1s else 0.0,
        'val_sparsity': sum(val_sparsities) / len(val_sparsities) if val_sparsities else 0.0,
        'val_frac_active': 1.0 - (sum(val_sparsities) / len(val_sparsities)) if val_sparsities else 0.0,
    }
    
    # Add reconstruction metrics if available
    if recon_mses:
        metrics['val_reconstruction_mse'] = sum(recon_mses) / len(recon_mses)
        metrics['val_orig_vs_input_mse'] = sum(orig_vs_input_mses) / len(orig_vs_input_mses)
        metrics['val_sae_vs_input_mse'] = sum(sae_vs_input_mses) / len(sae_vs_input_mses)
        
        # Add MAE loss metrics (averaged across batches)
        metrics['val_orig_mae_loss'] = sum(orig_mae_losses) / len(orig_mae_losses)
        metrics['val_sae_mae_loss'] = sum(sae_mae_losses) / len(sae_mae_losses)
        metrics['val_mae_loss_diff'] = sum(mae_loss_diffs) / len(mae_loss_diffs)
    
    return metrics


def visualise_masked_effect(mae, sae, vols, token_extractor,
                            layer_idx,
                            mask_ratio=0.85):
    """
    Figure with:
      GT  |  MAE recon  |  SAE-patched recon  |  Δ-error heat-map
    on the *same* random-mask pattern.
    """
    model_dtype = next(mae.parameters()).dtype
    # baseline MAE
    base_rec, mask = ActivationInjector._reconstruct_with_mask(mae, vols, mask_ratio)
    base_rec = base_rec.to(dtype=model_dtype)
    vols = vols.to(dtype=model_dtype)

    # prepare SAE activations
    tokens   = token_extractor.extract_tokens(vols)

    if isinstance(sae, GatedSAE):
        sae_rec, _, _ = sae(tokens)
    else:
        sae_rec, _ = sae(tokens)
    sae_tokens = sae_rec
    sae_tokens = sae_tokens.to(dtype=model_dtype)
    injector = ActivationInjector(mae, layer_idx)
    injector.set_replacement_activations(sae_tokens)

    # SAE-patched reconstruction
    sae_rec_vol, _ = ActivationInjector._reconstruct_with_mask(mae, vols, mask_ratio, injector) 
    sae_rec_vol = sae_rec_vol.to(dtype=model_dtype)

    # absolute-error maps on masked voxels only
    err_base = (base_rec - vols).abs()
    err_base = err_base.float()
    err_sae  = (sae_rec_vol - vols).abs()
    err_sae = err_sae.float()
    # keep masked voxels
    voxel_mask = patchmask_to_voxelmask(mask, patch_size=mae.patch_size,
                                    volume_size=mae.volume_size)
    err_base[~voxel_mask] = 0
    err_sae [~voxel_mask] = 0

    z = vols.size(2) // 2                     # mid-slice
    fig, ax = plt.subplots(1, 4, figsize=(16,4))

    ax[0].imshow(vols[0,0,z].float().cpu().numpy(),        cmap='gray'); ax[0].set_title('GT');             ax[0].axis('off')
    ax[1].imshow(base_rec[0,0,z].float().cpu().numpy(),    cmap='gray'); ax[1].set_title('MAE');            ax[1].axis('off')
    ax[2].imshow(sae_rec_vol[0,0,z].float().cpu().numpy(), cmap='gray'); ax[2].set_title('SAE-patched');    ax[2].axis('off')
    delta = (err_base - err_sae)[0,0,z].float().cpu()
    vmax  = delta.abs().max()
    ax[3].imshow(delta, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax[3].set_title('error Δ\n(blue = better)');         ax[3].axis('off')
    # fig.tight_layout()
    return fig

import math 

# add once near the other imports
import math   # needed for ceil() and divmod()

# ----------------------------------------------------------------------
def visualize_sae_slice_grid(
    mae, sae, vols, token_extractor,
    layer_idx,
    step, tag="sae_slice_grid",
    out_dir="vis", mask_ratio=0.85,
):
    """
    One figure showing, for every volume in `vols`:
        ┌───────── sample 0 ─────────┬──────── sample 1 ────────┬ … ┐
        │  GT | MAE | SAE            │  GT | MAE | SAE          │   │  ← Z‑row
        │  GT | MAE | SAE            │  GT | MAE | SAE          │   │  ← Y‑row
        │  GT | MAE | SAE            │  GT | MAE | SAE          │   │  ← X‑row
        └────────────────────────────┴──────────────────────────┴ … ┘
    A block of 3×3 images is placed for every sample; blocks are tiled
    on an R × C grid (e.g. 2 × 3 when `vis_samples==6`).
    """
    device = vols.device
    B      = vols.size(0)
    dtype  = next(mae.parameters()).dtype            # bf16 / fp16 / fp32

    # ── 1 · MAE truth‑masked reconstruction (baseline) ──────────────
    base_rec, mask = ActivationInjector._reconstruct_with_mask(
        mae, vols, mask_ratio
    )
    base_rec = base_rec.to(dtype=dtype)

    # ── 2 · SAE‑patched reconstruction on the *same* mask pattern ──
    toks = token_extractor.extract_tokens(vols)                   # (B·L, C)
    # toks_wh = (toks - act_mean) / act_std                         # whiten

    if isinstance(sae, GatedSAE):
        sae_recon, _, _ = sae(toks)                            # (B·L, C)
    else:
        sae_recon, _ = sae(toks)

    sae_tok = sae_recon    

    # sae_tok = sae_recon * act_std + act_mean                      # un‑whiten

    inj = ActivationInjector(mae, layer_idx)
    inj.set_replacement_activations(sae_tok.to(dtype=dtype))
    sae_rec, _ = ActivationInjector._reconstruct_with_mask(
        mae, vols, mask_ratio, injector=inj
    )
    sae_rec = sae_rec.to(dtype=dtype)

    # ── 3 · move to CPU / fp32 for Matplotlib ───────────────────────
    vols_cpu = vols.float().cpu()
    base_cpu = base_rec.float().cpu()
    sae_cpu  = sae_rec.float().cpu()

    D, H, W  = vols_cpu.shape[-3:]
    z, y, x  = D // 2, H // 2, W // 2

    # ── 4 · decide sample‑grid layout (≤3 cols/row) ────────────────
    grid_cols = min(3, B)
    grid_rows = math.ceil(B / grid_cols)

    fig_rows  = 3 * grid_rows               # 3 slice‑rows per sample
    fig_cols  = 3 * grid_cols               # GT | MAE | SAE

    fig, ax = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(2.4 * fig_cols, 2.4 * grid_rows),   # generous cell size
        gridspec_kw={"wspace": 0.01, "hspace": 0.01}
    )

    for s in range(B):
        rb, cb = divmod(s, grid_cols)       # sample‑block row / col
        r_off  = rb * 3                     # top row of this block
        c_off  = cb * 3                     # left col of this block

        tiles = [
            (vols_cpu[s, 0,  z],     f"#{s+1} Z"),   (base_cpu[s, 0,  z], "MAE"),   (sae_cpu[s, 0,  z], "SAE"),
            (vols_cpu[s, 0, :, y, :], ""),           (base_cpu[s, 0, :, y, :], ""), (sae_cpu[s, 0, :, y, :], ""),
            (vols_cpu[s, 0, :, :, x], ""),           (base_cpu[s, 0, :, :, x], ""), (sae_cpu[s, 0, :, :, x], ""),
        ]
        for k, (img, title) in enumerate(tiles):
            r = r_off + k // 3
            c = c_off + k % 3
            ax[r, c].imshow(img, cmap="gray")
            if title:
                ax[r, c].set_title(title, fontsize=7)
            ax[r, c].axis("off")

    fig.suptitle(f"{tag} – step {step}", fontsize=14)
    # manual padding ≪ tight_layout (avoids large white margins)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04,
                        wspace=0.01, hspace=0.01)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    fn = out_dir / f"{tag}_step_{step}.png"
    fig.savefig(fn, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fn]


def patchmask_to_voxelmask(mask, patch_size=(8, 8, 8),
                           volume_size=(96, 96, 96)):
    """
    mask : (B, L)      patch‑level mask from MAE
    returns (B, 1, D, H, W) voxel‑wise boolean mask
    """
    B, L = mask.shape
    pd, ph, pw = [volume_size[i] // patch_size[i] for i in range(3)]

    # ---- reshape to grid & convert to *bool* right away -------------
    mask_3d = mask.view(B, pd, ph, pw).bool()        # (B, 12, 12, 12)

    # ---- nearest‑neighbour up‑sampling to voxel resolution ----------
    mask_3d = mask_3d.repeat_interleave(patch_size[0], 1)   # D
    mask_3d = mask_3d.repeat_interleave(patch_size[1], 2)   # H
    mask_3d = mask_3d.repeat_interleave(patch_size[2], 3)   # W

    return mask_3d.unsqueeze(1)            # (B, 1, D, H, W)   *bool* 

# ═════════════════════════ training loop ═══════════════════════════════
def train_sae(args):
    device = get_device()
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # graceful shutdown (e.g., SLURM TIMEOUT)
    cancel = {"stop": False}
    def _handle_sigterm(*_):
        cancel["stop"] = True
        print("\n[signal] SIGTERM received – finishing current step then exiting.")
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ─── dataset ────────────────────────────────────────────────────
    shards = sorted(Path(args.shard_dir).expanduser().glob("shard*.tar"))
    n_val = max(1, int(len(shards) * args.val_split))
    val_shards, train_shards = shards[:n_val], shards[n_val:]
    
    dataset = TarShardDataset(train_shards, args.img_size, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # batch of *volumes*
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        timeout=300  # 5 min timeout for large tar files (proven optimization)
    )
    
    # Validation loader (smaller, for periodic evaluation)
    val_dataset = TarShardDataset(val_shards, args.img_size, shuffle=False)
    val_loader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size // 2,      # lighter on memory
        num_workers=args.num_workers // 2,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        timeout=300
    )

    # Visualization loader (for periodic SAE reconstruction comparisons)
    vis_loader = DataLoader(
        TarShardDataset(val_shards[:1], args.img_size, shuffle=False),
        batch_size=args.vis_samples,          # full mini‑batch for grids
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # ─── MAE backbone (frozen) ─────────────────────────────────────
    archs = {
        "small": mae_vit_3d_small_conv,
        "base": mae_vit_3d_base_conv,
        "base_conv": mae_vit_3d_base_conv,
        "large": mae_vit_3d_large_conv,
        "hemibrain_optimal": mae_vit_3d_hemibrain_optimal_conv,
        "base_patch_conv": mae_vit_3d_base_patch_conv,
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

    # Determine autocast dtype first
    autocast_dtype = _AMP_DTYPE_MAP.get(args.amp_dtype, torch.bfloat16)

    # Load checkpoint with proper dtype handling
    if args.checkpoint:
        try:
            # Try using the load_model_checkpoint function which handles dtype metadata
            mae, checkpoint = load_model_checkpoint(args.checkpoint, mae, device=device)
            print(f"Loaded MAE checkpoint using load_model_checkpoint")
        except Exception as e:
            print(f"Error with load_model_checkpoint, falling back to manual loading: {e}")
            # Fallback to manual loading (original approach)
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt["model_state_dict"]
            
            # Remove _orig_mod. prefix from torch.compile compiled models
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    clean_key = key[len('_orig_mod.'):]
                    clean_state_dict[clean_key] = value
                else:
                    clean_state_dict[key] = value
            
            missing = mae.load_state_dict(clean_state_dict, strict=False)
            print(f"Loaded MAE checkpoint - Missing keys: {len(missing.missing_keys)}, Unexpected keys: {len(missing.unexpected_keys)}")
            if missing.missing_keys:
                print(f"Missing keys (first 5): {missing.missing_keys[:5]}")
            if missing.unexpected_keys:
                print(f"Unexpected keys (first 5): {missing.unexpected_keys[:5]}")
    
    # Ensure model dtype matches autocast dtype (fallback for consistency)
    if args.use_amp:
        current_dtype = next(mae.parameters()).dtype
        if current_dtype != autocast_dtype:
            mae = mae.to(dtype=autocast_dtype, device=device)
            print(f"Converted MAE model from {current_dtype} to {autocast_dtype} for autocast consistency")
        else:
            print(f"MAE model already in correct dtype: {autocast_dtype}")
    else:
        # Ensure model is on correct device even if not using AMP
        mae = mae.to(device)

    # Compile MAE for faster inference AFTER loading checkpoint and dtype conversion
    if args.compile_mae:
        mae = torch.compile(mae, backend="inductor", mode="default")
        print("MAE backbone compiled with torch.compile")
    else:
        print("torch.compile disabled for MAE – avoiding potential memory leak")

    # Create optimized token extractor (register hook once)
    token_extractor = TokenExtractor(mae, args.layer, extract_from=args.extract_from)
    if args.extract_from == "patchembed":
        print(f"Token extractor created for PatchEmbed output (raw CNN features)")
    else:
        print(f"Token extractor created for encoder layer {args.layer}")
    
    # Determine token dimension C from model
    # Create dummy tensor with same dtype as model to avoid dtype mismatch
    model_dtype = next(mae.parameters()).dtype
    dummy = torch.zeros(1, 1, args.img_size, args.img_size, args.img_size, dtype=model_dtype)
    C = token_extractor.extract_tokens(dummy.to(device)).shape[1]
    print(f"Token feature dimension: {C}")

    latent_dim = args.latent_dim or C * args.latent_dim_multiplier
    print(f"Latent dim: {latent_dim}")

    # ─── Activation whitening (DISABLED by user) ──────────────────────────
    # print("Activation whitening DISABLED by user request.")
    # act_mean = torch.zeros(1, C, device=device, dtype=autocast_dtype)
    # act_std  = torch.ones(1, C, device=device, dtype=autocast_dtype)
    # print("Whitening stats set to no-op (mean=0, std=1).")

    if args.sae_variant=="linear":
        sae = LinearSAE(C, latent_dim, activation=args.activation, k_sparse=args.k_sparse).to(device)
    elif args.sae_variant=="gated":
        sae = GatedSAE(C, latent_dim, k_sparse=args.k_sparse).to(device)
    else:
        raise ValueError("unknown sae_variant")
    
    sae = sae.to(device=device, dtype=autocast_dtype)
    # sae = sae.to(device=device)
    
    # Debug: print parameter shapes
    if args.sae_variant=="linear":
        print(f"Encoder weight shape: {sae.encoder_weight.shape}")
        print(f"Encoder bias shape: {sae.encoder_bias.shape}")
    else:
        print(f"Mag weight shape:  {sae.mag_weight.shape}")
        print(f"Gate weight shape: {sae.gate_weight.shape}")
    print(f"Decoder weight shape: {sae.decoder_weight.shape}")
    print(f"Decoder bias shape: {sae.decoder_bias.shape}")
    
    # ─── Model compilation for speed (optional) ─────────────────────────────
    if args.compile_sae:
        sae = torch.compile(sae, backend="inductor", mode="default")
        print("SAE compiled with torch.compile")
    else:
        print("torch.compile disabled for SAE – avoiding potential memory leak")

    
    # Debug: test forward pass with dummy data
    with torch.no_grad():
        dummy_tokens = torch.randn(10, C, device=device, dtype=autocast_dtype)  # 10 tokens, C dimensions
        if args.sae_variant=="gated":
            dummy_recon, dummy_f, _ = sae(dummy_tokens)
        else:
            dummy_recon, dummy_f = sae(dummy_tokens)
        print(f"Forward pass test successful:")
        print(f"  Input: {dummy_tokens.shape}")
        print(f"  Features: {dummy_f.shape}")
        print(f"  Reconstruction: {dummy_recon.shape}")
    
    # Set up optimizer with different learning rates for encoder weights vs decoder bias
    if args.sae_variant=="linear":
        param_groups = [
            {'params': [sae.encoder_weight], 'lr': args.learning_rate, 'weight_decay': 1e-4},
            {'params': [sae.encoder_bias],   'lr': args.learning_rate*3.0, 'weight_decay': 0.0},
            {'params': [sae.decoder_weight], 'lr': args.learning_rate, 'weight_decay': 1e-4},
            {'params': [sae.decoder_bias],   'lr': args.learning_rate*3.0, 'weight_decay': 0.0},
        ]
    else:  # gated
        param_groups = [
            {'params': [sae.mag_weight, sae.gate_weight], 'lr': args.learning_rate, 'weight_decay': 1e-4},
            {'params': [sae.mag_bias, sae.gate_bias],     'lr': args.learning_rate*3.0, 'weight_decay': 0.0},
            {'params': [sae.decoder_weight],              'lr': args.learning_rate, 'weight_decay': 1e-4},
            {'params': [sae.decoder_bias],                'lr': args.learning_rate*3.0, 'weight_decay': 0.0},
        ]
    optim = torch.optim.AdamW(param_groups)
    
    # Debug: Print parameter group information
    print(f"Parameter groups setup for {args.sae_variant} SAE:")
    for i, group in enumerate(param_groups):
        lr = group['lr']
        num_params = len(group['params'])
        param_shapes = [list(p.shape) for p in group['params']]
        print(f"  Group {i}: lr={lr:.2e}, {num_params} parameters, shapes={param_shapes}")
    
    print(f"Warmup target LR: {args.learning_rate:.2e} (bias groups will get 3x = {args.learning_rate*3:.2e})")
    
    # Determine number of epochs / total steps --------------------------------
    steps_per_epoch = len(loader)
    if args.total_steps is not None:
        total_steps_budget = args.total_steps
        num_epochs = math.ceil(total_steps_budget / steps_per_epoch)
        print(f"[config] total_steps={total_steps_budget} → computed epochs={num_epochs}")
    else:
        num_epochs = args.epochs
        total_steps_budget = steps_per_epoch * num_epochs
        print(f"[config] epochs={num_epochs} → total_steps={total_steps_budget}")

    # Set up learning rate scheduler
    if args.use_cosine_decay:
        # total_steps_budget computed above
        total_steps = total_steps_budget
        warmup_steps = args.warmup_steps
        
        # Cosine annealing scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, 
            T_max=total_steps - warmup_steps,
            eta_min=args.min_learning_rate
        )
        
        # Simple warmup function
        def get_warmup_lr(step):
            if step < warmup_steps:
                return args.learning_rate * (step / warmup_steps)
            return args.learning_rate
        
        print(f"Using cosine decay: warmup for {warmup_steps} steps, then decay to {args.min_learning_rate}")
    else:
        scheduler = None
    
    # autocast_dtype already determined above during MAE model setup
    
    # GradScaler only needed for fp16, not bf16
    if args.use_amp and args.amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2**10,      # Lower initial scale (1024 vs 65536)
            growth_factor=2.0,     # Conservative growth
            backoff_factor=0.5,    # Standard backoff
            growth_interval=2000   # Slower growth
        )
        print("Using GradScaler for fp16 training")
    else:
        scaler = None
        print(f"No GradScaler needed for {args.amp_dtype} training")

    # ─── wandb ────────────────────────────────────────────────────────
    wandb.init(project=args.project_name, name=args.run_name, config=args)
    # Define custom x-axis so every metric uses our step index
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    # ─── Best model tracking ──────────────────────────────────────────
    best_val_mse = float('inf')
    best_model_path = None
    if args.save_best_model:
        # Create checkpoints directory
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        best_model_path = checkpoint_dir / f"{args.run_name}.pt"

    global_step, samples = 0, 0
    loss_ma = deque(maxlen=100)
    current_l1 = args.l1_coeff  # fixed L1 coeff (following Anthropic)
    sparsity_history = 0.0      # accumulate frac_active for auto-tuning
    intervals_within_target = 0 # consecutive intervals inside target window (for freeze logic)
    high_loss_buffer = deque(maxlen=1000)  # Store high-loss tokens for resampling
    
    # Create separate CUDA stream for MAE forward pass overlap
    mae_stream = torch.cuda.Stream(device=device, priority=-1)

    for epoch in range(num_epochs):
        # Use CUDA prefetcher for overlap
        prefetcher = CUDAPrefetcher(loader, device)
        pbar = tqdm(prefetcher, desc=f"SAE E{epoch+1}/{num_epochs}", leave=False)
        
        # Pre-extract tokens for first batch to start pipeline
        vols_iter = iter(pbar)
        try:
            current_vols = next(vols_iter)
            current_vols = current_vols.to(dtype=model_dtype)
        except StopIteration:
            continue
            
        with torch.cuda.stream(mae_stream):
            current_tokens = token_extractor.extract_tokens(current_vols)
        
        for next_vols in vols_iter:
            next_vols = next_vols.to(dtype=model_dtype)
            if cancel["stop"] or global_step >= total_steps_budget:
                break
            global_step += 1
            
            # Start extracting tokens for next batch on separate stream while processing current
            with torch.cuda.stream(mae_stream):
                next_tokens = token_extractor.extract_tokens(next_vols)
            
            # Wait for current tokens to be ready (optimized stream synchronization)
            torch.cuda.current_stream().wait_stream(mae_stream)
            tokens = current_tokens  # Already on GPU with optimized transfer
            # tokens.shape: (batch_size * 1728, 768) for 96^3 volumes with 8^3 patches

            # Process all tokens that fit in H100 memory
            optim.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=autocast_dtype):
                if isinstance(sae, GatedSAE):
                    recon, z, gate = sae(tokens)
                else:
                    recon, z = sae(tokens)
                mse = F.mse_loss(recon, tokens)

                # ── orthogonality penalty on decoder rows ─────────────
                if args.ortho_coeff > 0.0:
                    W     = sae.decoder_weight             # (latent, input_dim)
                    gram  = W @ W.T                        # (latent, latent)
                    eye   = torch.eye(gram.size(0), device=W.device, dtype=W.dtype)
                    ortho = ((gram - eye) ** 2).sum() / (W.shape[0] ** 2)   # scale‑invariant
                else:
                    ortho = torch.tensor(0.0, device=mse.device, dtype=mse.dtype)
                # ────────────────────────────────────────────────────────────

                if args.sae_variant == "gated":
                    gate_l1 = gate.abs().sum(dim=1).mean()
                    loss = mse + current_l1 * gate_l1 + args.ortho_coeff * ortho
                else:
                    l1_penalty = z.abs().sum(dim=1).mean()
                    loss = mse + current_l1 * l1_penalty + args.ortho_coeff * ortho

            # Fail fast if loss is NaN or Inf
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {global_step}: {loss.item()}")

            # Store high-loss tokens for dead latent resampling
            with torch.no_grad():
                token_losses = F.mse_loss(recon.detach(), tokens.detach(), reduction='none').mean(dim=1)
                # Get top 10% highest loss tokens
                if len(token_losses) > 10:
                    k = max(1, len(token_losses) // 10)
                    high_loss_indices = torch.topk(token_losses, k).indices
                    high_loss_tokens = tokens[high_loss_indices].cpu()
                    high_loss_buffer.extend(high_loss_tokens)

            # Backward pass and optimization step
            if scaler is not None:
                # fp16 path: use GradScaler
                scaler.scale(loss).backward()
                if args.grad_clip_norm:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip_norm)
                # Apply gradient projection before optimizer step (Anthropic method)
                sae.apply_gradient_projection()
                scaler.step(optim)
                scaler.update()
            else:
                # bf16 path: direct backward (no scaling needed)
                loss.backward()
                if args.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip_norm)
                # Apply gradient projection before optimizer step (Anthropic method)
                sae.apply_gradient_projection()
                optim.step()
                
            # Update learning rate scheduler
            if scheduler is not None:
                if args.use_cosine_decay and global_step < warmup_steps:
                    # Manual warmup
                    warmup_lr = get_warmup_lr(global_step)
                    for i, param_group in enumerate(optim.param_groups):
                        if i % 2 == 0:  # weight parameters (even indices: encoder_weight, decoder_weight)
                            param_group['lr'] = warmup_lr
                        else:  # bias parameters (odd indices: encoder_bias, decoder_bias)
                            param_group['lr'] = warmup_lr * 3.0
                elif args.use_cosine_decay and global_step >= warmup_steps:
                    # Cosine decay after warmup
                    scheduler.step()
                    # Apply 3x multiplier to bias learning rates
                    base_lr = scheduler.get_last_lr()[0]
                    for i, param_group in enumerate(optim.param_groups):
                        if i % 2 == 1:  # bias parameters (odd indices)
                            param_group['lr'] = base_lr * 3.0

            # Post-optimization steps (Anthropic-style)
            with torch.no_grad():
                # 1. Normalize decoder weights (sparse dictionary learning)
                # sae.normalize_decoder_weights_proper()
                
                # 2. Update dead latent tracking
                sae.update_dead_latent_stats(z.detach())
                
                # 3. Dead latent resampling
                if global_step % args.resample_interval == 0 and len(high_loss_buffer) > 0:
                    high_loss_tensor = torch.stack(list(high_loss_buffer)).to(device)
                    n_resampled = sae.resample_dead_latents_anthropic(high_loss_tensor, dead_threshold=args.dead_threshold)
                    if n_resampled > 0:
                        print(f"Step {global_step}: Resampled {n_resampled} dead latents")
                        # Clear optimizer state for resampled parameters
                        # This is a simplified approach - ideally we'd reset specific parameter states
                        for group in optim.param_groups:
                            for p in group['params']:
                                if p is sae.encoder_weight or p is sae.encoder_bias or p is sae.decoder_weight:
                                    state = optim.state[p]
                                    if 'exp_avg' in state:
                                        state['exp_avg'][sae.steps_since_active > args.dead_threshold] = 0
                                    if 'exp_avg_sq' in state:
                                        state['exp_avg_sq'][sae.steps_since_active > args.dead_threshold] = 0

            # Store moving-average loss (like vol_train.py)
            loss_ma.append(float(loss.detach()))
            samples += tokens.shape[0]
            
            # Compute sparsity every step and accumulate for auto-tune
            if args.sae_variant=="gated":
                sparsity_step = float((z.detach().abs() < 1e-6).float().mean())
            else:
                sparsity_step = float((z.detach().abs() < 1e-6).float().mean())
            frac_active_step = 1.0 - sparsity_step
            sparsity_history += frac_active_step
            
            # Collect all metrics for this step in one place (exactly like vol_train.py)
            metrics = {}
            
            if global_step % args.train_log_interval == 0:
                # Calculate current metrics
                sparsity = sparsity_step
                frac_active = frac_active_step
                l1_val = z.detach().abs().sum(dim=1).mean() # Still compute for logging
                
                # Log token stats
                with torch.no_grad():
                    metrics["token_mean"] = tokens.mean().item()
                    metrics["token_std"] = tokens.std().item()
                    metrics["token_min"] = tokens.min().item()
                    metrics["token_max"] = tokens.max().item()
                
                # For gated SAE, log separate gate and magnitude sparsity
                if args.sae_variant=="gated":
                    gate_sparsity = float((gate.detach() == 0).float().mean())
                    mag_sparsity = float((z.detach() == 0).float().mean())  # After Top-K
                    metrics.update({
                        "gate_sparsity": gate_sparsity,
                        "mag_sparsity": mag_sparsity,
                        "ortho": float(ortho.detach()),
                    })
                
                metrics.update({
                    "train_loss": sum(loss_ma) / len(loss_ma) if loss_ma else 0.0,
                    "train_mse": float(mse.detach()),
                    "train_l1": float(l1_val), # Log the L1 value even if not in loss
                    "train_ortho": float(ortho.detach()),
                    "train_sparsity": sparsity,
                    "train_frac_active": frac_active,
                    "learning_rate": optim.param_groups[0]['lr'],
                    "samples_processed": samples,
                    "epoch": epoch + 1
                })

            # ─── L1 auto-tuning ──────────────────────────────────────────
            if (
                args.l1_auto_tune and
                (global_step % args.l1_tune_interval == 0) and
                (global_step > 0)
            ):
                avg_frac = sparsity_history / args.l1_tune_interval
                sparsity_history = 0.0

                # Determine adjustment
                if avg_frac < args.l1_target_low:
                    current_l1 /= args.l1_tune_multiplier  # allow more activity
                    intervals_within_target = 0
                elif avg_frac > args.l1_target_high:
                    current_l1 *= args.l1_tune_multiplier  # encourage sparsity
                    intervals_within_target = 0
                else:
                    # Within target window
                    intervals_within_target += 1

                # Freeze after enough stable intervals
                if intervals_within_target >= args.l1_freeze_after:
                    args.l1_auto_tune = False  # disable further tuning
                    print(f"[L1-tuner] Target maintained for {intervals_within_target} intervals — tuning frozen at λ={current_l1:.3e}")

                metrics["current_l1"] = current_l1

            if global_step % args.val_interval == 0:
                val_metrics = run_sae_validation(sae, mae, val_loader, args.layer, device, token_extractor, extract_from=args.extract_from, sae_variant=args.sae_variant)
                metrics.update(val_metrics)
                
                # Light cleanup without blocking (like vol_train.py)
                torch.cuda.empty_cache()
                
                # Save model if we got a new best validation MSE
                if args.save_best_model and val_metrics['val_mse'] < best_val_mse:
                    best_val_mse = val_metrics['val_mse']
                    print(f"New best validation MSE: {best_val_mse:.6f} (step {global_step}) - saving SAE...")
                    
                    # Save the SAE state
                    torch.save({
                        'sae_weight': sae.encoder_weight.detach().cpu(),
                        'encoder_bias': sae.encoder_bias.detach().cpu(),
                        'decoder_weight': sae.decoder_weight.detach().cpu(),
                        'decoder_bias': sae.decoder_bias.detach().cpu(),
                        'input_dim': C,
                        'latent_dim': latent_dim,
                        'layer': args.layer,
                        'l1_coeff': current_l1,
                        'global_step': global_step,
                        'val_mse': best_val_mse,
                        'val_sparsity': val_metrics['val_sparsity'],
                        'config': vars(args)
                    }, best_model_path)
                    
                    # Also log to wandb that we saved a new best model
                    metrics["best_val_mse"] = best_val_mse
                    metrics["saved_best_model"] = True

            # SAE visualization (like vis_interval in vol_train.py)
            if args.vis_interval and global_step % args.vis_interval == 0:
                with torch.no_grad():
                    # take one mini‑batch from the vis_loader
                    try:
                        vis_vols = next(iter(vis_loader))
                    except StopIteration:
                        vis_vols = next(iter(DataLoader(
                            TarShardDataset(val_shards[:1], args.img_size, shuffle=False),
                            batch_size=args.vis_samples)))
                    vis_vols = vis_vols.to(device, dtype=model_dtype)

                    # run full SAE‑patched reconstruction just once
                    recon_data = compute_reconstruction_mse_with_sae(
                        mae, vis_vols, args.layer, sae,
                        device, token_extractor, extract_from=args.extract_from,
                        mask_ratio=0.85
                    )

                    # grid of SAE‑patched outputs
                    grid_pngs = visualize_sae_slice_grid(
                        mae, sae, vis_vols, token_extractor,
                        layer_idx=args.layer,
                        step=global_step
                    )
                    for p in grid_pngs:
                        wandb.log({"sae_slice_grid": wandb.Image(str(p))},
                                  step=global_step, commit=False)

                wandb.log({}, step=global_step, commit=True)   # flush

                    
            # Log dead latent statistics periodically
            if global_step % args.val_interval == 0:
                with torch.no_grad():
                    n_dead = (sae.steps_since_active > args.dead_threshold).sum().item()
                    pct_dead = 100.0 * n_dead / sae.encoder_weight.shape[0]
                    metrics["dead_latents"] = n_dead
                    metrics["dead_latents_pct"] = pct_dead
            
            # Log metrics (if any) - exactly like vol_train.py
            if metrics:
                wandb.log(metrics, step=global_step)

            if global_step % 50 == 0:
                pbar.set_postfix(
                    loss=(sum(loss_ma) / len(loss_ma) if loss_ma else 0.0),
                    lr=f"{optim.param_groups[0]['lr']:.2e}",
                    sparsity=f"{sparsity:.3f}" if 'sparsity' in locals() else "N/A"
                )
            
            # Advance pipeline for next iteration
            current_tokens = next_tokens
            current_vols = next_vols
        
        if cancel["stop"] or global_step >= total_steps_budget:
            break
        
        # Process the final batch
        if 'current_tokens' in locals():
            global_step += 1
            
            # Wait for final tokens to be ready
            torch.cuda.current_stream().wait_stream(mae_stream)
            tokens = current_tokens
            
            # Process final batch (same logic as above but without next batch prep)
            optim.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=autocast_dtype):
                if isinstance(sae, GatedSAE):
                    recon, z, gate = sae(tokens)
                else:
                    recon, z = sae(tokens)
                mse = F.mse_loss(recon, tokens)
                if isinstance(sae, GatedSAE):
                    gate_l1 = gate.abs().sum(dim=1).mean()
                    loss = mse + current_l1*gate_l1
                else:
                    l1_penalty = z.abs().sum(dim=1).mean()
                    loss = mse + current_l1 * l1_penalty

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {global_step}: {loss.item()}")

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip_norm:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip_norm)
                # Apply gradient projection before optimizer step (Anthropic method)
                sae.apply_gradient_projection()
                optim.step()
                
            # Update learning rate scheduler (for final batch)
            if scheduler is not None:
                if args.use_cosine_decay and global_step < warmup_steps:
                    # Manual warmup
                    warmup_lr = get_warmup_lr(global_step)
                    for i, param_group in enumerate(optim.param_groups):
                        if i % 2 == 0:  # weight parameters (even indices: encoder_weight, decoder_weight)
                            param_group['lr'] = warmup_lr
                        else:  # bias parameters (odd indices: encoder_bias, decoder_bias)
                            param_group['lr'] = warmup_lr * 3.0
                elif args.use_cosine_decay and global_step >= warmup_steps:
                    # Cosine decay after warmup
                    scheduler.step()
                    # Apply 3x multiplier to bias learning rates
                    base_lr = scheduler.get_last_lr()[0]
                    for i, param_group in enumerate(optim.param_groups):
                        if i % 2 == 1:  # bias parameters (odd indices)
                            param_group['lr'] = base_lr * 3.0

            # Post-optimization steps
            with torch.no_grad():
                # sae.normalize_decoder_weights_proper()
                sae.update_dead_latent_stats(z.detach())

            loss_ma.append(float(loss.detach()))
            samples += tokens.shape[0]



    # Cleanup token extractor
    token_extractor.cleanup()
    
    wandb.finish()
    print("SAE training complete.")

    # ─── save final model ──────────────────────────────────────────────────────
    out = {
        "sae_weight": sae.encoder_weight.detach().cpu(),
        "encoder_bias": sae.encoder_bias.detach().cpu(),
        "decoder_weight": sae.decoder_weight.detach().cpu(),
        "decoder_bias": sae.decoder_bias.detach().cpu(),
        "input_dim": C,
        "latent_dim": latent_dim,
        "layer": args.layer,
        "l1_coeff": current_l1,  # Save the final tuned L1 coefficient
        "config": vars(args),
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved final SAE to {out_path}")
    
    if args.save_best_model and best_model_path and best_model_path.exists():
        print(f"Best model saved to {best_model_path} with validation MSE: {best_val_mse:.6f}")

# ═════════════════════════ argparse ═══════════════════════════════════
if __name__ == "__main__":
    P = argparse.ArgumentParser("Sparse Auto-Encoder trainer for ViT-MAE-3D activations")
    # Data / MAE model
    P.add_argument("--shard_dir", required=True, help="Directory with shard_XXXXX.tar files")
    P.add_argument("--checkpoint", required=True, help="Pre-trained MAE checkpoint (.pt)")
    P.add_argument("--model_arch", default="base", choices=["small", "base", "large", "base_conv", "hemibrain_optimal", "base_patch_conv"],
                   help="Which MAE architecture to instantiate (must match checkpoint)")
    # EMA config reuse – needed for kwargs
    P.add_argument("--img_size", type=int, default=96)
    P.add_argument("--patch_size", type=int, default=8)
    P.add_argument("--initial_masking_ratio", type=float, default=0.0, help="Keep all patches during SAE training")

    # SAE specifics
    P.add_argument("--layer", type=int, default=6, help="Encoder block index (0-based) to extract activations from (ignored if extract_from='patchembed')")
    P.add_argument("--extract_from", choices=["patchembed", "encoder"], default="encoder", 
                   help="Where to extract features: 'patchembed' for raw CNN features, 'encoder' for transformer features")
    P.add_argument("--latent_dim", type=int, default=None, help="Explicit latent size (overrides multiplier)")
    P.add_argument("--latent_dim_multiplier", type=int, default=3,
                   help="If --latent_dim not set, use multiplier × input_dim (default 3×)")
    P.add_argument("--l1_coeff", type=float, default=8e-3, help="Fixed L1 coefficient for sparsity (following Anthropic's approach)")
    P.add_argument("--ortho_coeff", type=float, default=1e-3, help="weight for decoder orthogonality penalty (0→ disables)")

    P.add_argument("--l1_tune_interval", type=int, default=500, help="Steps between auto-tuning L1 coeff")
    P.add_argument("--dead_threshold", type=int, default=20000, help="Steps before a latent is considered dead")
    P.add_argument("--resample_interval", type=int, default=25000, help="Steps between dead latent resampling")
    P.add_argument("--activation", choices=["relu", "gelu"], default="relu", 
                   help="Activation function for SAE encoder (relu gives true zeros for sparsity)")

    # Optimisation
    P.add_argument("--batch_size", type=int, default=128, help="Number of *volumes* per MAE forward pass")
    P.add_argument("--learning_rate", type=float, default=1e-3)
    P.add_argument("--epochs", type=int, default=10)
    P.add_argument("--num_workers", type=int, default=16)
    P.add_argument("--prefetch_factor", type=int, default=4)
    P.add_argument("--log_interval", type=int, default=50)  # Legacy, kept for compatibility
    P.add_argument("--train_log_interval", type=int, default=10, help="Steps between training metric logs")
    P.add_argument("--val_interval", type=int, default=500, help="Steps between validation runs")
    P.add_argument("--val_split", type=float, default=0.02, help="Fraction of shards for validation")
    P.add_argument("--use_amp", action="store_true", default=True)
    P.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16",
                   help="Autocast dtype to use when --use_amp is enabled (bf16 recommended for H100).")
    P.add_argument("--grad_clip_norm", type=float, default=1.0)
    
    # Learning rate scheduling
    P.add_argument("--use_cosine_decay", action="store_true", 
                   help="Use cosine annealing LR decay with warmup")
    P.add_argument("--warmup_steps", type=int, default=500,
                   help="Number of warmup steps for cosine decay")
    P.add_argument("--min_learning_rate", type=float, default=1e-4,
                   help="Minimum learning rate for cosine decay")

    # Wandb logging
    P.add_argument("--project_name", default="sae-3d-activations", help="Wandb project name")
    P.add_argument("--run_name", default="sae_layer6", help="Wandb run name")

    # Output
    P.add_argument("--output_path", default="checkpoints/sae_layer6.pt")
    P.add_argument("--save_best_model", action="store_true",
                   help="Save the SAE when validation MSE improves")

    # ── L1 sparsity auto-tuning ────────────────────────────────────────────
    P.add_argument("--l1_auto_tune", action="store_true", help="Enable automatic L1 coefficient tuning")
    P.add_argument("--l1_target_low", type=float, default=0.01, help="Lower bound for desired fraction of active latents")
    P.add_argument("--l1_target_high", type=float, default=0.02, help="Upper bound for desired fraction of active latents")
    P.add_argument("--l1_tune_multiplier", type=float, default=1.5, help="Factor to multiply/divide L1 when out of target window")
    P.add_argument("--l1_freeze_after", type=int, default=6, help="Stop tuning after this many consecutive intervals are within target window")

    # k-sparse (top-k) masking
    P.add_argument("--k_sparse", type=int, default=None, help="Number of strongest activations to keep per token")
    # Compile flag
    P.add_argument("--compile_sae", action="store_true", help="Compile SAE with torch.compile (may increase memory usage)")
    P.add_argument("--compile_mae", action="store_true", help="Compile MAE with torch.compile (may increase memory usage)")

    # argparse additions
    P.add_argument("--total_steps", type=int, default=None, help="Total optimiser steps to run (overrides --epochs)")
    P.add_argument("--sae_variant", choices=["linear","gated"], default="linear", help="SAE architecture to use")
    P.add_argument("--vis_interval", type=int, default=500, help="Steps between SAE reconstruction visualizations")
    P.add_argument("--vis_samples", type=int, default=3, help="Number of samples to visualize for each reconstruction")

    args = P.parse_args()
    train_sae(args) 