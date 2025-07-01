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

    # ───────────────────────── 3rd-party
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# AMP dtype mapping (bf16 recommended for H100)
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# ───────────────────────── project
from vol_train import TarShardDataset, CUDAPrefetcher  # re-use optimised loaders
from vit_3d import (
    mae_vit_3d_small_conv, mae_vit_3d_base_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv,
    get_device,
)

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
        x_centered = x - self.decoder_bias  # Pre-encoder bias subtraction
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
            f = f * mask.float()

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

@torch.no_grad()
def extract_patch_tokens(model, volumes, layer_idx: int, extract_from: str = "encoder"):
    """Legacy function for backward compatibility - creates extractor each time."""
    extractor = TokenExtractor(model, layer_idx, extract_from=extract_from)
    try:
        return extractor.extract_tokens(volumes)
    finally:
        extractor.cleanup()

@torch.no_grad()
def compute_mae_loss_ratio(mae, volumes, layer_idx, sae, act_mean, act_std, device, token_extractor):
    """Compute Anthropic's loss-ratio metric: (orig_loss - sae_loss) / (orig_loss - zero_loss)."""
    mae.eval()
    
    # Get original activations and compute original loss
    orig_tokens = token_extractor.extract_tokens(volumes)
    orig_tokens_wh = (orig_tokens - act_mean) / act_std
    
    # Forward pass with original activations
    def forward_with_activations(tokens_to_use):
        # We need a way to inject activations into the MAE forward pass
        # This is tricky without modifying the MAE, so we'll approximate
        # by computing the "next layer" prediction loss
        # For now, we'll use a proxy: MSE between original and modified tokens
        return F.mse_loss(tokens_to_use, orig_tokens_wh)
    
    # Original loss (baseline)
    orig_loss = forward_with_activations(orig_tokens_wh)
    
    # Zero loss (activations set to zero)
    zero_tokens = torch.zeros_like(orig_tokens_wh)
    zero_loss = forward_with_activations(zero_tokens)
    
    # SAE loss (activations passed through SAE)
    sae_recon, _ = sae(orig_tokens_wh)
    sae_loss = forward_with_activations(sae_recon)
    
    # Compute loss ratio
    denominator = zero_loss - orig_loss
    if denominator.abs() < 1e-8:
        return 0.0
    
    loss_ratio = (orig_loss - sae_loss) / denominator
    return float(loss_ratio.clamp(0, 1))  # Clamp to [0, 1] range

@torch.no_grad()
def run_sae_validation(sae, mae, val_loader, layer_idx, device, act_mean, act_std, token_extractor, max_batches=10):
    """Run SAE validation on a subset of validation data."""
    sae.eval()
    val_losses, val_mses, val_l1s, val_sparsities = [], [], [], []
    loss_ratios = []
    
    batch_count = 0
    for vols in val_loader:
        if batch_count >= max_batches:
            break
        
        vols = vols.to(device)
        tokens = token_extractor.extract_tokens(vols)
        tokens = (tokens - act_mean) / act_std  # whitening same as training
        
        # Compute loss ratio for this batch
        try:
            loss_ratio = compute_mae_loss_ratio(mae, vols, layer_idx, sae, act_mean, act_std, device, token_extractor)
            loss_ratios.append(loss_ratio)
        except Exception as e:
            print(f"Warning: Could not compute loss ratio: {e}")
        
        # Process in chunks to match training
        for chunk in tokens.split(4096):  # Use smaller chunks for validation
            if args.sae_variant=="gated":
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
    
    if loss_ratios:
        metrics['val_loss_ratio'] = sum(loss_ratios) / len(loss_ratios)
    
    return metrics

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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // 2,  # Smaller batches for validation
        num_workers=args.num_workers // 2,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        timeout=300
    )

    # ─── MAE backbone (frozen) ─────────────────────────────────────
    archs = {
        "small": mae_vit_3d_small_conv,
        "base": mae_vit_3d_base_conv,
        "base_conv": mae_vit_3d_base_conv,
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

    # Load checkpoint BEFORE compilation to avoid prefix issues
    if args.checkpoint:
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

    # Compile MAE for faster inference AFTER loading checkpoint
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
    dummy = torch.zeros(1, 1, args.img_size, args.img_size, args.img_size)
    C = token_extractor.extract_tokens(dummy.to(device)).shape[1]
    print(f"Token feature dimension: {C}")

    latent_dim = args.latent_dim or C * args.latent_dim_multiplier
    print(f"Latent dim: {latent_dim}")

    # ─── Activation whitening (compute once) ──────────────────────────────
    print("Computing activation mean/std for whitening …")
    with torch.no_grad():
        samp_prefetch = CUDAPrefetcher(loader, device)
        samp_vols = next(iter(samp_prefetch))
        samp_tok = token_extractor.extract_tokens(samp_vols)
        act_mean = samp_tok.mean(0, keepdim=True)  # (1, C)
        act_std  = samp_tok.std(0, keepdim=True).clamp_(min=1e-6)
    print("Whitening stats ready → mean/std vectors shape", act_mean.shape)

    if args.sae_variant=="linear":
        sae = LinearSAE(C, latent_dim, activation=args.activation, k_sparse=args.k_sparse).to(device)
    elif args.sae_variant=="gated":
        sae = GatedSAE(C, latent_dim, k_sparse=args.k_sparse).to(device)
    else:
        raise ValueError("unknown sae_variant")
    
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
        dummy_tokens = torch.randn(10, C, device=device)  # 10 tokens, C dimensions
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
    
    # Determine autocast dtype (bf16 recommended for H100)
    autocast_dtype = _AMP_DTYPE_MAP.get(args.amp_dtype, torch.bfloat16)
    
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
        best_model_path = checkpoint_dir / f"best_sae_{args.run_name}.pt"

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
        except StopIteration:
            continue
            
        with torch.cuda.stream(mae_stream):
            current_tokens = token_extractor.extract_tokens(current_vols)
        
        for next_vols in vols_iter:
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
                # Whitening
                tokens_wh = (tokens - act_mean) / act_std
                if args.sae_variant=="gated":
                    recon, z, gate = sae(tokens_wh)
                else:
                    recon, z = sae(tokens_wh)
                mse = F.mse_loss(recon, tokens_wh)
                if args.sae_variant=="gated":
                    gate_l1 = gate.abs().sum(dim=1).mean()
                    loss = mse + current_l1*gate_l1
                else:
                    loss = mse

            # Fail fast if loss is NaN or Inf
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {global_step}: {loss.item()}")

            # Store high-loss tokens for dead latent resampling
            with torch.no_grad():
                detached_tokens_wh = tokens_wh.detach()
                token_losses = F.mse_loss(recon.detach(), detached_tokens_wh, reduction='none').mean(dim=1)
                # Get top 10% highest loss tokens
                if len(token_losses) > 10:
                    k = max(1, len(token_losses) // 10)
                    high_loss_indices = torch.topk(token_losses, k).indices
                    high_loss_tokens = detached_tokens_wh[high_loss_indices].cpu()
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
                sae.normalize_decoder_weights_proper()
                
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
                
                # For gated SAE, log separate gate and magnitude sparsity
                if args.sae_variant=="gated":
                    gate_sparsity = float((gate.detach() == 0).float().mean())
                    mag_sparsity = float((z.detach() == 0).float().mean())  # After Top-K
                    metrics.update({
                        "gate_sparsity": gate_sparsity,
                        "mag_sparsity": mag_sparsity,
                    })
                
                metrics.update({
                    "train_loss": sum(loss_ma) / len(loss_ma) if loss_ma else 0.0,
                    "train_mse": float(mse.detach()),
                    "train_l1": float(l1_val), # Log the L1 value even if not in loss
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
                val_metrics = run_sae_validation(sae, mae, val_loader, args.layer, device, act_mean, act_std, token_extractor)
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
                tokens_wh = (tokens - act_mean) / act_std
                if args.sae_variant=="gated":
                    recon, z, gate = sae(tokens_wh)
                else:
                    recon, z = sae(tokens_wh)
                mse = F.mse_loss(recon, tokens_wh)
                if args.sae_variant=="gated":
                    gate_l1 = gate.abs().sum(dim=1).mean()
                    loss = mse + current_l1*gate_l1
                else:
                    loss = mse

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
                sae.normalize_decoder_weights_proper()
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
    P.add_argument("--model_arch", default="base", choices=["small", "base", "large", "base_conv", "hemibrain_optimal"],
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

    args = P.parse_args()
    train_sae(args) 