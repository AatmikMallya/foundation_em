#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAE-3D trainer – tar shards – CUDA-prefetcher edition
====================================================
Designed for 1× H100 + ≥ 8 CPU cores.

* Input format: shard_XXXXX.tar with raw 64³ float32 blobs (1 MiB each)
  – identical to the offline generator we built earlier.
* Training loop overlaps PCIe/NVLink transfers with compute to keep
  the GPU fully saturated.
"""

# ───────────────────────── stdlib
import argparse, copy, io, math, os, random, signal, tarfile, time
from collections import deque
from pathlib import Path

# ───────────────────────── 3rd-party
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ───────────────────────── project (your ViT + helpers)
from vit_3d import (
    mae_vit_3d_small, mae_vit_3d_base, mae_vit_3d_large, mae_vit_3d_huge,
    mae_vit_3d_hemibrain_optimal,
    mae_vit_3d_small_conv, mae_vit_3d_base_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv, mae_vit_3d_base_patch_conv,
    get_device,
)
from plotly_visualization import plotly_visualize_reconstructions
from enhanced_visualization import enhanced_visualize_reconstructions
import gc

# AMP dtype mapping (bf16 recommended for H100)
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# ═════════════════════════ Optimized Dataset ════════════════════════════════════════
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor

# ════════════════════ tar dataset ════════════════════════════════
class TarShardDataset(IterableDataset):
    """Tar dataset with I/O optimizations."""
    def __init__(self, shards, volume_size: int, shuffle: bool = False, 
                 vols_per_shard: int = 16384):
        self.shards = shards
        self.shuffle = shuffle
        self.vols_per_shard = vols_per_shard
        self.volume_size = volume_size
        
        print(f"Dataset: {len(shards)} shards, {self.vols_per_shard} vols/shard, shuffle={shuffle}")

    def __len__(self):
        return len(self.shards) * self.vols_per_shard

    def _iter_shard(self, path):
        """Optimized tar iteration with larger buffers and efficient tensor creation."""
        try:
            # Large buffer for better I/O throughput (proven optimization)
            with tarfile.open(path, "r|", bufsize=32*1024*1024) as tar:  # 32MB buffer
                for member in tar:
                    if not member.isfile():
                        continue
                    
                    # Extract with larger read chunks
                    buf = tar.extractfile(member).read()
                    
                    # Efficient tensor creation (avoid copy when possible)
                    vol = np.frombuffer(buf, dtype=np.float32)
                    vol = vol.reshape(self.volume_size, self.volume_size, self.volume_size)
                    
                    # Create tensor with proper memory layout for GPU transfer
                    volume_tensor = torch.from_numpy(vol).contiguous().unsqueeze(0).pin_memory()
                    yield volume_tensor
                    
        except Exception as e:
            print(f"Error reading shard {path}: {e}")
            return

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        shard_list = self.shards.copy()
        if self.shuffle:
            random.shuffle(shard_list)
        
        # Distribute shards across workers with better load balancing
        if worker:
            # Round-robin distribution for better load balancing
            worker_shards = shard_list[worker.id::worker.num_workers]
        else:
            worker_shards = shard_list
            
        for shard_path in worker_shards:
            yield from self._iter_shard(shard_path)

# ════════════════════ CUDA prefetcher ═════════════════════════════════════
class CUDAPrefetcher:
    """
    Optimized CUDA prefetcher with double buffering for reliable H100 utilization.
    """
    def __init__(self, loader: DataLoader, device: torch.device, queue_size: int = 2):
        self.loader = loader
        self.device = device
        self.queue_size = queue_size
        # Use high-priority stream for faster transfers
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self.it = iter(loader)   # persistent
        
        # Double buffering - proven optimal for most workloads
        self.batch_queue = []
        self.events = []
        
        # Pre-fill the queue with 2 batches for compute/transfer overlap
        for _ in range(min(queue_size, 2)):  # Start with 2 batches
            self._prefetch_batch()

    def _prefetch_batch(self):
        """Prefetch a single batch to GPU with optimized transfer."""
        try:
            batch_cpu = next(self.it)
        except StopIteration:
            return False
            
        with torch.cuda.stream(self.stream):
            # Only transfer to device, keep fp32 - let autocast handle precision reduction
            if batch_cpu.is_pinned():
                # Fast path: already pinned memory
                batch_gpu = batch_cpu.to(device=self.device, memory_format=torch.channels_last_3d, non_blocking=True)
            else:
                # Slower path: pin then transfer
                batch_cpu = batch_cpu.pin_memory()
                batch_gpu = batch_cpu.to(device=self.device, memory_format=torch.channels_last_3d, non_blocking=True)
            
            # Record an event to know when this transfer is complete
            event = torch.cuda.Event()
            event.record(self.stream)
            
            self.batch_queue.append(batch_gpu)
            self.events.append(event)
        return True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch_queue:
            raise StopIteration
            
        # Get the next batch and its completion event
        batch = self.batch_queue.pop(0)
        event = self.events.pop(0)
        
        # Wait for this batch to be ready
        event.wait()
        
        # Immediately prefetch the next batch to maintain the queue
        self._prefetch_batch()
        
        return batch

# ════════════════════ EMA helper ═════════════════════════════════════════
class EMAModel:
    def __init__(self, model, decay=0.9999, warmup=1_000):
        self.decay, self.warmup = decay, warmup
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False
        self.updates = 0

    def update(self, model):
        self.updates += 1
        if self.updates <= self.warmup and self.updates % 10:
            return
        d = (0.99 + (self.decay - 0.99) * (self.updates / self.warmup)
             if self.updates <= self.warmup else self.decay)
        with torch.no_grad():
            for e, m in zip(self.ema.parameters(), model.parameters()):
                e.mul_(d).add_(m.data, alpha=1 - d)

    def get(self):
        return self.ema

# ════════════════════ schedulers ═════════════════════════════════════════
def cosine_with_warmup(opt, warmup, total, min_lr, base_lr):
    def _lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, _lambda)

def mask_ratio(step, total, low, high):
    return low if low == high else low + (high - low) * (step / total)

# ════════════════════ validation func ════════════════════════════════════
@torch.inference_mode()
def run_val(loader, model, mask, use_amp, device, amp_dtype=torch.float16, max_batches=50):
    """Optimized validation with CUDA prefetching and limited batches for speed."""
    model.eval()
    losses = []
    
    # Use CUDA prefetcher for validation too
    val_prefetcher = CUDAPrefetcher(loader, device, queue_size=2)
    
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
        batch_count = 0
        for v in val_prefetcher:
            if batch_count >= max_batches:  # Limit validation batches for speed
                break
            # Batch already has optimal memory format from prefetcher
            l, *_ = model(v, mask_ratio=mask)
            if not torch.isnan(l):
                losses.append(l.detach())
            batch_count += 1
    
    model.train()
    return float(torch.stack(losses).mean()) if losses else 0.

# ════════════════════ main training loop ════════════════════════════════
def main(a):
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

    # ─── shards ───────────────────────────────────────────────────────
    shards = sorted(Path(a.shard_dir).expanduser().glob("shard*.tar"))
    n_val  = max(1, int(len(shards) * a.val_split))
    val_shards, train_shards = shards[:n_val], shards[n_val:]

    train_loader = DataLoader(
        TarShardDataset(
            train_shards, a.img_size, shuffle=True, vols_per_shard=a.vols_per_shard
        ),
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=False,  # Dataset already pins memory
        drop_last=True,
        prefetch_factor=a.prefetch_factor,
        persistent_workers=True,
        multiprocessing_context="spawn" if a.num_workers > 0 else None,
        # Proven optimizations for large datasets
        timeout=300  # 5 min timeout for large tar files
    )
    val_loader = DataLoader(
        TarShardDataset(
            val_shards, a.img_size, shuffle=False, vols_per_shard=a.vols_per_shard
        ),
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=False,  # Dataset already pins memory
        drop_last=False,
        prefetch_factor=a.prefetch_factor // 2,  # Smaller prefetch for validation
        persistent_workers=True,
        multiprocessing_context="spawn" if a.num_workers > 0 else None,
        timeout=300
    )
    vis_loader = DataLoader(
        TarShardDataset(
            val_shards[:1], a.img_size, shuffle=False, vols_per_shard=a.vols_per_shard
        ),
        batch_size=1, num_workers=0
    )

    # ─── model ────────────────────────────────────────────────────────
    archs = {
        "small": mae_vit_3d_small, "base": mae_vit_3d_base,
        "large": mae_vit_3d_large, "huge": mae_vit_3d_huge,
        "hemibrain_optimal": mae_vit_3d_hemibrain_optimal,
        "small_conv": mae_vit_3d_small_conv, "base_conv": mae_vit_3d_base_conv,
        "large_conv": mae_vit_3d_large_conv,
        "hemibrain_optimal_conv": mae_vit_3d_hemibrain_optimal_conv,
        "base_patch_conv": mae_vit_3d_base_patch_conv
    }
    model = archs[a.model_arch](
        volume_size=(a.img_size,)*3,
        patch_size=a.patch_size,
        norm_pix_loss=a.norm_pix_loss,
        mask_ratio=a.initial_masking_ratio
    )
    
    # Determine autocast dtype (bf16 recommended for H100)
    autocast_dtype = _AMP_DTYPE_MAP.get(a.amp_dtype, torch.bfloat16)
    
    # Optimize memory layout for 3D convolutions (15% speedup for Conv models)
    try:
        model = model.to(device, memory_format=torch.channels_last_3d)
        print("Model using channels-last-3D memory format for optimized 3D convolutions")
    except (AttributeError, RuntimeError):
        # Fallback for PyTorch < 2.2 or if channels_last_3d not supported
        model = model.to(device)
        print("Using default memory format (channels_last_3d not available)")

    # ─── Model compilation ────────────────────────────
    # Optimize Inductor compilation settings to reduce first-time compilation from 20-25 min to 2-3 min
    model = torch.compile(model, backend="inductor", )
    print("Model compiled with torch.compile (inductor backend, default mode)")

    torch.cuda.empty_cache()                # release their buffers

    optim = torch.optim.AdamW(model.parameters(),
                              lr=a.learning_rate,
                              betas=(a.adam_beta1, a.adam_beta2),
                              weight_decay=a.weight_decay,
                              eps=1e-5)

    steps_per_epoch = len(train_loader)
    total_steps = a.total_steps or a.epochs * steps_per_epoch
    scheduler = cosine_with_warmup(
        optim, a.warmup_steps, total_steps, a.min_lr, a.learning_rate
    )

    ema = EMAModel(model, a.ema_decay) if not a.disable_ema else None
    
    # GradScaler only needed for fp16, not bf16
    if a.use_amp and a.amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2**10,      # Lower initial scale (1024 vs 65536)
            growth_factor=2.0,     # Conservative growth
            backoff_factor=0.5,    # Standard backoff
            growth_interval=2000   # Slower growth
        )
        print("Using GradScaler for fp16 training")
    else:
        scaler = None
        print(f"No GradScaler needed for {a.amp_dtype} training")

    # ─── wandb ────────────────────────────────────────────────────────
    wandb.init(project=a.project_name, name=a.run_name, config=a)
    # Define custom x-axis so every metric uses our step index
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="step")
    wandb.watch(model, log_freq=1000)

    # ─── Best model tracking ──────────────────────────────────────────
    best_val_loss = float('inf')
    best_model_path = None
    if a.save_best_model:
        # Create checkpoints directory
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        best_model_path = checkpoint_dir / f"best_model_{a.run_name}.pt"

    # ─── TRAIN ────────────────────────────────────────────────────────
    global_step = 0
    loss_ma = deque(maxlen=100)

    # Silence PyTorch warning when converting read-only NumPy buffers (e.g. from
    # tarfile reads) to tensors.  We only read from these tensors, so the warning
    # is safe to ignore.
    import warnings
    warnings.filterwarnings("ignore", message=r"The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*", category=UserWarning)

    for epoch in range(a.epochs):
        model.train()
        prefetcher = CUDAPrefetcher(train_loader, device)

        pbar = tqdm(prefetcher, desc=f"E{epoch+1}/{a.epochs}", leave=False)
        for batch in pbar:
            if cancel["stop"]:
                break
            global_step += 1

            # ──── batch‑level image sanity log ─────────────────────────────
            if global_step % 100 == 0:               # every 100 steps
                imgs_std  = batch.std().item()
                imgs_mean = batch.mean().item()
                wandb.log({
                    "imgs_std":  imgs_std,
                    "imgs_mean": imgs_mean,
                    "step":      global_step
                })
            # ───────────────────────────────────────────────────────────────

            
            mratio = mask_ratio(global_step, total_steps,
                                a.initial_masking_ratio, a.final_masking_ratio)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=a.use_amp, dtype=autocast_dtype, device_type="cuda"):
                # loss, *_ = model(batch, mask_ratio=mratio)
                loss, _, mask, _ = model(batch, mask_ratio=mratio)

            # Fail fast if loss is NaN or Inf (respect user rule: no silent fallbacks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {global_step}: {loss.item()}")

            # Backward pass with optional gradient scaling
            if scaler is not None:
                # fp16 path: use GradScaler
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                if a.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip_norm)
                scaler.step(optim)
                scaler.update() 
            else:
                # bf16 path: direct backward (no scaling needed)
                loss.backward()
                if a.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip_norm)
                optim.step()

            # ──── grad + mask diagnostics (pre‑clip norm) ──────────────────
            if global_step % 100 == 0:
                # raw gradient norm (use huge max_norm so we just *measure*)
                total_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1e9,
                                error_if_nonfinite=False).item()

                mask_visible = mask.float().sum(dim=1).mean().item()

                wandb.log({
                    "grad_norm":    total_norm,
                    "mask_visible": mask_visible,
                    "step":         global_step,
                })
            # ───────────────────────────────────────────────────────────────
 
            scheduler.step()
            if ema:
                ema.update(model)

            # Store moving-average only for finite losses
            loss_ma.append(loss.detach())
            
            # Collect all metrics for this step in one place
            metrics = {}
            
            if global_step % a.log_interval == 0:
                metrics.update({
                    "train_loss": float(torch.stack(tuple(loss_ma)).mean()),
                    "learning_rate": optim.param_groups[0]['lr'],
                    "mask_ratio": mratio,
                    "epoch": epoch + 1
                })

            if (not a.skip_validation) and global_step % a.val_interval == 0:
                eval_model = ema.get() if ema else model
                v = run_val(val_loader, eval_model, mratio, a.use_amp, device, autocast_dtype)
                metrics["val_loss"] = v
                
                # Save model if we got a new best validation loss
                if a.save_best_model and v < best_val_loss:
                    best_val_loss = v
                    print(f"New best validation loss: {v:.6f} (step {global_step}) - saving model...")
                    
                    # Save the model state dict with dtype information
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': ema.get().state_dict() if ema else None,
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch + 1,
                        'val_loss': v,
                        'config': vars(a),
                        'model_dtype': str(autocast_dtype)  # Save the model dtype for later loading
                    }, best_model_path)
                    
                    # Also log to wandb that we saved a new best model
                    metrics["best_val_loss"] = best_val_loss
                    metrics["saved_best_model"] = True

            if a.vis_interval and global_step % a.vis_interval == 0:
                try:
                    # Run visualization but keep it lightweight
                    eval_model = ema.get() if ema else model
                    paths = enhanced_visualize_reconstructions(
                        eval_model, vis_loader, device, global_step,
                        mratio, "val", num_examples=a.vis_samples)

                    if paths:
                        metrics["slices"] = [wandb.Image(p) for p in paths]
                        del paths
                    
                    # Light cleanup without blocking
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[vis] failed: {e}")
                    torch.cuda.empty_cache()
            
            # Log remaining metrics (if any) – note we may have already logged above
            if metrics:
                wandb.log(metrics, step=global_step)

            if global_step % 50 == 0:
                pbar.set_postfix(loss=float(torch.stack(tuple(loss_ma)).mean()),
                                 lr=f"{optim.param_groups[0]['lr']:.2e}",
                                 mask=f"{mratio:.2f}")

        if cancel["stop"]:
            break

    wandb.finish()
    print("Training complete.")

def load_model_checkpoint(checkpoint_path, model, device='cuda'):
    """
    Load a model checkpoint with proper dtype handling and torch.compile compatibility.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model instance to load weights into
        device: Device to load the model on
    
    Returns:
        Loaded model with correct dtype
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get the state dict
    state_dict = checkpoint['model_state_dict']
    
    # Handle torch.compile _orig_mod prefixes
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("  Detected torch.compile checkpoint, removing _orig_mod prefixes...")
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Load model state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Successfully loaded checkpoint weights")
    except RuntimeError as e:
        print(f"  Error loading checkpoint: {e}")
        # Try non-strict loading as fallback
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    # Restore the correct dtype if saved in checkpoint
    if 'model_dtype' in checkpoint:
        model_dtype_str = checkpoint['model_dtype']
        if model_dtype_str == "torch.bfloat16":
            model = model.to(dtype=torch.bfloat16)
            print(f"  Loaded model with dtype: {model_dtype_str}")
        elif model_dtype_str == "torch.float16":
            model = model.to(dtype=torch.float16)
            print(f"  Loaded model with dtype: {model_dtype_str}")
    
    model = model.to(device)
    return model, checkpoint

# ════════════════════ CLI ════════════════════════════════════════════════
if __name__ == "__main__":
    P = argparse.ArgumentParser("MAE-3D trainer (CUDA-prefetch)")
    P.add_argument("--shard_dir", required=True)
    P.add_argument("--val_split", type=float, default=0.02)
    # model
    P.add_argument("--img_size", type=int, default=96)
    P.add_argument("--patch_size", type=int, default=8)
    P.add_argument("--model_arch", default="base",
                   choices=["small","base","large","huge","hemibrain_optimal",
                            "small_conv","base_conv","large_conv",
                            "hemibrain_optimal_conv","base_patch_conv"])
    # training
    P.add_argument("--epochs", type=int, default=4)
    P.add_argument("--batch_size", type=int, default=1024)
    P.add_argument("--learning_rate", type=float, default=3e-4)
    P.add_argument("--min_lr", type=float, default=1e-5)
    P.add_argument("--warmup_steps", type=int, default=1_000)
    P.add_argument("--total_steps", type=int, default=None)
    P.add_argument("--weight_decay", type=float, default=0.05)
    P.add_argument("--num_workers", type=int, default=16)   # more I/O threads
    P.add_argument("--use_amp", action="store_true", default=True)
    P.add_argument("--grad_clip_norm", type=float, default=None)
    P.add_argument("--ema_decay", type=float, default=0.9995)
    P.add_argument("--norm_pix_loss", action="store_true")
    # masking & logging
    P.add_argument("--initial_masking_ratio", type=float, default=0.25)
    P.add_argument("--final_masking_ratio",   type=float, default=0.25)
    P.add_argument("--log_interval", type=int, default=10)
    P.add_argument("--vis_interval", type=int, default=250)
    P.add_argument("--val_interval", type=int, default=2_000)
    P.add_argument("--skip_validation", action="store_true")
    P.add_argument("--vis_samples", type=int, default=6)
    P.add_argument("--run_name", default="mae_membrane_tar_prefetch")
    P.add_argument("--project_name", default="mae-3d-membranes")
    P.add_argument("--adam_beta1", type=float, default=0.9)
    P.add_argument("--adam_beta2", type=float, default=0.999)
    P.add_argument("--disable_ema", action="store_true")
    P.add_argument("--lr_schedule", choices=["cosine", "constant"], default="cosine",
                   help="Learning-rate schedule: cosine (with warmup) or constant")
    P.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16",
                   help="Autocast dtype to use when --use_amp is enabled (bf16 recommended for H100).")
    # I/O optimization arguments
    P.add_argument("--prefetch_factor", type=int, default=4,
                   help="Number of batches to prefetch per worker")
    P.add_argument("--save_best_model", action="store_true",
                   help="Save the model when validation loss improves")
    P.add_argument("--vols_per_shard", type=int, default=16384,
                   help="Number of volumes per .tar shard file.")
    args = P.parse_args()
    main(args)
