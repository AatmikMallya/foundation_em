#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compute SAE Activations and Labels
======================================
This script performs the first stage of a two-stage analysis pipeline.
Its sole purpose is to run the expensive data loading and model inference
to extract the necessary data for SAE analysis, saving it to an intermediate file.

Workflow:
1. Loads a pretrained and compiled Masked Autoencoder (MAE) model.
2. Creates an optimized DataLoader to iterate through the synthetic dataset.
3. For a specified number of volumes, it performs a forward pass through the
   MAE's encoder to get patch-level activations from a specific layer.
4. Simultaneously, it processes the corresponding segmentation masks to get a
   ground-truth class label for each patch.
5. It saves two tensors to a single file:
   - A `(N, D)` tensor of activations, where N is the total number of patches
     and D is the feature dimension.
   - A `(N,)` tensor of integer class labels for each patch.

This decouples the slow data extraction from the fast analysis, making the
downstream analysis much more efficient and reliable.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from vit_3d import mae_vit_3d_base_conv, get_device
from comprehensive_sae_analysis import TarShardDatasetWithMasks, patchify_numpy, get_patch_class
from sae_train import TokenExtractor

@torch.no_grad()
def precompute_activations(args):
    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("--- Loading MAE Model ---")
    mae = mae_vit_3d_base_conv(volume_size=(args.img_size,)*3, patch_size=args.patch_size).to(device)
    mae_ckpt = torch.load(args.mae_checkpoint, map_location="cpu")
    mae.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in mae_ckpt.get("model_state_dict", mae_ckpt).items()}, strict=False)
    
    print("Compiling MAE model with torch.compile...")
    mae = torch.compile(mae, mode="default", backend="inductor")
    mae.eval()

    print("--- Setting up Dataset ---")
    shards = sorted(Path(args.shard_dir).expanduser().glob("shard_*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.tar files found in {args.shard_dir}")
    dataset = TarShardDatasetWithMasks(shards, args.img_size, mask_dir=Path(args.shard_dir) / "masks")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    token_extractor = TokenExtractor(mae, args.layer, extract_from="encoder")

    all_activations = []
    all_labels = []
    
    volumes_processed = 0
    target_volumes = args.num_volumes
    pbar = tqdm(loader, desc="Extracting activations and labels", unit="batch")

    print(f"Starting extraction for {target_volumes} volumes...")
    for volumes, masks in pbar:
        if volumes_processed >= target_volumes:
            break
        
        batch_size = volumes.shape[0]
        actual_batch_size = min(batch_size, target_volumes - volumes_processed)
        if actual_batch_size < batch_size:
            volumes = volumes[:actual_batch_size]
            masks = masks[:actual_batch_size]

        # Get activations from MAE
        activations = token_extractor.extract_tokens(volumes.to(device))
        all_activations.append(activations.cpu())

        # Get corresponding labels from masks
        for i in range(actual_batch_size):
            mask_patches = patchify_numpy(masks[i].squeeze().numpy(), args.patch_size)
            labels = [get_patch_class(p) for p in mask_patches]
            all_labels.extend(labels)
        
        volumes_processed += actual_batch_size
        pbar.set_postfix(volumes_processed=f"{volumes_processed}/{target_volumes}")

    token_extractor.cleanup()

    print("\nConcatenating results...")
    final_activations = torch.cat(all_activations, dim=0)
    final_labels = torch.tensor(all_labels, dtype=torch.uint8)

    print(f"Final activations shape: {final_activations.shape}")
    print(f"Final labels shape: {final_labels.shape}")

    print(f"Saving pre-computed data to {output_path}...")
    torch.save({
        'activations': final_activations,
        'labels': final_labels,
        'config': vars(args)
    }, output_path)

    print("Pre-computation complete.")

if __name__ == "__main__":
    P = argparse.ArgumentParser("Pre-compute SAE Activations and Labels")
    P.add_argument("--mae_checkpoint", type=str, required=True, help="Path to the pretrained MAE checkpoint")
    P.add_argument("--shard_dir", type=str, required=True, help="Directory with volume and mask Tar Shards")
    P.add_argument("--output_file", type=str, required=True, help="Path to save the output .pt file")
    P.add_argument("--layer", type=int, default=8, help="Encoder layer to extract activations from")
    P.add_argument("--num_volumes", type=int, default=4096, help="Number of volumes to process")
    P.add_argument("--img_size", type=int, default=96)
    P.add_argument("--patch_size", type=int, default=8)
    P.add_argument("--batch_size", type=int, default=64)
    P.add_argument("--num_workers", type=int, default=16)
    main(P.parse_args()) 