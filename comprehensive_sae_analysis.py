#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive SAE Analysis Suite (from Pre-computed Data)
=========================================================
A unified script to perform multi-faceted analysis of a trained Sparse
Autoencoder (SAE) using pre-computed activation data.

This script performs two major analysis modalities:

1.  **Quantitative Monosemanticity Analysis:**
    - Loads a large tensor of patch activations and a corresponding tensor
      of ground-truth class labels.
    - Feeds the activations through the SAE to get neuron firing rates.
    - Correlates individual neuron activations with the ground-truth classes.
    - Computes a "monosemanticity score" for each neuron.
    - Generates a JSON report with detailed statistics for all neurons.
    - *Note: Top-patch visualization is not performed here as the original
      image patches are not stored in the pre-computed file to save space.*

2.  **Qualitative Reconstruction Analysis (via a separate script):**
    - This script focuses on quantitative analysis. A separate script,
      `create_sae_reconstruction_viewer.py`, can be used for visual
      reconstruction analysis if needed.

All outputs are saved to a single, specified directory for a complete overview.
"""

# ───────────────────────── stdlib
import argparse, json, warnings
from pathlib import Path
from collections import defaultdict

# ───────────────────────── 3rd-party
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# ───────────────────────── project
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from sae_train import LinearSAE, GatedSAE
from vit_3d import get_device

warnings.filterwarnings("ignore", category=UserWarning, module='plotly')
CLASS_MAP = {0: "background", 1: "membrane", 2: "sphere", 3: "cube"}

@torch.no_grad()
def analyze_precomputed_data(args):
    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Loading SAE Model and Pre-computed Data ---")
    sae_ckpt = torch.load(args.sae_checkpoint, map_location="cpu")
    sae_cfg = sae_ckpt['config']
    
    # Load SAE model
    SAEClass = LinearSAE if sae_cfg['sae_variant'] == 'linear' else GatedSAE
    sae = SAEClass(input_dim=sae_ckpt['input_dim'], latent_dim=sae_ckpt['latent_dim']).to(device)
    sae_state_dict = {'encoder_weight' if k == 'sae_weight' else k: v for k, v in sae_ckpt.items() if k in ['sae_weight', 'encoder_bias', 'decoder_weight', 'decoder_bias']}
    sae.load_state_dict(sae_state_dict, strict=False)
    
    if args.compile_sae:
        print("Compiling SAE model with torch.compile...")
        sae = torch.compile(sae, mode="default", backend="inductor")
    sae.eval()
    
    # Load pre-computed activations and labels
    print(f"Loading pre-computed data from {args.activations_file}...")
    data = torch.load(args.activations_file)
    activations = data['activations']
    labels = data['labels']
    print(f"Loaded {activations.shape[0]} activations.")

    # Compute whitening stats from the loaded activations
    print("Computing whitening statistics...")
    act_mean = activations.mean(0, keepdim=True).to(device)
    act_std = activations.std(0, keepdim=True).to(device).clamp_(min=1e-6)
    print("Whitening stats computed.")

    print("\n--- Running Quantitative Analysis ---")
    latent_dim = sae_ckpt['latent_dim']
    neuron_class_activations = [defaultdict(list) for _ in range(latent_dim)]
    
    pbar = tqdm(range(0, len(activations), args.batch_size), desc="Analyzing activations")
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    for i in pbar:
        batch_activations = activations[i:i+args.batch_size].to(device)
        batch_labels = labels[i:i+args.batch_size]

        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=autocast_dtype):
            tokens_wh = (batch_activations - act_mean) / act_std
            sae_acts = sae(tokens_wh)[1]

        sae_acts = sae_acts.float().cpu().numpy()

        for j in range(sae_acts.shape[0]):
            label = batch_labels[j].item()
            class_name = CLASS_MAP[label]
            for n_idx in range(latent_dim):
                activation = sae_acts[j, n_idx]
                if activation > 1e-4:
                    neuron_class_activations[n_idx][class_name].append(activation)
    
    print("\nAnalyzing results and generating report...")
    analysis_results = []
    for n_idx in range(latent_dim):
        mean_activations = {name: np.mean(acts) if acts else 0.0 for name, acts in neuron_class_activations[n_idx].items()}
        total_activation = sum(mean_activations.values())
        top_class, top_act = max(mean_activations.items(), key=lambda item: item[1]) if mean_activations else ("none", 0)
        score = (top_act / total_activation) if total_activation > 1e-4 and top_act > 0 else 0.0
        analysis_results.append({
            "neuron_index": n_idx,
            "preferred_class": top_class,
            "monosemanticity_score": score,
            "mean_activations_by_class": mean_activations,
            "num_non_zero_activations": {k: len(v) for k, v in neuron_class_activations[n_idx].items()}
        })
    analysis_results.sort(key=lambda x: x['monosemanticity_score'], reverse=True)
    
    report_path = output_dir / "monosemanticity_report.json"
    with open(report_path, 'w') as f: json.dump(analysis_results, f, indent=4)
    print(f"Quantitative analysis complete. Report saved to: {report_path}")

    # Optional: Plotting top neuron scores
    plt.figure(figsize=(10, 6))
    top_scores = [r['monosemanticity_score'] for r in analysis_results[:100]]
    plt.plot(top_scores)
    plt.title("Monosemanticity Scores of Top 100 Neurons")
    plt.xlabel("Neuron Rank")
    plt.ylabel("Monosemanticity Score")
    plt.grid(True)
    plt.savefig(output_dir / "top_100_neuron_scores.png")
    plt.close()
    print("Saved plot of top 100 neuron scores.")

if __name__ == "__main__":
    P = argparse.ArgumentParser("Comprehensive SAE Analysis from Pre-computed Data")
    P.add_argument("--sae_checkpoint", type=str, required=True, help="Path to the trained SAE checkpoint")
    P.add_argument("--activations_file", type=str, required=True, help="Path to the pre-computed activations .pt file")
    P.add_argument("--output_dir", type=str, required=True, help="Directory to save all analysis results")
    P.add_argument("--batch_size", type=int, default=8192, help="Batch size for processing activations")
    P.add_argument("--compile_sae", action="store_true", help="Enable torch.compile for the SAE model.")
    P.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision.")
    P.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16", help="Autocast dtype for AMP.")
    main(P.parse_args()) 