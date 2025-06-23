#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from membrane_synthetic_data import MembraneSyntheticDataset
import torch

def generate_comparison_volumes():
    """
    Generate volumes with different parameters to show the effect on membrane size.
    """
    
    configs = [
        {
            "name": "Small Membranes (Original)",
            "num_gaussians_range": (8, 15),
            "gaussian_sigma_range": (5, 12),
            "isoband_width": 0.15
        },
        {
            "name": "Medium Membranes", 
            "num_gaussians_range": (6, 10),
            "gaussian_sigma_range": (10, 18),
            "isoband_width": 0.15
        },
        {
            "name": "Large Membranes (Updated)",
            "num_gaussians_range": (4, 8),
            "gaussian_sigma_range": (15, 25),
            "isoband_width": 0.15
        },
        {
            "name": "Extra Large Membranes",
            "num_gaussians_range": (3, 6),
            "gaussian_sigma_range": (20, 35),
            "isoband_width": 0.20
        }
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    fig.suptitle('Effect of Parameters on Membrane Structure Size', fontsize=20)
    
    for config_idx, config in enumerate(configs):
        print(f"\nGenerating {config['name']}...")
        
        # Create dataset with current config
        dataset = MembraneSyntheticDataset(
            volume_size=(64, 64, 64),
            num_gaussians_range=config["num_gaussians_range"],
            gaussian_sigma_range=config["gaussian_sigma_range"],
            isovalue=0.5,
            isoband_width=config["isoband_width"],
            noise_level=0.02,
            num_samples=1,
            seed=42,  # Same seed for fair comparison
            num_additional_spheres_range=(2, 5),
            additional_sphere_radius_range=(3.0, 8.0),
            blur_sigma=1.0,
            isovalue_variation=0.1,
            intensity_gradient_strength=0.4
        )
        
        volume = dataset[0]
        volume_np = volume.squeeze().numpy()
        D, H, W = volume_np.shape
        
        # Show three representative slices
        slice_positions = [D//4, D//2, 3*D//4]
        
        for slice_idx, pos in enumerate(slice_positions):
            ax = axes[config_idx, slice_idx]
            ax.imshow(volume_np[pos, :, :], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{config["name"]}\nZ-slice {pos}\nGaussians: {config["num_gaussians_range"]}\nSigma: {config["gaussian_sigma_range"]}')
            ax.axis('off')
        
        print(f"  - Volume mean: {volume.mean():.4f}")
        print(f"  - Volume std: {volume.std():.4f}")
    
    plt.tight_layout()
    output_file = "membrane_size_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to {output_file}")
    
    return fig

def explain_parameters():
    """
    Print detailed explanation of how parameters affect membrane structure.
    """
    print("\n" + "="*80)
    print("PARAMETER EFFECTS ON MEMBRANE STRUCTURE SIZE")
    print("="*80)
    
    print("\n1. GAUSSIAN_SIGMA_RANGE - Controls membrane structure SIZE")
    print("   • Small sigma (5-12): Creates small, fine membrane structures")
    print("   • Medium sigma (10-18): Creates medium-sized membrane regions") 
    print("   • Large sigma (15-25): Creates large, continuous membrane areas")
    print("   • Extra large sigma (20-35): Creates very large membrane sheets")
    print("   → This is the MOST IMPORTANT parameter for structure size!")
    
    print("\n2. NUM_GAUSSIANS_RANGE - Controls membrane COMPLEXITY/FRAGMENTATION")
    print("   • Many gaussians (8-15): More complex, fragmented structures")
    print("   • Fewer gaussians (4-8): Simpler, more continuous structures")
    print("   • Very few gaussians (2-5): Very simple, large continuous regions")
    print("   → Fewer gaussians = larger continuous membrane regions")
    
    print("\n3. ISOBAND_WIDTH - Controls membrane THICKNESS (not size)")
    print("   • Small width (0.05-0.1): Thin membranes")
    print("   • Medium width (0.15-0.2): Normal thickness membranes")
    print("   • Large width (0.25-0.3): Thick membranes")
    print("   → Only affects thickness, NOT the overall structure size")
    
    print("\n4. VOLUME_SIZE - Affects RELATIVE scale")
    print("   • Larger volume with same gaussian parameters = relatively smaller structures")
    print("   • Smaller volume with same gaussian parameters = relatively larger structures")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR LARGER MEMBRANES:")
    print("="*80)
    print("✓ INCREASE gaussian_sigma_range (e.g., 15-25 or 20-30)")
    print("✓ DECREASE num_gaussians_range (e.g., 3-6 or 4-8)")
    print("✓ Optionally increase isoband_width for thicker membranes")
    print("✓ Reduce additional spheres to avoid cluttering")

if __name__ == "__main__":
    explain_parameters()
    generate_comparison_volumes() 