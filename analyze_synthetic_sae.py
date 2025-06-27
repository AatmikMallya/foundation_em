#!/usr/bin/env python3
"""
Synthetic Data SAE Analysis Helper
=================================
Helps interpret SAE results for synthetic data with 6 spheres + 1 membrane.
Provides guidance on what patterns to expect and how to improve results.

Usage:
    python3 analyze_synthetic_sae.py
"""

import argparse
from pathlib import Path

def analyze_sae_for_synthetic_data():
    """
    Guide for interpreting SAE results on synthetic sphere + membrane data.
    """
    
    print("=" * 60)
    print("SAE ANALYSIS GUIDE FOR SYNTHETIC DATA")
    print("=" * 60)
    
    print("\n🎯 EXPECTED MONOSEMANTIC FEATURES:")
    print("   For your data (6 identical spheres + 1 membrane), you should see:")
    
    print("\n   SPHERE FEATURES:")
    print("   ├── Sphere centers (dark cores)")
    print("   ├── Sphere boundaries (curved edges)")  
    print("   ├── Sphere convex surfaces (different orientations)")
    print("   └── Sphere-background transitions")
    
    print("\n   MEMBRANE FEATURES:")
    print("   ├── Membrane edges (linear boundaries)")
    print("   ├── Membrane thickness patterns")
    print("   ├── Membrane orientation (if curved)")
    print("   └── Membrane-background transitions")
    
    print("\n   SPATIAL FEATURES:")
    print("   ├── Background regions (empty space)")
    print("   ├── Object proximity patterns")
    print("   └── Corner/edge regions of volume")
    
    print("\n📊 WHAT GOOD vs BAD LATENTS LOOK LIKE:")
    
    print("\n   ✅ GOOD LATENTS (Monosemantic):")
    print("   ├── Fire sparsely (2-6% of patches)")
    print("   ├── Show clear geometric patterns (circles, lines)")
    print("   ├── Consistent across different slices")
    print("   ├── High object selectivity (>0.6)")
    print("   └── Form coherent clusters (not scattered)")
    
    print("\n   ❌ BAD LATENTS (Polysemantic/Noisy):")
    print("   ├── Fire too frequently (>20% of patches)")
    print("   ├── Scattered, random-looking activations")
    print("   ├── Inconsistent patterns across slices")
    print("   ├── Low selectivity (mixed object/background)")
    print("   └── Many small disconnected clusters")
    
    print("\n🔧 TROUBLESHOOTING NOISY PATTERNS:")
    
    print("\n   If you see scattered activations like in your example:")
    
    print("\n   1. L1 COEFFICIENT TOO HIGH:")
    print("      ├── Symptoms: Very sparse but noisy activations")
    print("      ├── Solution: Reduce L1 from 2e-3 to 1e-3 or 5e-4")
    print("      └── Goal: Allow latents to fire more coherently")
    
    print("\n   2. INSUFFICIENT TRAINING:")
    print("      ├── Symptoms: Activations still organizing")
    print("      ├── Solution: Train for more epochs")
    print("      └── Check: Is validation MSE still decreasing?")
    
    print("\n   3. WRONG LAYER CHOICE:")
    print("      ├── Current: Layer 6 (abstract features)")
    print("      ├── Try: Layers 3-5 (more geometric)")
    print("      └── Reason: Earlier layers capture shape better")
    
    print("\n   4. PATCH SIZE MISMATCH:")
    print("      ├── Current: 4×4×4 patches")
    print("      ├── Problem: Too small for sphere curvature")
    print("      ├── Solution: Try 8×8×8 patches")
    print("      └── Trade-off: Larger patches, less spatial resolution")
    
    print("\n   5. DATA TOO SIMPLE:")
    print("      ├── Problem: 6 identical spheres → limited diversity")
    print("      ├── Solution: Add sphere size variation")
    print("      ├── Or: Add different membrane shapes")
    print("      └── Goal: Give SAE more patterns to learn")
    
    print("\n🚀 RECOMMENDED NEXT STEPS:")
    
    print("\n   1. Run enhanced inspection:")
    print("      python3 inspect_sae.py \\")
    print("          --sae_checkpoint your_checkpoint.pt \\")
    print("          --enhanced \\")
    print("          --num_latents 40")
    
    print("\n   2. Look for these metrics in output:")
    print("      ├── Object selectivity > 0.6 (prefers spheres/membranes)")
    print("      ├── Activation clusters < 10 (coherent patterns)")
    print("      └── Active fraction 2-6% (appropriate sparsity)")
    
    print("\n   3. If still noisy, retrain with:")
    print("      ├── Lower L1: 1e-3 instead of 2e-3")
    print("      ├── Different layer: Try layer 4 or 5")
    print("      ├── More epochs: Until validation MSE plateaus")
    print("      └── Different latent multiplier: Try 4× or 16×")
    
    print("\n   4. Generate more diverse data:")
    print("      ├── Vary sphere sizes (radius 8-16 instead of fixed)")
    print("      ├── Add membrane curvature/branching")
    print("      ├── Include partially overlapping spheres")
    print("      └── Add noise/texture to make it more realistic")
    
    print("\n💡 INTERPRETATION TIPS:")
    
    print("\n   When examining latent_XXXX_enhanced.png files:")
    
    print("\n   ├── TOP ROW: Raw EM data")
    print("   ├── MIDDLE ROW: All activations (red = strong)")
    print("   ├── BOTTOM ROW: Top 30% activations only")
    print("   └── TEXT BOX: Quantitative metrics")
    
    print("\n   Look for latents that:")
    print("   ├── Show clear circular patterns (sphere detectors)")
    print("   ├── Show linear patterns (membrane detectors)")
    print("   ├── Have high object selectivity in text box")
    print("   └── Show consistent patterns across all 3 slice views")
    
    print("\n🔍 YOUR CURRENT RESULTS ANALYSIS:")
    print("\n   The scattered pattern you showed suggests:")
    print("   ├── L1 coefficient may be too high (over-regularized)")
    print("   ├── Latents are firing but not forming coherent patterns")
    print("   ├── SAE needs more training or different hyperparameters")
    print("   └── Consider trying layer 4-5 instead of layer 6")
    
    print("\n   This is NORMAL for initial training attempts!")
    print("   SAE training often requires hyperparameter tuning.")
    
    print("\n" + "=" * 60)
    print("Good luck with your SAE analysis! 🧠✨")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE analysis guide for synthetic data")
    
    args = parser.parse_args()
    
    analyze_sae_for_synthetic_data() 