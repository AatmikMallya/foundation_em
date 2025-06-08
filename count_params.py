#!/usr/bin/env python3

import torch
from vit_3d import mae_vit_3d_small, mae_vit_3d_base, mae_vit_3d_large, mae_vit_3d_huge

def count_parameters(model):
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_params(num_params):
    """Format parameter count in human readable form."""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)

def main():
    # Standard parameters for all models
    volume_size = (64, 64, 64)
    patch_size = (16, 16, 16)
    
    models = {
        'Small': mae_vit_3d_small,
        'Base': mae_vit_3d_base, 
        'Large': mae_vit_3d_large,
        'Huge': mae_vit_3d_huge
    }
    
    print("MAE 3D Vision Transformer Parameter Counts")
    print("=" * 50)
    print(f"Volume size: {volume_size}")
    print(f"Patch size: {patch_size}")
    print("=" * 50)
    
    for name, model_fn in models.items():
        model = model_fn(volume_size=volume_size, patch_size=patch_size)
        total_params, trainable_params = count_parameters(model)
        
        print(f"{name:6s}: {format_params(total_params):>8s} total parameters")
        print(f"        {format_params(trainable_params):>8s} trainable parameters")
        print()

if __name__ == '__main__':
    main() 