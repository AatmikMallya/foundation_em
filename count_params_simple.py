#!/usr/bin/env python3

def estimate_mae_vit_3d_params(embed_dim, depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads, 
                               volume_size=(64,64,64), patch_size=(16,16,16), mlp_ratio=4):
    """Estimate parameters for MAE ViT 3D model without instantiating it."""
    
    # Calculate number of patches
    num_patches = (volume_size[0] // patch_size[0]) * (volume_size[1] // patch_size[1]) * (volume_size[2] // patch_size[2])
    patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * 1  # in_chans = 1
    
    # Encoder parameters
    # Patch embedding
    patch_embed_params = patch_dim * embed_dim + embed_dim
    
    # Position embedding (learnable)
    pos_embed_params = (num_patches + 1) * embed_dim
    
    # CLS token
    cls_token_params = embed_dim
    
    # Transformer blocks
    block_params_per_layer = (
        # LayerNorm 1
        embed_dim * 2 +
        # Attention (qkv + proj)
        embed_dim * embed_dim * 3 + embed_dim * 3 +  # qkv with bias
        embed_dim * embed_dim + embed_dim +  # proj with bias
        # LayerNorm 2  
        embed_dim * 2 +
        # MLP
        embed_dim * (mlp_ratio * embed_dim) + (mlp_ratio * embed_dim) +  # fc1 with bias
        (mlp_ratio * embed_dim) * embed_dim + embed_dim  # fc2 with bias
    )
    encoder_block_params = block_params_per_layer * depth
    
    # Final LayerNorm
    final_norm_params = embed_dim * 2
    
    # Decoder parameters
    # Decoder embedding projection
    decoder_embed_params = embed_dim * decoder_embed_dim + decoder_embed_dim
    
    # Mask token
    mask_token_params = decoder_embed_dim
    
    # Decoder pos embed (fixed, so 0 trainable)
    decoder_pos_embed_params = 0  # Fixed sinusoidal
    
    # Decoder transformer blocks
    decoder_block_params_per_layer = (
        # LayerNorm 1
        decoder_embed_dim * 2 +
        # Attention
        decoder_embed_dim * decoder_embed_dim * 3 + decoder_embed_dim * 3 +
        decoder_embed_dim * decoder_embed_dim + decoder_embed_dim +
        # LayerNorm 2
        decoder_embed_dim * 2 +
        # MLP
        decoder_embed_dim * (mlp_ratio * decoder_embed_dim) + (mlp_ratio * decoder_embed_dim) +
        (mlp_ratio * decoder_embed_dim) * decoder_embed_dim + decoder_embed_dim
    )
    decoder_block_params = decoder_block_params_per_layer * decoder_depth
    
    # Decoder norm
    decoder_norm_params = decoder_embed_dim * 2
    
    # Decoder prediction head
    decoder_pred_params = decoder_embed_dim * patch_dim + patch_dim
    
    total_params = (
        patch_embed_params + pos_embed_params + cls_token_params + 
        encoder_block_params + final_norm_params + 
        decoder_embed_params + mask_token_params + decoder_pos_embed_params +
        decoder_block_params + decoder_norm_params + decoder_pred_params
    )
    
    return total_params

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
    models = {
        'Small': (384, 8, 6, 256, 4, 8),      # embed_dim, depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads
        'Base':  (768, 12, 12, 512, 8, 16),
        'Large': (1024, 24, 16, 512, 8, 16),
        'Huge':  (1280, 32, 16, 640, 8, 16),
        'Hemibrain': (1024, 24, 16, 768, 12, 12) # embed, depth, heads, dec_embed, dec_depth, dec_heads
    }
    
    print("MAE 3D Vision Transformer Parameter Estimates")
    print("=" * 50)
    print("Volume size: (64, 64, 64)")
    print("Patch size: (16, 16, 16)")
    print("=" * 50)
    
    for name, params in models.items():
        total_params = estimate_mae_vit_3d_params(*params)
        print(f"{name:6s}: {format_params(total_params):>8s} parameters")
    
    print("\nParameter scaling:")
    base_params = estimate_mae_vit_3d_params(*models['Base'])
    for name, params in models.items():
        total_params = estimate_mae_vit_3d_params(*params)
        ratio = total_params / base_params
        print(f"{name:10s}: {ratio:.1f}x Base model size")

if __name__ == '__main__':
    main() 