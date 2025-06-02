import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union
import platform

class PatchEmbedding3D(nn.Module):
    """3D patch embedding for converting voxel volumes to patches."""
    
    def __init__(self, volume_size: Tuple[int, int, int] = (64, 64, 64), 
                 patch_size: Tuple[int, int, int] = (8, 8, 8),
                 in_channels: int = 1, 
                 embed_dim: int = 768):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches per dimension
        self.num_patches_d = volume_size[0] // patch_size[0]
        self.num_patches_h = volume_size[1] // patch_size[1]
        self.num_patches_w = volume_size[2] // patch_size[2]
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w
        
        # Patch projection via 3D convolution
        self.projection = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Patches of shape (B, num_patches, embed_dim)
        """
        B, C, D, H, W = x.shape
        
        # Project to patches
        x = self.projection(x)  # (B, embed_dim, num_patches_d, num_patches_h, num_patches_w)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x

class PositionalEncoding3D(nn.Module):
    """3D positional encoding for voxel patches."""
    
    def __init__(self, embed_dim: int, num_patches_d: int, num_patches_h: int, num_patches_w: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_d = num_patches_d
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Create learnable positional embeddings
        num_patches = num_patches_d * num_patches_h * num_patches_w
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        """Add positional encoding to patch embeddings."""
        return x + self.pos_embedding

class MultiHeadAttention3D(nn.Module):
    """Multi-head self attention for 3D patches."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class MLP(nn.Module):
    """MLP block for Transformer."""
    
    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention3D(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class ViT3D(nn.Module):
    """3D Vision Transformer for EM voxel data."""
    
    def __init__(self, 
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 patch_size: Tuple[int, int, int] = (8, 8, 8),
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 num_classes: int = 1000):  # For classification head if needed
        super().__init__()
        
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(volume_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional encoding
        self.pos_embed = PositionalEncoding3D(
            embed_dim, 
            self.patch_embed.num_patches_d,
            self.patch_embed.num_patches_h, 
            self.patch_embed.num_patches_w
        )
        
        # Class token (optional, useful for classification tasks)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head (optional)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """Forward pass through the transformer blocks."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        return x
    
    def forward(self, x):
        """Forward pass."""
        x = self.forward_features(x)
        
        # Extract class token for classification
        cls_token = x[:, 0]
        x = self.head(cls_token)
        
        return x
    
    def get_intermediate_features(self, x, layer_idx: int = 6):
        """Get features from a specific layer (useful for SAE training)."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        
        # Pass through blocks up to layer_idx
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == layer_idx:
                return x
        
        return self.norm(x)

class MaskedAutoencoderViT3D(nn.Module):
    """3D Masked Autoencoder using Vision Transformer for EM data."""
    
    def __init__(self,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 patch_size: Tuple[int, int, int] = (8, 8, 8),
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 mask_ratio: float = 0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Encoder (ViT)
        self.encoder = ViT3D(
            volume_size=volume_size,
            patch_size=patch_size, 
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_classes=0  # No classification head
        )
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(decoder_embed_dim) * 0.02)
        
        # Decoder positional encoding
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, decoder_embed_dim) * 0.02)
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Decoder prediction head
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_volume, bias=True)
        self.decoder_pred_activation = nn.Sigmoid()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, volumes):
        """Convert volumes to patches."""
        B, C, D, H, W = volumes.shape
        pd, ph, pw = self.patch_size
        
        assert D % pd == 0 and H % ph == 0 and W % pw == 0
        
        d = D // pd
        h = H // ph  
        w = W // pw
        
        x = volumes.reshape(B, C, d, pd, h, ph, w, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, d * h * w, C * pd * ph * pw)
        
        return x
    
    def unpatchify(self, patches):
        """Convert patches back to volumes."""
        B, N, patch_dim = patches.shape
        pd, ph, pw = self.patch_size
        C = self.in_channels
        
        assert patch_dim == C * pd * ph * pw
        
        # Use explicit patch counts from encoder's patch_embed module
        d_patches = self.encoder.patch_embed.num_patches_d
        h_patches = self.encoder.patch_embed.num_patches_h
        w_patches = self.encoder.patch_embed.num_patches_w
        assert d_patches * h_patches * w_patches == N, f"Total patches {N} does not match product of per-dimension patches {d_patches*h_patches*w_patches}"
        
        x = patches.reshape(B, d_patches, h_patches, w_patches, C, pd, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() # B, C, d_patches, pd, h_patches, ph, w_patches, pw
        x = x.reshape(B, C, d_patches * pd, h_patches * ph, w_patches * pw) # B, C, D, H, W
        
        return x
    
    def random_masking(self, x, mask_ratio):
        """Perform random masking."""
        B, N, D = x.shape  # batch, length, dim
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """Forward pass through encoder with masking."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.encoder.patch_embed(x)
        x = self.encoder.pos_embed(x)
        
        # Add class token
        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Masking: only mask the patch tokens (not the class token)
        cls_tokens = x[:, :1, :]  # Extract class token
        patch_tokens = x[:, 1:, :]  # Extract patch tokens
        
        patch_tokens_masked, mask, ids_restore = self.random_masking(patch_tokens, mask_ratio)
        
        # Concatenate class token back
        x = torch.cat((cls_tokens, patch_tokens_masked), dim=1)
        
        # Apply Transformer blocks
        x = self.encoder.dropout(x)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Forward pass through decoder."""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Separate class token from patches
        cls_token = x[:, :1, :]  # Class token
        patch_tokens = x[:, 1:, :]  # Patch tokens
        
        # Append mask tokens to sequence (only for patches)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - patch_tokens.shape[1], 1)
        x_ = torch.cat([patch_tokens, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([cls_token, x_], dim=1)  # append cls token
        
        # Add positional encoding - make sure dimensions match
        if x.shape[1] != self.decoder_pos_embed.shape[1]:
            # Resize positional embedding if needed
            decoder_pos_embed = self.decoder_pos_embed[:, :x.shape[1], :]
        else:
            decoder_pos_embed = self.decoder_pos_embed
            
        x = x + decoder_pos_embed
        
        # Apply Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        x = self.decoder_pred_activation(x)
        
        # Remove class token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, volumes, pred, mask):
        """Compute reconstruction loss."""
        target = self.patchify(volumes)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        
        # Handle case where mask.sum() is zero (e.g., mask_ratio = 0.0)
        if mask.sum() == 0:
            # If no masks, MAE acts like a normal autoencoder, loss is over all patches
            # This also means the 'mask' input to this function is all zeros.
            # The 'loss' variable currently holds per-patch losses.
            # We need the mean of these per-patch losses.
            final_loss = loss.mean()
        else:
            final_loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            
        return final_loss
    
    def forward(self, volumes, mask_ratio=None):
        """Forward pass."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore = self.forward_encoder(volumes, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(volumes, pred, mask)
        
        return loss, pred, mask

def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and platform.system() == 'Darwin':
        # Check if Conv3D is supported on MPS
        try:
            # Test with a small conv3d operation
            test_tensor = torch.randn(1, 1, 4, 4, 4).to('mps')
            test_conv = nn.Conv3d(1, 1, 3).to('mps')
            _ = test_conv(test_tensor)
            return torch.device('mps')
        except RuntimeError as e:
            if "Conv3D is only supported on MPS for MacOS" in str(e):
                print("Warning: Conv3D not supported on this MPS version, falling back to CPU")
                return torch.device('cpu')
            else:
                print(f"MPS error: {e}, falling back to CPU")
                return torch.device('cpu')
    else:
        return torch.device('cpu')

# Model configurations for different scales
def vit_3d_tiny(**kwargs):
    """ViT-3D Tiny configuration."""
    model = ViT3D(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs
    )
    return model

def vit_3d_small(**kwargs):
    """ViT-3D Small configuration."""
    model = ViT3D(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs
    )
    return model

def vit_3d_base(**kwargs):
    """ViT-3D Base configuration (ViT-B)."""
    model = ViT3D(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs
    )
    return model

def vit_3d_large(**kwargs):
    """ViT-3D Large configuration."""
    model = ViT3D(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs
    )
    return model

def mae_vit_3d_base(**kwargs):
    """MAE ViT-3D Base configuration."""
    model = MaskedAutoencoderViT3D(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs
    )
    return model

def mae_vit_3d_small(**kwargs):
    """MAE ViT-3D Small configuration for testing."""
    model = MaskedAutoencoderViT3D(
        embed_dim=384, depth=8, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, **kwargs
    )
    return model

if __name__ == "__main__":
    # Test the model
    device = get_device()
    print(f"Using device: {device}")
    
    # Create a small test volume
    batch_size = 2
    volume_size = (32, 32, 32)  # Smaller for testing
    test_volume = torch.randn(batch_size, 1, *volume_size).to(device)
    
    # Test regular ViT
    print("Testing ViT-3D...")
    model = vit_3d_small(volume_size=volume_size, patch_size=(8, 8, 8)).to(device)
    output = model(test_volume)
    print(f"ViT output shape: {output.shape}")
    
    # Test MAE
    print("\nTesting MAE ViT-3D...")
    mae_model = mae_vit_3d_small(volume_size=volume_size, patch_size=(8, 8, 8)).to(device)
    loss, pred, mask = mae_model(test_volume)
    print(f"MAE loss: {loss.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masking ratio: {mask.mean().item():.2f}")
    
    # Test intermediate features (for SAE later)
    print("\nTesting intermediate features...")
    features = model.get_intermediate_features(test_volume, layer_idx=6)
    print(f"Layer 6 features shape: {features.shape}")
    
    print("\nAll tests passed! ✅") 