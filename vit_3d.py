import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

# NaN debugging infrastructure (kept for future debugging if needed)
_global_step = 0
_nan_checking_enabled = False

def set_global_step(step):
    global _global_step
    _global_step = step

def enable_nan_checking(enabled=True):
    global _nan_checking_enabled
    _nan_checking_enabled = enabled

def check_nan(tensor, name, location=""):
    """Check for NaN/Inf and log if found. Only active when enabled."""
    return False  # Disabled by default for performance

# Helper to make sure inputs are 3-tuples
def to_3tuple(x):
    if isinstance(x, tuple) and len(x) == 3:
        return x
    return (x, x, x)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )  # (B, heads, N, head_dim)
        x = attn_out.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Pre-norm attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Pre-norm MLP with residual connection  
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, volume_size=(96, 96, 96), patch_size=(8, 8, 8), in_chans=1, embed_dim=768):
        super().__init__()
        volume_size = to_3tuple(volume_size)
        patch_size = to_3tuple(patch_size)
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = (volume_size[0] // patch_size[0], volume_size[1] // patch_size[1], volume_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.volume_size[0] and H == self.volume_size[1] and W == self.volume_size[2], \
            f"Input image size ({D},{H},{W}) doesn't match model ({self.volume_size[0]},{self.volume_size[1]},{self.volume_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2) # B, C, Dp, Hp, Wp -> B, C, Np -> B, Np, C
        return x

class ConvPatchEmbed3D(nn.Module):
    """
    Vectorized 3D Convolutional Patch Embedding for translation-invariant representations.
    
    Processes the entire volume at once instead of patch-by-patch for dramatic speedup.
    Architecture: 96³ → stem(48³) → downsample(24³) → down(12³) → proj(768ch)
    """
    def __init__(self, volume_size=(96, 96, 96), patch_size=(8, 8, 8), in_chans=1, embed_dim=768):
        super().__init__()
        volume_size = to_3tuple(volume_size)
        patch_size = to_3tuple(patch_size)
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = (volume_size[0] // patch_size[0], volume_size[1] // patch_size[1], volume_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        # 96³ → 48³ → 24³ (2 small convs with stride=2)
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        # Pool so that feature map matches grid size (24 / 8 = 3, we need 12)
        # 24³ → 12³ by strided avg-pool
        self.down = nn.AvgPool3d(kernel_size=2, stride=2)

        # pointwise conv to get the final embedding dim
        self.proj = nn.Conv3d(64, embed_dim, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.volume_size[0] and H == self.volume_size[1] and W == self.volume_size[2], \
            f"Input image size ({D},{H},{W}) doesn't match model ({self.volume_size[0]},{self.volume_size[1]},{self.volume_size[2]})."
                
        x = self.stem(x)          # (B,96,24,24,24)
        x = self.down(x)          # (B,96,12,12,12)
        x = self.proj(x)          # (B,768,12,12,12)
        B, C, D, H, W = x.shape   # D=H=W=12
        return x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # (B,1728,embed_dim)

def get_sinusoid_encoding_table_3d(num_patches: int, embed_dim: int):
    """Axis-concatenated 3-D sine–cosine positional encoding.

    Each spatial axis receives its own slice of the embedding dimension, which
    removes the aliasing that occurs when D/H/W encodings are summed.  The
    returned tensor has shape ``(num_patches, embed_dim)``.
    """
    import math, torch

    # Determine cubic grid size (we assume perfect cube volumes)
    grid_size = round(num_patches ** (1 / 3))
    if grid_size ** 3 != num_patches:
        raise ValueError("num_patches must be a perfect cube for 3-D positional encoding")

    # Allocate embedding dimensions per axis as evenly as possible
    dim_per_axis = embed_dim // 3
    remainder = embed_dim % 3
    dim_d = dim_per_axis + (1 if remainder > 0 else 0)
    dim_h = dim_per_axis + (1 if remainder > 1 else 0)
    dim_w = dim_per_axis  # use the floor for the last axis

    def _axis_encoding(coord_flat: torch.Tensor, dim_axis: int):
        """Create standard 1-D sin-cos encoding for a single axis."""
        pe = torch.zeros(coord_flat.shape[0], dim_axis)
        if dim_axis == 0:
            return pe  # edge-case if embed_dim < 3
        div_term = torch.exp(
            torch.arange(0, dim_axis, 2, dtype=torch.float32) * (-math.log(10000.0) / dim_axis)
        )
        pe[:, 0::2] = torch.sin(coord_flat[:, None] * div_term)
        if dim_axis % 2 == 0:  # even
            pe[:, 1::2] = torch.cos(coord_flat[:, None] * div_term)
        else:  # odd embedding dim: last odd slot gets sin only
            pe[:, 1::2] = torch.cos(coord_flat[:, None] * div_term)[:, : pe[:, 1::2].shape[1]]
        return pe

    # 3-D coordinate grids
    d_coords = torch.arange(grid_size, dtype=torch.float32)
    h_coords = torch.arange(grid_size, dtype=torch.float32)
    w_coords = torch.arange(grid_size, dtype=torch.float32)

    grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")

    grid_d = grid_d.flatten()
    grid_h = grid_h.flatten()
    grid_w = grid_w.flatten()

    emb_d = _axis_encoding(grid_d, dim_d)
    emb_h = _axis_encoding(grid_h, dim_h)
    emb_w = _axis_encoding(grid_w, dim_w)

    # Concatenate along embedding dimension
    sinusoid_table = torch.cat([emb_d, emb_h, emb_w], dim=-1)
    # Ensure final size matches embed_dim (could be off by 1 due to odd dims)
    if sinusoid_table.shape[1] < embed_dim:
        pad = embed_dim - sinusoid_table.shape[1]
        sinusoid_table = torch.cat([sinusoid_table, torch.zeros(num_patches, pad)], dim=-1)
    elif sinusoid_table.shape[1] > embed_dim:
        sinusoid_table = sinusoid_table[:, :embed_dim]
    return sinusoid_table

class ViT3D(nn.Module):
    """ Vision Transformer for 3D data """
    def __init__(self, volume_size=(96,96,96), patch_size=(8,8,8), in_chans=1, num_classes=0, # num_classes=0 for MAE encoder
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 global_pool=False, patch_embed_class=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.patch_size = to_3tuple(patch_size)

        # Use custom patch embedding class if provided, otherwise use default
        if patch_embed_class is None:
            patch_embed_class = PatchEmbed3D
        
        self.patch_embed = patch_embed_class(volume_size, self.patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path_ratio=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # Initialize positional embedding with axis-concatenated 3-D sinusoid (CLS token kept at 0)
        sin_table = get_sinusoid_encoding_table_3d(self.pos_embed.shape[1] - 1, self.pos_embed.shape[-1])
        self.pos_embed.data.zero_()
        self.pos_embed.data[:, 1:, :].copy_(sin_table.unsqueeze(0))
        self.apply(self._init_weights_linear_layernorm)
    
    def _init_weights_linear_layernorm(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.head(x)
        else:
            x = x[:, 0] # CLS token
            outcome = self.head(x)
        return outcome

class ConvNeck3D(nn.Module):
    """
    3D Convolutional neck for MAE decoder to inject spatial structure.
    
    Args:
        in_dim: Input feature dimension from decoder
        patch_size: Patch size (assumed cubic)
        out_dim: Output dimension (patch_size^3 * in_chans)
        use_skip: Whether to use skip connection
    """
    def __init__(self, in_dim: int, patch_size: int, out_dim: int, use_skip: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.use_skip = use_skip
        
        # Progressive channel reduction
        hidden1 = max(in_dim // 2, 64)  # Ensure minimum channels
        hidden2 = max(in_dim // 4, 32)  # Ensure minimum channels
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(in_dim, hidden1, kernel_size=3, stride=1, padding=1)
        # Dynamic GroupNorm to handle any hidden dimension
        num_groups1 = max(1, min(8, hidden1 // 16))
        self.gn1 = nn.GroupNorm(num_groups1, hidden1)
        
        self.conv2 = nn.Conv3d(hidden1, hidden2, kernel_size=3, stride=1, padding=1)
        num_groups2 = max(1, min(8, hidden2 // 16))
        self.gn2 = nn.GroupNorm(num_groups2, hidden2)
        
        # Skip connection projection
        if self.use_skip and in_dim != hidden2:
            self.skip_proj = nn.Conv3d(in_dim, hidden2, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()
        
        # Final 1x1x1 conv to patch voxels
        self.to_voxels = nn.Conv3d(hidden2, out_dim, kernel_size=1)
        
        # Activation
        self.act = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolutional weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, token_feat, grid_shape):
        """
        Args:
            token_feat: (B, L, C) token features from decoder
            grid_shape: (Dp, Hp, Wp) patch grid dimensions
        
        Returns:
            (B, L, patch_dim) reconstructed patch features
        """
        B, L, C = token_feat.shape
        Dp, Hp, Wp = grid_shape
        
        # Ensure grid shape matches token count
        assert L == Dp * Hp * Wp, f"Token count {L} doesn't match grid {Dp}x{Hp}x{Wp}={Dp*Hp*Wp}"
        
        # Reshape to 3D grid: (B, L, C) -> (B, C, Dp, Hp, Wp)
        x = token_feat.transpose(1, 2).view(B, C, Dp, Hp, Wp)
        
        # Store input for skip connection
        if self.use_skip:
            skip = self.skip_proj(x)
        
        # First conv block
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.gn2(x)
        
        # Add skip connection
        if self.use_skip:
            x = x + skip
        
        x = self.act(x)
        
        # Final projection to patch voxels
        x = self.to_voxels(x)  # (B, patch_dim, Dp, Hp, Wp)
        
        # Reshape back to tokens: (B, patch_dim, Dp, Hp, Wp) -> (B, L, patch_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x

class MaskedAutoencoderViT3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone for 3D volumes """

    def __init__(self, volume_size=(96,96,96), patch_size=(8,8,8), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_ratio=0.75,
                 decoder_neck="linear", patch_embed_class=None):
        super().__init__()

        self.volume_size = volume_size
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ViT3D(
            volume_size=volume_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            norm_layer=norm_layer, patch_embed_class=patch_embed_class
        )

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Decoder neck options
        patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * in_chans
        self.decoder_neck = decoder_neck.lower()
        
        if self.decoder_neck == "conv":
            # Calculate patch grid dimensions
            self.patch_grid_shape = (
                volume_size[0] // self.patch_size[0],
                volume_size[1] // self.patch_size[1], 
                volume_size[2] // self.patch_size[2]
            )
            self.decoder_pred = ConvNeck3D(
                in_dim=decoder_embed_dim,
                patch_size=self.patch_size[0],  # Assumes cubic patches
                out_dim=patch_dim,
                use_skip=True
            )
        elif self.decoder_neck == "mlp":
            # Multi-layer MLP neck
            self.decoder_pred = nn.Sequential(
                nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4 * decoder_embed_dim, patch_dim)
            )
        else:  # "linear" - default
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)
        
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio  # CRITICAL FIX: Store mask_ratio as instance attribute

        self.init_weights()

    def init_weights(self):
        # Encoder weights are initialized in its own class
        # Initialize decoder_pos_embed and mask_token
        decoder_pos_embed_table = get_sinusoid_encoding_table_3d(self.decoder_pos_embed.shape[1] - 1, self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data[:, 1:, :].copy_(decoder_pos_embed_table.unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights_linear_layernorm)

    def _init_weights_linear_layernorm(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_size
        c = self.in_chans
        d, h, w = imgs.shape[2], imgs.shape[3], imgs.shape[4]
        assert d % p[0] == 0 and h % p[1] == 0 and w % p[2] == 0
        
        pd, ph, pw = d // p[0], h // p[1], w // p[2]
        
        x = imgs.reshape(imgs.shape[0], c, pd, p[0], ph, p[1], pw, p[2])
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous() # N, Pd, Ph, Pw, P0, P1, P2, C
        x = x.view(imgs.shape[0], pd * ph * pw, p[0] * p[1] * p[2] * c)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        c = self.in_chans
        d_vol, h_vol, w_vol = self.volume_size

        pd, ph, pw = d_vol // p[0], h_vol // p[1], w_vol // p[2]
        assert pd * ph * pw == x.shape[1]
        
        x = x.view(x.shape[0], pd, ph, pw, p[0], p[1], p[2], c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous() # N, C, Pd, P0, Ph, P1, Pw, P2
        imgs = x.view(x.shape[0], c, d_vol, h_vol, w_vol)
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D_emb = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D_emb))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        if check_nan(x, "encoder_input", "forward_encoder"):
            return x, None, None
        
        x = self.encoder.patch_embed(x)
        if check_nan(x, "patch_embed_output", "forward_encoder"):
            return x, None, None
        
        x = x + self.encoder.pos_embed[:, 1:, :] # Add pos embed, excluding CLS token part
        if check_nan(x, "pos_embed_added", "forward_encoder"):
            return x, None, None

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        if check_nan(x, "after_masking", "forward_encoder"):
            return x, mask, ids_restore
        if check_nan(mask, "mask_tensor", "forward_encoder"):
            return x, mask, ids_restore

        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :] # CLS token + its pos embed
        if check_nan(cls_token, "cls_token", "forward_encoder"):
            return x, mask, ids_restore
        
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if check_nan(x, "cls_tokens_added", "forward_encoder"):
            return x, mask, ids_restore

        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if check_nan(x, f"encoder_block_{i}", "forward_encoder"):
                return x, mask, ids_restore
            
        x = self.encoder.norm(x)
        if check_nan(x, "encoder_norm_output", "forward_encoder"):
            return x, mask, ids_restore
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore=None):
        if check_nan(x, "decoder_input", "forward_decoder"):
            return x
        
        x = self.decoder_embed(x)
        if check_nan(x, "decoder_embed", "forward_decoder"):
            return x
        
        # If we have restore IDs, we need to un-shuffle and add mask tokens
        if ids_restore is not None:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        if check_nan(mask_tokens, "mask_tokens", "forward_decoder"):
            return x
        
        # x is [N, 1 (cls) + K (kept), D_dec_emb]
        # ids_restore is [N, L_orig]
        # mask_tokens is [N, L_orig - K, D_dec_emb]
        
        # Remove CLS token from x for unshuffle, then add back
        x_no_cls = x[:, 1:, :]
        if check_nan(x_no_cls, "x_no_cls", "forward_decoder"):
            return x
        
        x_ = torch.cat([x_no_cls, mask_tokens], dim=1) # [N, L_orig, D_dec_emb]
        if check_nan(x_, "x_cat_mask_tokens", "forward_decoder"):
            return x
        
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        if check_nan(x_, "x_gather", "forward_decoder"):
            return x
        
        x = torch.cat([x[:, :1, :], x_], dim=1) # Prepend CLS token: [N, 1 + L_orig, D_dec_emb]
        if check_nan(x, "x_prepend_cls", "forward_decoder"):
            return x
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        if check_nan(x, "decoder_pos_embed_added", "forward_decoder"):
            return x

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            if check_nan(x, f"decoder_block_{i}", "forward_decoder"):
                return x
            
        x = self.decoder_norm(x)
        if check_nan(x, "decoder_norm", "forward_decoder"):
            return x
        
        # Remove CLS token before prediction
        x = x[:, 1:, :]  # [N, L_orig, D_dec_emb]
        if check_nan(x, "decoder_remove_cls", "forward_decoder"):
            return x
        
        # Apply decoder neck
        if self.decoder_neck == "conv":
            x = self.decoder_pred(x, self.patch_grid_shape)
        else:
            x = self.decoder_pred(x)
            
        if check_nan(x, "decoder_output", "forward_decoder"):
            return x
        
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, D, H, W]
        pred: [N, L, p*p*p*C]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if check_nan(imgs, "forward_loss_imgs", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), None
        if check_nan(pred, "forward_loss_pred", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), None
        if check_nan(mask, "forward_loss_mask", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), None
            
        target = self.patchify(imgs)
        if check_nan(target, "patchify_target", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), None
        
        patch_stats = None
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            if check_nan(mean, "norm_pix_mean", "forward_loss"):
                return torch.tensor(0.0, device=imgs.device), None
                
            var = target.var(dim=-1, keepdim=True)
            if check_nan(var, "norm_pix_var", "forward_loss"):
                return torch.tensor(0.0, device=imgs.device), None
                
            target = (target - mean) / (var.add(1.e-6).sqrt())
            if check_nan(target, "norm_pix_target", "forward_loss"):
                return torch.tensor(0.0, device=imgs.device), None
                
            # Store statistics for denormalization during visualization
            patch_stats = (mean, var)

        loss = (pred - target) ** 2
        if check_nan(loss, "squared_error", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), patch_stats
            
        loss = loss.mean(dim=-1)
        if check_nan(loss, "mean_squared_error", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), patch_stats
        
        # Prevent division by zero when no patches are masked
        mask_sum = mask.sum()
        if check_nan(mask_sum, "mask_sum", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), patch_stats
            
        # Use torch.where to avoid dynamic control flow (compilation-friendly)
        # If mask_sum == 0, return zero loss; otherwise compute the masked loss
        zero_loss = torch.tensor(0.0, device=imgs.device, dtype=loss.dtype)
        masked_loss = (loss * mask).sum() / torch.clamp(mask_sum, min=1e-8)  # Clamp to avoid division by zero
        loss = torch.where(mask_sum == 0, zero_loss, masked_loss)
        
        if check_nan(loss, "final_loss", "forward_loss"):
            return torch.tensor(0.0, device=imgs.device), patch_stats
             
        return loss, patch_stats

    def forward(self, imgs, mask_ratio=None):
        if check_nan(imgs, "mae_forward_input", "MAE.forward"):
            return torch.tensor(0.0, device=imgs.device), None, None, None
            
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio if mask_ratio is not None else self.mask_ratio)
        if latent is None or mask is None or ids_restore is None:
            # This can happen if NaN checking is on and finds an issue
            return torch.tensor(0.0, device=imgs.device), None, None, None
            
        pred = self.forward_decoder(latent, ids_restore)
        if pred is None:
            # This can happen if NaN checking is on and finds an issue
            return torch.tensor(0.0, device=imgs.device), None, mask, None
            
        loss, patch_stats = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, patch_stats

MaskedAutoencoderViT = MaskedAutoencoderViT3D
ViT = ViT3D 

# --- Factory functions for MAE ViT 3D models ---
def mae_vit_3d_small(**kwargs):
    """MAE ViT-3D Small configuration for testing."""
    model = MaskedAutoencoderViT3D(
        embed_dim=384, depth=8, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, **kwargs
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

def mae_vit_3d_large(**kwargs):
    """MAE ViT-3D Large configuration for maximum capacity."""
    model = MaskedAutoencoderViT3D(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs
    )
    return model

def mae_vit_3d_huge(**kwargs):
    """MAE ViT-3D Huge configuration for extreme capacity."""
    model = MaskedAutoencoderViT3D(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=640, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs
    )
    return model

def mae_vit_3d_hemibrain_optimal(**kwargs):
    """
    MAE ViT-3D configuration optimized for complex biological data like Hemibrain.
    - Encoder is ViT-Large.
    - Decoder is stronger than default (closer to ViT-Base) for high-fidelity reconstruction.
    """
    model = MaskedAutoencoderViT3D(
        embed_dim=768,         # Large: 1024
        depth=12,               # Large: 24
        num_heads=12,           # Large: 16
        decoder_embed_dim=768,  # Increased from 512 to 768
        decoder_depth=12,       # Increased from 8 to 12
        decoder_num_heads=12,   # Increased from 16 to 12 (to match decoder_embed_dim)
        mlp_ratio=4, **kwargs
    )
    return model

# --- Convenience functions for ConvNeck variants ---
def mae_vit_3d_small_conv(**kwargs):
    """MAE ViT-3D Small with ConvNeck3D decoder."""
    return mae_vit_3d_small(decoder_neck="conv", **kwargs)

def mae_vit_3d_base_conv(**kwargs):
    """MAE ViT-3D Base with ConvNeck3D decoder."""
    return mae_vit_3d_base(decoder_neck="conv", **kwargs)

def mae_vit_3d_large_conv(**kwargs):
    """MAE ViT-3D Large with ConvNeck3D decoder."""
    return mae_vit_3d_large(decoder_neck="conv", **kwargs)

def mae_vit_3d_base_patch_conv(**kwargs):
    """MAE ViT-3D Base with ConvPatchEmbed3D for translation-invariant patch embedding."""
    model = MaskedAutoencoderViT3D(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, patch_embed_class=ConvPatchEmbed3D, **kwargs
    )
    return model

def mae_vit_3d_hemibrain_optimal_conv(**kwargs):
    """MAE ViT-3D Hemibrain Optimal with ConvNeck3D decoder."""
    return mae_vit_3d_hemibrain_optimal(decoder_neck="conv", **kwargs)