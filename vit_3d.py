import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, volume_size=(64, 64, 64), patch_size=(8, 8, 8), in_chans=1, embed_dim=768):
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

def get_sinusoid_encoding_table_3d(num_patches, embed_dim):
    """
    3D Sinusoidal Positional Encoding Table
    """
    grid_size_float = num_patches**(1/3)
    grid_size = int(round(grid_size_float))
    if abs(grid_size_float - grid_size) > 1e-6 or grid_size**3 != num_patches :
        raise ValueError(f"Number of patches ({num_patches}) must be a perfect cube for 3D sinusoidal encoding.")

    sinusoid_table = torch.zeros(num_patches, embed_dim)
    
    grid_d_coords = torch.arange(grid_size, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).expand(-1, grid_size, grid_size)
    grid_h_coords = torch.arange(grid_size, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(grid_size, -1, grid_size)
    grid_w_coords = torch.arange(grid_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(grid_size, grid_size, -1)

    grid_d = grid_d_coords.flatten()
    grid_h = grid_h_coords.flatten()
    grid_w = grid_w_coords.flatten()

    emb_d = torch.zeros(num_patches, embed_dim)
    emb_h = torch.zeros(num_patches, embed_dim)
    emb_w = torch.zeros(num_patches, embed_dim)

    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

    emb_d[:, 0::2] = torch.sin(grid_d[:, None] * div_term)
    emb_d[:, 1::2] = torch.cos(grid_d[:, None] * div_term)

    emb_h[:, 0::2] = torch.sin(grid_h[:, None] * div_term)
    emb_h[:, 1::2] = torch.cos(grid_h[:, None] * div_term)

    emb_w[:, 0::2] = torch.sin(grid_w[:, None] * div_term)
    emb_w[:, 1::2] = torch.cos(grid_w[:, None] * div_term)

    sinusoid_table = emb_d + emb_h + emb_w # Summing the positional encodings
    return sinusoid_table

class ViT3D(nn.Module):
    """ Vision Transformer for 3D data """
    def __init__(self, volume_size=(64,64,64), patch_size=(8,8,8), in_chans=1, num_classes=0, # num_classes=0 for MAE encoder
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 global_pool=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.patch_size = to_3tuple(patch_size)

        self.patch_embed = PatchEmbed3D(volume_size, self.patch_size, in_chans, embed_dim)
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
        # Initialize positional embedding with truncated normal, as is standard for learnable embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
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

class MaskedAutoencoderViT3D(nn.Module):
    """ Masked Auto-Encoder with VisionTransformer backbone
    """
    def __init__(self, volume_size=(64,64,64), patch_size=(8,8,8), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_ratio=0.75):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans
        self.volume_size = volume_size

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ViT3D(
            volume_size=volume_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

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
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :] # Add pos embed, excluding CLS token part

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :] # CLS token + its pos embed
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        # x is [N, 1 (cls) + K (kept), D_dec_emb]
        # ids_restore is [N, L_orig]
        # mask_tokens is [N, L_orig - K, D_dec_emb]
        
        # Remove CLS token from x for unshuffle, then add back
        x_no_cls = x[:, 1:, :]
        x_ = torch.cat([x_no_cls, mask_tokens], dim=1) # [N, L_orig, D_dec_emb]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        x = torch.cat([x[:, :1, :], x_], dim=1) # Prepend CLS token: [N, 1 + L_orig, D_dec_emb]
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :] # Remove CLS token
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, D, H, W]
        pred: [N, L, p*p*p*C]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        
        patch_stats = None
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var.add(1.e-6).sqrt())
            # Store statistics for denormalization during visualization
            patch_stats = (mean, var)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, patch_stats

    def forward(self, imgs, mask_ratio=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio if mask_ratio is not None else self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, patch_stats = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, patch_stats

MaskedAutoencoderViT = MaskedAutoencoderViT3D
ViT = ViT3D 

# --- Factory functions for MAE ViT 3D models ---

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
        embed_dim=1024,         # Large: 1024
        depth=24,               # Large: 24
        num_heads=16,           # Large: 16
        decoder_embed_dim=768,  # Increased from 512 to 768
        decoder_depth=12,       # Increased from 8 to 12
        decoder_num_heads=12,   # Increased from 16 to 12 (to match decoder_embed_dim)
        mlp_ratio=4, **kwargs
    )
    return model