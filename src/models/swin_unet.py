"""
Swin-Unet: Transformer-based U-Net with Swin Transformer blocks for image denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        residual = x  # keep input for residual skip
        residual = x  # keep input for residual skip
        residual = x  # keep resized input for residual skip
        residual = x  # keep resized input for residual skip
        x = self.proj(x)  # B, embed_dim, H/patch_size, W/patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer (Downsampling)"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        
        # Padding if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer (Upsampling)"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = self.expand(x)
        x = x.view(B, H, W, C * 2)
        
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Window partition
        x_windows = self.window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer"""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])
        
        self.downsample = downsample
    
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2
        
        return x, H, W


class SwinUnet(nn.Module):
    """Swin-Unet for image denoising"""
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        out_chans=3,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.
    ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = None
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(dim=int(embed_dim * 2 ** i_layer))
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                downsample=downsample
            )
            self.encoder_layers.append(layer)
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (self.num_layers - 2 - i_layer)),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                downsample=None
            )
            self.decoder_layers.append(layer)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            upsample = PatchExpanding(
                dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            )
            self.upsample_layers.append(upsample)
        
        # Skip connection fusion
        self.skip_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            skip = nn.Linear(
                int(embed_dim * 2 ** (self.num_layers - 2 - i_layer)) * 2,
                int(embed_dim * 2 ** (self.num_layers - 2 - i_layer))
            )
            self.skip_layers.append(skip)
        
        # Final patch expanding
        self.final_expand = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim // 4, out_chans, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # Patch embedding
        x = self.patch_embed(x)
        H_p, W_p = H // self.patch_size, W // self.patch_size
        
        # Encoder
        skip_connections = []
        for i, layer in enumerate(self.encoder_layers):
            skip_connections.append((x, H_p, W_p))
            x, H_p, W_p = layer(x, H_p, W_p)
        
        # Decoder
        for i, (decoder_layer, upsample_layer, skip_layer) in enumerate(
            zip(self.decoder_layers, self.upsample_layers, self.skip_layers)
        ):
            # Upsample
            x = upsample_layer(x, H_p, W_p)
            H_p, W_p = H_p * 2, W_p * 2
            
            # Skip connection
            skip_x, skip_H, skip_W = skip_connections[-(i + 2)]
            x = torch.cat([x, skip_x], dim=-1)
            x = skip_layer(x)
            
            # Decoder block
            x, H_p, W_p = decoder_layer(x, H_p, W_p)
        
        # Reshape to image
        x = x.transpose(1, 2).view(B, self.embed_dim, H_p, W_p)
        
        # Final expansion and output
        x = self.final_expand(x)
        x = self.output_proj(x)

        # Residual connection to preserve content
        if residual.shape[2:] != x.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.clamp(x + residual, 0.0, 1.0)
        
        return x


def test_model():
    """Test the Swin-Unet model"""
    model = SwinUnet(img_size=256, patch_size=4, in_chans=3, out_chans=3)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    test_model()
