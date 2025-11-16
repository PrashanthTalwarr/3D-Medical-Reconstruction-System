import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from local.cuda_ops import cuda_ops

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MultiPlaneReconstruction(nn.Module):
    """
    3D Vision: Multi-plane reconstruction
    Processes axial, coronal, and sagittal planes separately
    """
    def __init__(self, channels):
        super().__init__()
        self.axial_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.coronal_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.sagittal_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fusion = nn.Conv3d(channels * 3, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # Extract middle planes
        axial = self.axial_conv(x[:, :, d//2, :, :])  # Z plane
        coronal = self.coronal_conv(x[:, :, :, h//2, :].permute(0, 1, 3, 2))  # Y plane
        sagittal = self.sagittal_conv(x[:, :, :, :, w//2])  # X plane
        
        # Broadcast to 3D
        axial_3d = axial.unsqueeze(2).expand(-1, -1, d, -1, -1)
        coronal_3d = coronal.unsqueeze(3).expand(-1, -1, -1, h, -1)
        sagittal_3d = sagittal.unsqueeze(4).expand(-1, -1, -1, -1, w)
        
        # Fuse features from all planes
        fused = torch.cat([axial_3d, coronal_3d, sagittal_3d], dim=1)
        output = self.fusion(fused)
        
        return self.norm(output) + x


class Attention3D(nn.Module):
    """3D Self-Attention mechanism for capturing long-range dependencies"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        
        # Attention scores
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) * scale, dim=-1)
        
        # Apply attention
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(b, c, d, h, w)
        
        return self.proj(out) + x


class ResidualBlock3D(nn.Module):
    """3D Residual block with time embedding injection"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, use_multiplane=False):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Optional multi-plane reconstruction (3D Vision component)
        self.multiplane = MultiPlaneReconstruction(out_channels) if use_multiplane else None
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = cuda_ops.fused_silu(h)  # Use custom CUDA SiLU
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = cuda_ops.fused_silu(self.time_mlp(t_emb))
        h = h + time_emb[:, :, None, None, None]
        
        h = self.norm2(h)
        h = cuda_ops.fused_silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Optional multi-plane processing
        if self.multiplane:
            h = self.multiplane(h)
        
        return h + self.shortcut(x)


class DownBlock3D(nn.Module):
    """Downsampling block for encoder"""
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False, use_multiplane=False):
        super().__init__()
        self.res1 = ResidualBlock3D(in_channels, out_channels, time_emb_dim, use_multiplane=use_multiplane)
        self.res2 = ResidualBlock3D(out_channels, out_channels, time_emb_dim)
        self.attention = Attention3D(out_channels) if use_attention else None
        self.downsample = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        
        if self.attention:
            x = self.attention(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock3D(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        self.res1 = ResidualBlock3D(in_channels * 2, out_channels, time_emb_dim)
        self.res2 = ResidualBlock3D(out_channels, out_channels, time_emb_dim)
        self.attention = Attention3D(out_channels) if use_attention else None
    
    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        
        if self.attention:
            x = self.attention(x)
        
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for Diffusion Models
    Combines: PyTorch + 3D Vision + Attention mechanisms
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, time_emb_dim=128):
        super().__init__()
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down1 = DownBlock3D(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock3D(base_channels * 2, base_channels * 4, time_emb_dim, 
                                  use_attention=True, use_multiplane=True)
        self.down3 = DownBlock3D(base_channels * 4, base_channels * 8, time_emb_dim, 
                                  use_attention=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(base_channels * 8, base_channels * 8, time_emb_dim),
            Attention3D(base_channels * 8),
            ResidualBlock3D(base_channels * 8, base_channels * 8, time_emb_dim),
        )
        
        # Decoder
        self.up1 = UpBlock3D(base_channels * 8, base_channels * 4, time_emb_dim, use_attention=True)
        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2, time_emb_dim, use_attention=True)
        self.up3 = UpBlock3D(base_channels * 2, base_channels, time_emb_dim)
        
        # Final output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 1),
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        t_emb = self.time_mlp(timesteps)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder with skip connections
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock3D):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        
        # Decoder with skip connections
        x = self.up1(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up3(x, skip1, t_emb)
        
        # Output
        return self.conv_out(x)