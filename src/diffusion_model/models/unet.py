from dataclasses import dataclass
from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    num_heads: int = 4
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_classes: Optional[int] = None
    dropout: float = 0.0
    use_scale_shift_norm: bool = True

    @property
    def time_emb_dim(self) -> int:
        return self.base_channels * 4

def get_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """
    Embedding sinusoïdal des timesteps (comme dans DDPM/ADM).
    timesteps : [B]
    dim       : dimension de l'embedding
    """
    half = dim // 2
    device = timesteps.device
    timesteps = timesteps.float()

    exponents = torch.arange(half, device=device).float() / half
    freqs = torch.exp(-math.log(max_period) * exponents)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb

class AttentionBlock(nn.Module):
    """
    Self-attention 2D classique (channels-last en flatten spatial).
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(b, c, h * w)  # [B, C, HxW]

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def to_heads(t):
            # [B, C, N] -> [B, heads, head_dim, N]
            return t.view(b, self.num_heads, self.head_dim, h * w)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h * w)
        out = self.proj(out)
        out = out.view(b, c, h, w)

        return residual + out

class ResBlock(nn.Module):
    """
    ResNet block + embedding de temps + (optionnel) self-attention.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        config: ModelConfig,
        use_attention: bool = False,
        num_groups: int = 32,
    ):
        super().__init__()

        time_emb_dim = config.time_emb_dim
        self.use_scale_shift_norm = config.use_scale_shift_norm

        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        if self.use_scale_shift_norm:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_ch * 2)
        else:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # skip connection si changement de dimension
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        # attention optionnelle
        self.attn = AttentionBlock(out_ch, config.num_heads, num_groups) if use_attention else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        t_emb: [B, time_emb_dim]
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # injection de l'embedding de temps
        t = self.time_emb_proj(t_emb)  # [B, out_ch] ou [B, 2*out_ch]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(t, 2, dim=1)
            h = self.norm2(h)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = self.norm2(h)
            h = h + t[:, :, None, None]

        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip(x)

        if self.attn is not None:
            h = self.attn(h)

        return h

class Downsample(nn.Module):
    """
    Downsample par conv stride 2.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample par interpolation + conv 3x3.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class UNetModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.in_conv = nn.Conv2d(
            config.in_channels,
            config.base_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(config.base_channels, config.time_emb_dim),
            nn.SiLU(),
            nn.Linear(config.time_emb_dim, config.time_emb_dim),
        )

        if config.num_classes is not None:
            self.class_embedding = nn.Embedding(config.num_classes, config.time_emb_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = config.base_channels
        curr_res = config.image_size
        skip_channels = []

        for level, mult in enumerate(config.channel_mult):
            out_ch = config.base_channels * mult
            blocks = nn.ModuleList()

            use_attn = curr_res in config.attention_resolutions

            for _ in range(config.num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, config, use_attention=use_attn))
                ch = out_ch
                skip_channels.append(ch)

            self.down_blocks.append(blocks)

            if level != len(config.channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                curr_res //= 2

        use_attn_mid = curr_res in config.attention_resolutions
        self.mid_block1 = ResBlock(ch, ch, config, use_attention=use_attn_mid)
        self.mid_block2 = ResBlock(ch, ch, config, use_attention=use_attn_mid)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(config.channel_mult))):
            out_ch = config.base_channels * mult
            blocks = nn.ModuleList()
            use_attn = curr_res in config.attention_resolutions

            for _ in range(config.num_res_blocks):
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, config, use_attention=use_attn))
                ch = out_ch

            self.up_blocks.append(blocks)

            if level != 0:
                self.upsamples.append(Upsample(ch))
                curr_res *= 2

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, config.in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x         : [B, C, H, W]
        timesteps : [B] (ints ou floats)
        y         : [B] (labels de classe, optionnel)
        """
        # embedding temps
        t_emb = get_timestep_embedding(timesteps, self.config.base_channels)
        t_emb = self.time_mlp(t_emb)

        # class-cond (si dispo)
        if self.class_embedding is not None and y is not None:
            t_emb = t_emb + self.class_embedding(y)

        # entrée
        h = self.in_conv(x)
        hs = []

        # down path
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
                hs.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        # bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # up path
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        # sortie
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)