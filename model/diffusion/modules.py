import math

import torch
from torch import nn

from model.utils import modulate

class Attention(nn.Module):
    """ Modified from PyTorch Image Models timm.models.vision_transformer.Attention
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L59
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, N, C]
        :param attention_mask: [B, 1, 1, N]
        :returns: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # shape = (B, num_heads, N, N)
        attn += attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LearnedSinusodialPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            approx_gelu(),
            nn.Dropout(block_kwargs["proj_drop"]),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(block_kwargs["proj_drop"])
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask):
        """
        :param x: [B, max_N, hidden_dim]
        :param c: [B, max_N, time_dim] = [B, max_N, hidden_dim]
        :param attention_mask: [B, 1, 1, max_N]
        :returns: [B, max_N, hidden_dim]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1) # 6 * [B, max_N, hidden_dim]
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attention_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, latent_dim, enhance_dit=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if enhance_dit:
            self.linear = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, latent_dim, bias=True)
            )
        else:
            self.linear = nn.Linear(hidden_size, latent_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask):
        """
        :param x: [B, max_N, hidden_dim]
        :param c: [B, max_N, time_dim] = [B, max_N, hidden_dim]
        :param attention_mask: [B, max_N]
        :returns: [B, max_N, latent_dim]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)  # 2 * [B, max_N, hidden_dim]
        x = modulate(self.norm_final(x), shift, scale)  # [B, max_N, hidden_dim]
        x = x * attention_mask[:, :, None]
        x = self.linear(x)
        return x
