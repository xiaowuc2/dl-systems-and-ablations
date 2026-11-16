"""Modular GPT implementation with toggleable RoPE, RMSNorm, SwiGLU, and GQA."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6       # Set < n_head for Grouped-Query Attention (GQA)
    n_embd: int = 384
    norm_type: str = "rmsnorm"  # "layernorm" | "rmsnorm"
    mlp_type: str = "swiglu"    # "gelu" | "swiglu"
    use_rope: bool = True
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
