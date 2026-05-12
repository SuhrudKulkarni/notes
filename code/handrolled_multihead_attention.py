import math

import einops
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    "SelfAttention with causal mask."

    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        var_scale = 0.02
        rotation_var_scale = 1.0 / math.sqrt(2 * num_layers)
        # W_Q/W_K/W_V : (E, H * D)
        self.W_QKV = nn.Parameter(
            torch.randn(self.embedding_dim, self.embedding_dim * 3) * var_scale
        )
        self.W_O = nn.Parameter(
            torch.randn(self.embedding_dim, self.embedding_dim) * rotation_var_scale
        )

    def forward(self, X: torch.Tensor):
        # X: (B, L, E)
        assert len(X.shape) == 3
        _, L, _ = X.shape
        assert X.shape[-1] == self.embedding_dim

        # Q, K, V: (B, H, L, D)
        QKV = torch.einsum("ble, e...-> bl...", X, self.W_QKV)
        Q, K, V = einops.rearrange(
            QKV, "b l (three h d) -> three b h l d", h=self.num_heads, three=3
        )

        # (B, H, L, L)
        masked_QKt = (
            torch.einsum("bhld,bhmd->bhlm", Q, K) / math.sqrt(self.head_dim)
            +
            # mask is const, can impl once at class level
            torch.triu(torch.full((L, L), float("-inf"), device=X.device), diagonal=1)
        )
        QKt_exp = torch.exp(masked_QKt - masked_QKt.max(dim=-1, keepdim=True).values)
        softmax = QKt_exp / QKt_exp.sum(dim=-1, keepdim=True)  # (B, H, L, L)

        # (B, L, E)
        attention = einops.rearrange(
            torch.einsum("bhlm,bhmd->bhld", softmax, V),
            "b h l d -> b l (h d)",
            h=self.num_heads,
        )

        # (B, L, E)
        return torch.einsum("ble,ej->blj", attention, self.W_O)
