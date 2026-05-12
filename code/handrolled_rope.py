"""
Two implementations of Rotary Position Embedding (RoPE).

`apply_rope` uses even/odd slicing.
`apply_rope_matrix` builds the block-diagonal rotation matrix and applies it.
 Worse because we're instantiating a larger intermediate matrix than the output.
"""

import einops
import torch


def compute_rope_angles(positions: torch.Tensor, d: int, base: float = 10000.0) -> torch.Tensor:
    """
    positions: (T,)
    Returns angles: (T, d/2) where angles[t, i] = positions[t] / base^(2i/d).
    """
    assert d % 2 == 0
    freq = 1.0 / (base ** (torch.arange(d // 2).float() / (d // 2)))
    return positions[:, None].float() * freq[None, :]


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    x: (..., T, d)
    angles: (T, d/2). One rotation angle per (position, pair-of-dims).
    """
    rotated = torch.zeros_like(x)
    rotated[..., ::2] = (
        torch.cos(angles) * x[..., ::2] - torch.sin(angles) * x[..., 1::2]
    )
    rotated[..., 1::2] = (
        torch.sin(angles) * x[..., ::2] + torch.cos(angles) * x[..., 1::2]
    )
    return rotated


def apply_rope_matrix(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Same functionality as apply_rope. Instead of building a (d, d) block-diagonal
    rotation matrix with d/2 blocks of 2x2 rotations, keep a (T, d/2, 2, 2) tensor for each position t.
    Apply as matmul to x reshaped as (..., T, d/2, 2).
    """
    cos = torch.cos(angles) 
    sin = torch.sin(angles)

    # Per-pair rotation: 
    # [
    #   [cos, -sin],
    #   [sin, cos]
    # ]. 
    # Shape (T, d/2, 2, 2).
    rotation = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=-2,
    )

    # Turn x pairs into 2-dim vectors: (..., d) -> (..., d/2, 2)
    x_pairs = einops.rearrange(x, "... (half_d pair) -> ... half_d pair", pair=2)

    rotated_pairs = einops.einsum(
        rotation,
        x_pairs,
        "pos pair i j, ... pos pair j -> ... pos pair i",
    )

    return einops.rearrange(rotated_pairs, "... t half_d pair -> ... t (half_d pair)")
