import torch
import torch.nn as nn

__all__ = ["PatchEmbedding"]

class PatchEmbedding(nn.Module):
    """Converts 1D signal into patch embeddings for Transformer input."""
    def __init__(self, in_channels: int, patch_size: int, emb_dim: int, seq_length: int):
        super().__init__()
        assert seq_length % patch_size == 0, "Signal length must be divisible by patch size."
        self.num_patches = seq_length // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, L]
        Returns:
            Tensor of shape [B, num_patches, emb_dim]
        """
        x = self.proj(x)  # [B, emb_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, emb_dim]
        return x
