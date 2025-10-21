# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from fm4ar.nn.patch_embeds import PatchEmbedding
from fm4ar.nn.attentions import SABlock
from fm4ar.nn.mlp import MLPBlock

__all__ = ["TransformerEncoder"]
   
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Args:
        hidden_size: dimension of hidden layer.
        mlp_dim: dimension of feedforward layer.
        num_heads: number of attention heads.
        dropout_rate: faction of the input units to drop.
        qkv_bias: apply bias term for the qkv linear layer
        causal: whether to use causal attention.
        sequence_length: if causal is True, it is necessary to specify the sequence length.
        with_cross_attention: Whether to use cross attention for conditioning.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        with_cross_attention: bool = False,
        use_flash_attention: bool = False,
        use_adaLN_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.with_cross_attention = with_cross_attention
        self.use_adaLN_modulation = use_adaLN_modulation
        
        # cannot use adaln modulation if cross attention is used
        assert not (use_adaLN_modulation and with_cross_attention), "adaLN_modulation cannot be used with cross attention."


        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            causal=causal,
            sequence_length=sequence_length,
            use_flash_attention=use_flash_attention,
        )

        self.norm2 = None
        self.cross_attn = None
        if self.with_cross_attention:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.cross_attn = SABlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                with_cross_attention=with_cross_attention,
                causal=False,
                use_flash_attention=use_flash_attention,
            )


        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        
        self.adaLN_modulation = None
        if use_adaLN_modulation:

            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(min(hidden_size, 1024), 6 * hidden_size, bias=True),
            ) 

            # init zero
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is not None and self.use_adaLN_modulation:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(context).chunk(6, dim=1)
            )
            x = x + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa), mask=mask
            )
        else:
            x = x + self.attn(self.norm1(x), mask=mask)

        if self.with_cross_attention and context is not None:
            x = x + self.cross_attn(self.norm2(x), mask=mask, context=context)
        
        if self.use_adaLN_modulation and context is not None:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm3(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.mlp(self.norm3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        patch_size: int,
        in_channels: int,
        attn_layers_dim: int,
        attn_layers_depth: int,
        attn_layers_heads: int,
        embedding_dropout_rate: float = 0.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        assert seq_length % patch_size == 0, "Signal length must be divisible by patch size."
        self.num_patches = seq_length // patch_size
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.patch_size = patch_size
        # token encoding

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=attn_layers_dim,
            seq_length=seq_length
        )

        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches, attn_layers_dim), requires_grad=True) # patch position 
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=attn_layers_dim,
                mlp_dim=attn_layers_dim * 4,
                num_heads=attn_layers_heads,
                dropout_rate=0.0,
                qkv_bias=False,
                causal=False,
                sequence_length=self.num_patches,
                with_cross_attention=False,
                use_flash_attention=use_flash_attention,
            )
            for _ in range(attn_layers_depth)
        ])

        # self.norm = nn.LayerNorm(attn_layers_dim)
        # self.head = nn.Linear(attn_layers_dim, 1)  # Regression or binary classification; customize

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, L]
        Returns:
            logits: [B, 1] or [B, num_classes]
        """

        # TODO: add support to handle different input shapes 
        x = x.view(-1, self.in_channels, self.seq_length)  # Ensure input shape is [B, C, L]
        x = self.patch_embed(x)  # [B, num_patches, D]
        x = x + self.positional_embedding
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, context=context)
        x = x.flatten(start_dim=1)
        return x

