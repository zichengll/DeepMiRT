#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false
"""
Full miRNA-target model: shared RNA-FM encoder + Cross-Attention + MLP classifier head.

Complete data flow (with tensor shapes):

    miRNA tokens (B, M_tok)  ---> [RNA-FM Encoder] ---> miRNA_emb  (B, M, D) ---┐
                                                                                  |
                                                                                  v
    target tokens (B, T_tok) ---> [RNA-FM Encoder] ---> target_emb (B, T, D) --> [Cross-Attention]
                                                                                  |
                                                                                  v
                                                                        cross_out (B, T, D)
                                                                                  |
                                                                                  v
                                                                      masked mean pool
                                                                                  |
                                                                                  v
                                                                              (B, D)
                                                                                  |
                                                                                  v
                                                                            [MLP Head]
                                                                                  |
                                                                                  v
                                                                              logits
                                                                              (B, 1)

Where D is automatically inferred from RNA-FM (typically 640) to avoid hard-coding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .classifier import MLPClassifier
from .cross_attention import CrossAttentionBlock
from .rnafm_encoder import RNAFMEncoder


class MiRNATargetModel(nn.Module):
    """End-to-end model for miRNA-target binary classification."""

    def __init__(
        self,
        freeze_backbone: bool = True,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        classifier_hidden: Sequence[int] | None = None,
        dropout: float = 0.3,
        # Ablation parameters
        ablation_interaction: str = "cross_attention",
        ablation_pooling: str = "mean",
        ablation_encoder: str = "shared",
        ablation_random_init: bool = False,
        ablation_classifier: str = "mlp",
    ) -> None:
        super().__init__()
        hidden_dims = list(classifier_hidden) if classifier_hidden is not None else [256, 64]
        self.ablation_interaction = ablation_interaction
        self.ablation_pooling = ablation_pooling

        self.encoder = RNAFMEncoder(
            freeze_backbone=freeze_backbone,
            random_init=ablation_random_init,
        )
        embed_dim = self.encoder.embed_dim

        # Ablation B2: separate encoder for target
        self.encoder_target = None
        if ablation_encoder == "separate":
            self.encoder_target = RNAFMEncoder(
                freeze_backbone=freeze_backbone,
                random_init=ablation_random_init,
            )

        # Ablation A1: no cross-attention (concat mode) — skip building the module
        if ablation_interaction == "concat":
            self.cross_attention = None
            classifier_input_dim = embed_dim * 2
        else:
            self.cross_attention = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=cross_attn_heads,
                dropout=dropout * 0.33,
                num_layers=cross_attn_layers,
            )
            classifier_input_dim = embed_dim

        self.classifier = MLPClassifier(
            input_dim=classifier_input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            linear_only=(ablation_classifier == "linear"),
        )

    def forward(
        self,
        mirna_tokens: Tensor,
        target_tokens: Tensor,
        attention_mask_mirna: Tensor | None = None,
        attention_mask_target: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass with ablation-aware branching.

        Default path: encode -> cross-attention (target=Q, miRNA=KV) -> mean pool -> MLP
        Ablation variants modify interaction, pooling, or encoding steps.
        """
        # Step 1-2: Encode miRNA and target
        mirna_emb = self.encoder(mirna_tokens)
        target_encoder = self.encoder_target if self.encoder_target is not None else self.encoder
        target_emb = target_encoder(target_tokens)

        # --- Ablation A1: concat mode (no cross-attention) ---
        if self.ablation_interaction == "concat":
            mirna_pooled = self._masked_mean_pool(mirna_emb, attention_mask_mirna)
            target_pooled = self._masked_mean_pool(target_emb, attention_mask_target)
            pooled = torch.cat([mirna_pooled, target_pooled], dim=-1)
            return self.classifier(pooled)

        # Step 3: Build key_padding_mask (True=ignore for PyTorch MHA)
        if self.ablation_interaction == "reverse_qk":
            # Ablation A4: miRNA=Q, target=KV (reversed)
            key_padding_mask = None
            if attention_mask_target is not None:
                key_padding_mask = attention_mask_target == 0
            cross_out = self.cross_attention(
                query=mirna_emb,
                key_value=target_emb,
                key_padding_mask=key_padding_mask,
            )
            # Pool over miRNA dimension (since miRNA is the query)
            pooled = self._pool(cross_out, attention_mask_mirna)
        else:
            # Default: target=Q, miRNA=KV
            key_padding_mask = None
            if attention_mask_mirna is not None:
                key_padding_mask = attention_mask_mirna == 0
            cross_out = self.cross_attention(
                query=target_emb,
                key_value=mirna_emb,
                key_padding_mask=key_padding_mask,
            )
            # Pool over target dimension
            pooled = self._pool(cross_out, attention_mask_target)

        return self.classifier(pooled)

    def _pool(self, hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Apply the configured pooling strategy."""
        if self.ablation_pooling == "max":
            return self._masked_max_pool(hidden, attention_mask)
        elif self.ablation_pooling == "cls":
            return hidden[:, 0, :]
        else:
            return self._masked_mean_pool(hidden, attention_mask)

    @staticmethod
    def _masked_mean_pool(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Masked mean pooling over the sequence dimension."""
        if attention_mask is None:
            mask = torch.ones(hidden.size(0), hidden.size(1), 1,
                              device=hidden.device, dtype=hidden.dtype)
        else:
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    @staticmethod
    def _masked_max_pool(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Masked max pooling over the sequence dimension."""
        if attention_mask is None:
            mask = torch.ones(hidden.size(0), hidden.size(1), 1,
                              device=hidden.device, dtype=hidden.dtype)
        else:
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        # Set padding positions to large negative value before max
        masked = hidden * mask + (1 - mask) * (-1e9)
        return masked.max(dim=1)[0]
