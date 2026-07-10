"""MDM-style conditional Transformer denoiser.

The paper states the generator follows the MDM architecture (Tevet et al.,
ICLR 2023) as adapted by PARC (Xu et al., SIGGRAPH 2025); layer counts and
hidden sizes are not published. Defaults here follow the official MDM
implementation (8 layers, 4 heads, d_model=512, ff=1024, GELU, dropout 0.1,
MIT license) scaled down per config for single-GPU training.

Token sequence: [cond, past_0..past_{P-1}, future_0..future_{H-1}] with
sinusoidal positional encoding; the cond token is the sum of timestep,
heading and terrain embeddings. Conditions can be independently masked
(replaced by learned null embeddings) for classifier-free guidance.

All operations are standard (Linear/LayerNorm/TransformerEncoder); nothing
here is known to block ONNX/TensorRT export. torch.onnx export of
nn.TransformerEncoder is supported; verify at export time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Classic sinusoidal embedding, used for both positions and timesteps."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (...,) -> (..., dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class MotionDiffusionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        past_frames: int = 2,
        future_frames: int = 25,
        terrain_dim: int = 121,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.terrain_dim = terrain_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.past_proj = nn.Linear(feature_dim, d_model)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.heading_embed = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.terrain_embed = nn.Sequential(
            nn.Linear(terrain_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # Learned null embeddings for condition masking / classifier-free guidance.
        self.null_past = nn.Parameter(torch.zeros(1, 1, d_model))
        self.null_heading = nn.Parameter(torch.zeros(1, d_model))
        self.null_terrain = nn.Parameter(torch.zeros(1, d_model))

        self.pos_embedding = SinusoidalEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, feature_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        past: torch.Tensor,
        heading: torch.Tensor,
        terrain: torch.Tensor,
        drop_past: torch.Tensor | None = None,
        drop_heading: torch.Tensor | None = None,
        drop_terrain: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the denoising target for noised future frames.

        Args:
            x_t: (B, H, D) noised future features (normalized space).
            t: (B,) integer diffusion timesteps.
            past: (B, P, D) conditioning frames (normalized space).
            heading: (B, 2) unit target heading.
            terrain: (B, terrain_dim) height scan.
            drop_*: optional (B,) bool masks; True replaces the condition with
                its learned null embedding (conditional masking / CFG).
            seq_mask: optional (B, H) bool; True = valid future frame. Padded
                positions are excluded from attention.
        Returns:
            (B, H, D) prediction (x0 or epsilon depending on the diffusion
            parameterization configured outside the model).
        """
        bsz = x_t.shape[0]

        past_tok = self.past_proj(past)
        if drop_past is not None:
            past_tok = torch.where(
                drop_past.view(bsz, 1, 1), self.null_past.expand_as(past_tok), past_tok
            )
        head_emb = self.heading_embed(heading)
        if drop_heading is not None:
            head_emb = torch.where(drop_heading.view(bsz, 1), self.null_heading.expand_as(head_emb), head_emb)
        terr_emb = self.terrain_embed(terrain)
        if drop_terrain is not None:
            terr_emb = torch.where(drop_terrain.view(bsz, 1), self.null_terrain.expand_as(terr_emb), terr_emb)

        cond_tok = (self.time_embed(t) + head_emb + terr_emb).unsqueeze(1)  # (B, 1, d)
        fut_tok = self.input_proj(x_t)

        tokens = torch.cat([cond_tok, past_tok, fut_tok], dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)

        key_padding = None
        if seq_mask is not None:
            prefix = torch.ones(bsz, 1 + past.shape[1], dtype=torch.bool, device=x_t.device)
            key_padding = ~torch.cat([prefix, seq_mask], dim=1)  # True = ignore

        out = self.encoder(tokens, src_key_padding_mask=key_padding)
        return self.output_proj(out[:, 1 + past.shape[1] :])
