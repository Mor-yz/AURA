"""Public AURA network definition.

This module contains only the actuator-degradation identification network used
by AURA.  It does not include robot transport, policies, datasets, or
deployment control loops.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalConvEncoder(nn.Module):
    """Dilated causal convolutions over one motor's history."""
    def __init__(self, in_channels=4, hidden=64, out_channels=128,
                 kernel_size=3, window=30):
        super().__init__()
        if window < 1 or kernel_size < 2:
            raise ValueError("window >= 1 and kernel_size >= 2 are required")
        n_layers = max(1, math.ceil(math.log2((window - 1) / (kernel_size - 1) + 1)))
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else hidden,
                      out_channels if i == n_layers - 1 else hidden,
                      kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            padding = (layer.kernel_size[0] - 1) * layer.dilation[0]
            x = F.elu(layer(F.pad(x, (padding, 0))))
        return x[:, :, -1]


class CrossMotorAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        attended, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class AURA(nn.Module):
    """Map ``[batch, motors, history, features]`` to alpha and kappa."""
    def __init__(self, n_motors=12, window=30, feat_dim=4, d_model=128,
                 n_heads=4, d_embed=32, tau_max=None):
        super().__init__()
        if tau_max is None:
            tau_max = torch.ones(n_motors)
        limits = torch.as_tensor(tau_max, dtype=torch.float32)
        if limits.shape != (n_motors,) or (limits <= 0).any():
            raise ValueError("tau_max must contain one positive value per motor")
        self.n_motors, self.window, self.feat_dim = n_motors, window, feat_dim
        self.d_model = d_model
        self.register_buffer("tau_max", limits)
        self.temporal_enc = TemporalConvEncoder(feat_dim, out_channels=d_model,
                                                window=window)
        self.cross_motor = CrossMotorAttention(d_model, n_heads)
        self.token_embed = nn.Embedding(n_motors, d_model)
        self.motor_embed = nn.Embedding(n_motors, d_embed)
        self.predictor = nn.Sequential(nn.Linear(d_model + d_embed, 128), nn.ELU(),
                                       nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 2))

    def forward(self, x):
        expected = (self.n_motors, self.window, self.feat_dim)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{self.n_motors},{self.window},{self.feat_dim}]")
        batch = x.shape[0]
        temporal = x.permute(0, 1, 3, 2).reshape(batch * self.n_motors,
                                                  self.feat_dim, self.window)
        h = self.temporal_enc(temporal).reshape(batch, self.n_motors, self.d_model)
        idx = torch.arange(self.n_motors, device=x.device)
        h = self.cross_motor(h + self.token_embed(idx).unsqueeze(0))
        latent = h.mean(dim=1).unsqueeze(1).expand(-1, self.n_motors, -1)
        embed = self.motor_embed(idx).unsqueeze(0).expand(batch, -1, -1)
        out = self.predictor(torch.cat((latent, embed), dim=-1))
        return torch.sigmoid(out[..., 0]), F.softplus(out[..., 1]) + 1e-6

    def reconstruct_torque(self, tau_cmd, alpha, kappa):
        return alpha * self.tau_max * torch.tanh(
            tau_cmd / (self.tau_max * kappa.clamp_min(1e-6)))

