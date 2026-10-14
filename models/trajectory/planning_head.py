"""Learned trajectory planner (replaces / complements the classical baseline planner).

    BEV features (B, 256, 100, 100) + coord channels
      └─ 3 × stride-2 conv ................................ (B, 128, 13, 13) -> 169 tokens
    ego query = MLP([speed, yaw_rate, accel, one_hot(command)])   (B, 1, 128)
      └─ multi-head cross-attention (query -> BEV tokens) ...... context (B, 128)
          └─ GRU decoder, T steps: residual waypoint deltas
    waypoints_t = constant-velocity prior_t + Σ residuals ....... (B, T, 2)

Predicting a residual on top of the constant-velocity prior makes the head trainable on
the small nuScenes-mini split. The attention weights show *where* the planner looked.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.common.blocks import ConvBNReLU


class PlanningHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, horizon: int = 6, dt: float = 0.5, num_heads: int = 4,
                 num_commands: int = 3, ego_state_dim: int = 3) -> None:
        super().__init__()
        self.horizon, self.dt, self.num_commands = horizon, dt, num_commands
        self.reduce = nn.Sequential(ConvBNReLU(in_channels + 2, hidden, 3, stride=2), ConvBNReLU(hidden, hidden, 3, stride=2),
                                    ConvBNReLU(hidden, hidden, 3, stride=2))
        self.query = nn.Sequential(nn.Linear(ego_state_dim + num_commands, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.gru = nn.GRUCell(hidden + 2, hidden)
        self.out = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("state_scale", torch.tensor([10.0, 0.5, 2.0]), persistent=False)

    def forward(self, bev: torch.Tensor, ego_state: torch.Tensor, command: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = bev.shape
        ys = torch.linspace(-1, 1, h, device=bev.device, dtype=bev.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=bev.device, dtype=bev.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        tokens = self.reduce(torch.cat([bev, xs, ys], 1))
        tokens = tokens.flatten(2).transpose(1, 2).float()                     # (B, L, hidden)
        q_in = torch.cat([ego_state.float() / self.state_scale, F.one_hot(command.long(), self.num_commands).float()], 1)
        query = self.query(q_in).unsqueeze(1)                                   # (B, 1, hidden)
        ctx, attn_w = self.attn(query, tokens, tokens, need_weights=True)       # (B, 1, hidden), (B, 1, L)
        ctx = ctx.squeeze(1)
        speed = ego_state[:, 0].float().clamp(min=0.0)
        t = torch.arange(1, self.horizon + 1, device=bev.device, dtype=torch.float32) * self.dt
        prior = torch.stack([speed[:, None] * t[None], torch.zeros_like(speed[:, None] * t[None])], -1)  # (B, T, 2)
        hidden = ctx
        residual = torch.zeros(b, 2, device=bev.device)
        outs = []
        for _ in range(self.horizon):
            hidden = self.gru(torch.cat([ctx, residual], 1), hidden)
            residual = residual + self.out(hidden)
            outs.append(residual)
        waypoints = prior + torch.stack(outs, 1)
        side = int(round(tokens.shape[1] ** 0.5))
        return {"waypoints": waypoints, "prior": prior, "attention": attn_w.view(b, side, -1) if side * side == tokens.shape[1] else attn_w}
