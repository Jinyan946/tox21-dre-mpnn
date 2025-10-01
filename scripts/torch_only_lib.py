# torch_only_lib.py
import torch, torch.nn as nn, torch.nn.functional as F
# --- add near the top ---
import math


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: shape [B] in [0,1]. Returns [B, dim].
    """
    # make frequencies
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=t.device)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0,1))
    return emb

class TimeConditionedRatioMLP(nn.Module):
    """
    s_theta(x, t) -> scalar (time-conditioned log-density time-derivative)
    """
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 2, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        layers = []
        d = input_dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, D], t: [B] in [0,1]
        if x.dtype != torch.float32: x = x.float()
        if t.dtype != torch.float32: t = t.float()
        te = sinusoidal_time_embedding(t, self.t_dim)  # [B, t_dim]
        out = self.net(torch.cat([x, te], dim=-1))
        return out.squeeze(-1)  # [B]


class RatioMLP(nn.Module):
    """Outputs a scalar logit ~ log r(x) + const."""
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # ensure float32 and return 1-D logits
        if x.dtype != torch.float32:
            x = x.float()
        out = self.net(x)            # [B, 1]
        return out.squeeze(-1)       # [B]

def build_model(input_dim: int, hidden: int = 256, depth: int = 2):
    return RatioMLP(input_dim, hidden, depth)

@torch.no_grad()
def estimate_log_ratio(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return predicted log-density ratio (up to an additive constant)."""
    model.eval()
    return model(x)  # logits are fine as log-ratio scores

def logistic_nce_loss(model, xp, xq):
    # 1-D logits to match 1-D targets
    pos = model(xp).squeeze(-1)      # [B]
    neg = model(xq).squeeze(-1)      # [B]
    logits = torch.cat([pos, neg], dim=0)                    # [2B]
    targets = torch.cat([torch.ones_like(pos),               # [2B]
                         torch.zeros_like(neg)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, targets)
