# torch_only_lib.py
import torch, torch.nn as nn, torch.nn.functional as F

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
