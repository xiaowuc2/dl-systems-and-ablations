"""Side-by-side DDPM vs. Rectified Flow Matching training objective."""
import torch


def rectified_flow_loss(model, x0: torch.Tensor) -> torch.Tensor:
    """Linear interpolation x_t = (1 - t) * x0 + t * x1; target velocity v = x1 - x0."""
    x1 = torch.randn_like(x0)
    t = torch.rand(x0.size(0), 1, device=x0.device)
    xt = (1.0 - t) * x0 + t * x1
    v_pred = model(xt, t)
    return ((v_pred - (x1 - x0)) ** 2).mean()
