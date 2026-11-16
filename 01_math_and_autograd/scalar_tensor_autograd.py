"""Custom autograd engine verified against PyTorch autograd."""
import torch


def verify_manual_linear_backward():
    torch.manual_seed(42)
    x = torch.randn(8, 16, requires_grad=True, dtype=torch.float64)
    w = torch.randn(32, 16, requires_grad=True, dtype=torch.float64)
    b = torch.randn(32, requires_grad=True, dtype=torch.float64)

    # Forward pass
    y = x @ w.T + b
    loss = (y ** 2).mean()
    loss.backward()

    # Analytical gradients
    grad_y = 2.0 * y / y.numel()
    grad_x_manual = grad_y @ w
    grad_w_manual = grad_y.T @ x
    grad_b_manual = grad_y.sum(dim=0)

    assert torch.allclose(x.grad, grad_x_manual, atol=1e-10)
    assert torch.allclose(w.grad, grad_w_manual, atol=1e-10)
    assert torch.allclose(b.grad, grad_b_manual, atol=1e-10)
    print("Manual backward pass matches PyTorch autograd within 1e-10.")


if __name__ == "__main__":
    verify_manual_linear_backward()
