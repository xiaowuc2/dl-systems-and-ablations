"""PyTorch 2.x torch.compile & Profiler trace template."""
import torch
import torch.nn as nn


class FFNBlock(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        return x + self.net(x)


if __name__ == "__main__":
    m = FFNBlock()
    x = torch.randn(32, 128, 512)
    y = m(x)
    print("Output shape:", y.shape)
