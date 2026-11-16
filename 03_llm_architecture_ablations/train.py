"""Training harness with deterministic seeding and CSV metric logging."""
import csv
import torch
from model import GPTConfig


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(42)
    cfg = GPTConfig()
    print("Initialized GPTConfig:", cfg)
