"""Correctness & latency benchmark harness for custom kernels vs. PyTorch."""
import torch

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
