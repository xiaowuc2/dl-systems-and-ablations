"""Multi-seed ResNet vs. ConvNeXt vs. ViT benchmark harness."""
import numpy as np
import torch


def summarize_seeds(scores: list[float]) -> str:
    arr = np.array(scores)
    return f"{arr.mean():.4f} +/- {arr.std(ddof=1):.4f}"


if __name__ == "__main__":
    print("Example 3-seed summary:", summarize_seeds([0.8812, 0.8854, 0.8798]))
