"""Rank r={4,8,16,32} SVD spectrum analysis script."""
import torch
from lora_dora_from_scratch import LoRALinear
import torch.nn as nn

if __name__ == "__main__":
    layer = LoRALinear(nn.Linear(256, 256), rank=8)
    svals = layer.delta_weight_svd()
    print("Top singular values:", svals[:8])
