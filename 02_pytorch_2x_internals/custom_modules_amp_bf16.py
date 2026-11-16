"""Mixed-precision (fp32 vs bf16/fp16) latency and memory benchmark."""
import time
import torch
import torch.nn as nn


def benchmark_amp(dim=1024, batch_size=64, steps=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)).to(device)
    x = torch.randn(batch_size, dim, device=device)

    for use_amp in [False, True]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                out = model(x).sum()
            out.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000 / steps
        print(f"Device={device} | AMP={use_amp} | Avg Step={dt:.3f} ms")


if __name__ == "__main__":
    benchmark_amp()
