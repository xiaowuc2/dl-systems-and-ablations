# Deep Learning Systems & Controlled Architectural Ablations

First-principles PyTorch implementations, controlled architectural ablations (LLMs, ViTs, LoRA), and custom CUDA/Triton kernels.

## Summary of Quantitative Experiments

| Module | Experiment / System | Key Metric | Result | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `01_math_and_autograd` | Scalar/Tensor Autograd & Optimizers | Max Grad Error vs. PyTorch | `< 1e-6` | Verified via finite differences |
| `02_pytorch_2x_internals` | `bf16` AMP + `torch.compile` | Throughput / Peak VRAM | `TBD` | Benchmarked on Transformer block |
| `03_llm_architecture_ablations` | RoPE + RMSNorm + SwiGLU + GQA | Val Perplexity @ Equal FLOPs | `TBD` | 3-seed controlled ablation |
| `04_peft_and_lora_spectrum` | Custom LoRA/DoRA Rank Spectrum | SVD Rank vs. Accuracy | `TBD` | Ranks r in {4, 8, 16, 32} |
| `05_vision_and_calibration` | ViT vs. ConvNeXt + Focal Calibration | PR-AUC / ECE (3 seeds) | `TBD` | Imbalanced medical/agri benchmarks |
| `06_cuda_and_triton_kernels` | Tiled GEMM & Fused Softmax | Speedup / Bandwidth (GB/s) | `TBD` | Bound via `pybind11` |
| `07_generative_diffusion_flow` | DDPM vs. Rectified Flow Matching | FID @ NFE={10,25,50,100} | `TBD` | CIFAR-10 ODE vs. SDE comparison |
