"""First-principles AdamW & Cosine Annealing implementation."""
import math
import torch


class CustomAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {p: {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)} for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]
            st["step"] += 1
            t = st["step"]
            p.mul_(1.0 - self.lr * self.weight_decay)
            st["m"].mul_(self.beta1).add_(p.grad, alpha=1.0 - self.beta1)
            st["v"].mul_(self.beta2).addcmul_(p.grad, p.grad, value=1.0 - self.beta2)
            m_hat = st["m"] / (1.0 - self.beta1 ** t)
            v_hat = st["v"] / (1.0 - self.beta2 ** t)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
