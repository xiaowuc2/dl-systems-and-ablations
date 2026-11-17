"""
Truncated Singular Value Decomposition (SVD), Eckart-Young-Mirsky Theorem,
and Low-Rank Weight Update Spectrum Analysis for LoRA.

Mathematical Foundation:
    Given a weight update matrix $\Delta W \in \mathbb{R}^{m \times n}$ with rank $R = \min(m, n)$,
    its full SVD is:
        $$\Delta W = U \Sigma V^\top$$
    where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal,
    and $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_R)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_R \ge 0$.

    By the Eckart-Young-Mirsky Theorem, the optimal rank-$r$ approximation ($r < R$)
    under both the Frobenius norm $\|\cdot\|_F$ and Spectral norm $\|\cdot\|_2$ is:
        $$\Delta W_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top = U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^\top$$

    Exact Approximation Errors:
        $$\|\Delta W - \Delta W_r\|_F = \sqrt{\sum_{i=r+1}^{R} \sigma_i^2}, \qquad \|\Delta W - \Delta W_r\|_2 = \sigma_{r+1}$$

    Parameter Compression Ratio (LoRA $B A$ with $B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}$):
        $$\text{Compression} = \frac{r(m + n)}{m \cdot n}$$

Complexity:
    - Exact SVD via divide-and-conquer: $\mathcal{O}(\min(mn^2, m^2n))$ FLOPs, $\mathcal{O}(mn)$ memory.
    - Rank-$r$ forward pass ($x A^\top B^\top$): $\mathcal{O}(B \cdot T \cdot r(m + n))$ FLOPs vs. $\mathcal{O}(B \cdot T \cdot mn)$ dense.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Enforce deterministic execution across CPU and CUDA."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construct_synthetic_delta_w(
    in_features: int = 768,
    out_features: int = 768,
    intrinsic_rank: int = 16,
    noise_std: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Construct a realistic fine-tuning weight update $\Delta W \in \mathbb{R}^{m \times n}$
    exhibiting rapid spectral decay (Aghajanyan et al., 2020; Hu et al., 2021).

    Shape:
        Returns $\Delta W$ of shape `(out_features, in_features)`.
    """
    # Low-rank signal subspace with power-law singular value decay
    u_true, _ = torch.linalg.qr(torch.randn(out_features, intrinsic_rank, device=device, dtype=torch.float64))
    v_true, _ = torch.linalg.qr(torch.randn(in_features, intrinsic_rank, device=device, dtype=torch.float64))
    decay = torch.linspace(1.0, 0.05, intrinsic_rank, device=device, dtype=torch.float64) ** 2
    signal = (u_true * decay.unsqueeze(0)) @ v_true.T

    # Full-rank isotropic optimization noise
    noise = noise_std * torch.randn(out_features, in_features, device=device, dtype=torch.float64) / (in_features ** 0.5)
    return signal + noise


def truncated_svd_reconstruct(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact rank-$r$ truncated SVD reconstruction $U_{:,:r} \Sigma_{:r} V_{:,:r}^\top$.

    Args:
        matrix: Tensor of shape `(M, N)` in float64.
        rank: Target truncation rank $r \le \min(M, N)$.

    Returns:
        approx: Reconstructed rank-$r$ tensor of shape `(M, N)`.
        singular_values: Full vector of singular values of shape `(min(M, N),)`.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    approx = (U_r * S_r.unsqueeze(0)) @ Vh_r
    return approx, S


def verify_eckart_young_theorem(
    matrix: torch.Tensor,
    ranks: List[int],
) -> List[Dict[str, float]]:
    """
    Verify that truncated SVD matches theoretical Frobenius & Spectral norm residuals
    and strictly outperforms random rank-$r$ projections (Eckart-Young optimality).
    """
    m, n = matrix.shape
    _, S = truncated_svd_reconstruct(matrix, rank=1)
    total_energy = torch.sum(S ** 2).item()
    dense_norm_f = torch.linalg.norm(matrix, ord="fro").item()

    results: List[Dict[str, float]] = []

    for r in ranks:
        approx, _ = truncated_svd_reconstruct(matrix, rank=r)
        residual = matrix - approx

        # Empirical norms
        emp_fro = torch.linalg.norm(residual, ord="fro").item()
        emp_spec = torch.linalg.norm(residual, ord=2).item()

        # Theoretical Eckart-Young-Mirsky norms
        theo_fro = torch.sqrt(torch.sum(S[r:] ** 2)).item()
        theo_spec = S[r].item() if r < len(S) else 0.0

        # Numerical verification assertions
        assert torch.allclose(
            torch.tensor(emp_fro, dtype=torch.float64),
            torch.tensor(theo_fro, dtype=torch.float64),
            atol=1e-10,
        ), f"Frobenius mismatch at r={r}: {emp_fro} vs {theo_fro}"

        assert torch.allclose(
            torch.tensor(emp_spec, dtype=torch.float64),
            torch.tensor(theo_spec, dtype=torch.float64),
            atol=1e-10,
        ), f"Spectral mismatch at r={r}: {emp_spec} vs {theo_spec}"

        # Compare against a random rank-r subspace projection to verify optimality
        q_rand, _ = torch.linalg.qr(torch.randn(m, r, device=matrix.device, dtype=torch.float64))
        rand_proj = q_rand @ (q_rand.T @ matrix)
        rand_fro = torch.linalg.norm(matrix - rand_proj, ord="fro").item()
        assert emp_fro <= rand_fro + 1e-12, f"Eckart-Young optimality violated at r={r}!"

        captured_energy = (torch.sum(S[:r] ** 2).item() / total_energy) * 100.0
        rel_fro_error = (emp_fro / dense_norm_f) * 100.0
        param_ratio = (r * (m + n)) / (m * n) * 100.0

        results.append(
            {
                "rank": r,
                "fro_error": emp_fro,
                "rel_fro_error_pct": rel_fro_error,
                "spectral_error": emp_spec,
                "energy_captured_pct": captured_energy,
                "lora_param_pct": param_ratio,
                "random_proj_fro_error": rand_fro,
            }
        )

    return results


def save_artifacts(
    results: List[Dict[str, float]],
    singular_values: torch.Tensor,
    out_dir: Path,
) -> None:
    """Save quantitative CSV metrics and publication-ready SVD spectrum plot."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write CSV table
    csv_path = out_dir / "eckart_young_lora_metrics.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # 2. Generate 2-panel diagnostic figure
    sv_np = singular_values.detach().cpu().numpy()
    cum_energy = (torch.cumsum(singular_values ** 2, dim=0) / torch.sum(singular_values ** 2)).detach().cpu().numpy() * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Panel A: Singular Value Spectrum (Log Scale)
    axes[0].plot(range(1, len(sv_np) + 1), sv_np, color="#1f77b4", linewidth=2.0)
    axes[0].axvline(16, color="#d62728", linestyle="--", linewidth=1.3, label="Intrinsic Rank (r=16)")
    axes[0].set_yscale("log")
    axes[0].set_title("Singular Value Spectrum of $\Delta W$ (768x768)")
    axes[0].set_xlabel("Singular Value Index $i$")
    axes[0].set_ylabel("$\sigma_i$ (log scale)")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend()

    # Panel B: Cumulative Spectral Energy vs. Parameter Footprint
    ranks = [row["rank"] for row in results]
    rel_err = [row["rel_fro_error_pct"] for row in results]
    param_pct = [row["lora_param_pct"] for row in results]

    ax2 = axes[1]
    line1 = ax2.plot(ranks, rel_err, marker="o", color="#d62728", linewidth=2.0, label="Relative Frobenius Error (%)")
    ax2.set_xlabel("Truncation Rank $r$")
    ax2.set_ylabel("Relative Frobenius Error (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(ranks, param_pct, marker="s", color="#2ca02c", linewidth=2.0, linestyle="--", label="LoRA Params vs Dense (%)")
    ax2_twin.set_ylabel("LoRA Parameter Ratio (%)", color="#2ca02c")
    ax2_twin.tick_params(axis="y", labelcolor="#2ca02c")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center right")
    ax2.set_title("Rank-$r$ Approximation Error vs. Parameter Cost")

    plt.tight_layout()
    plot_path = out_dir / "svd_spectrum_and_lora_tradeoff.png"
    plt.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running SVD & Eckart-Young Verification on device: {device}")

    delta_w = construct_synthetic_delta_w(
        in_features=768,
        out_features=768,
        intrinsic_rank=16,
        noise_std=1e-3,
        device=device,
    )

    ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    metrics = verify_eckart_young_theorem(delta_w, ranks=ranks)
    _, singular_values = truncated_svd_reconstruct(delta_w, rank=1)

    print("\n" + "=" * 86)
    print(f"{'Rank r':>6} | {'Frob Err':>10} | {'Rel Err %':>10} | {'Spec Err':>10} | {'Energy %':>10} | {'LoRA Params %':>13}")
    print("-" * 86)
    for row in metrics:
        print(
            f"{int(row['rank']):>6} | "
            f"{row['fro_error']:>10.6f} | "
            f"{row['rel_fro_error_pct']:>9.2f}% | "
            f"{row['spectral_error']:>10.6f} | "
            f"{row['energy_captured_pct']:>9.2f}% | "
            f"{row['lora_param_pct']:>12.2f}%"
        )
    print("=" * 86)
    print("[PASS] Eckart-Young-Mirsky Frobenius & Spectral assertions passed (atol=1e-10).")

    results_dir = Path(__file__).resolve().parent / "results"
    save_artifacts(metrics, singular_values, results_dir)
    print(f"[SAVED] CSV metrics & spectrum plot written to: {results_dir}")


if __name__ == "__main__":
    main()
