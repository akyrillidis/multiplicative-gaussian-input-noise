"""
Smoothed ReLU under multiplicative Gaussian input noise.

Validates the closed-form expectation
    E_c[ ReLU(w^T (x ⊙ c)) ] = z Φ(z/σ) + σ φ(z/σ),
where z = w^T x, σ = κ ||w⊙x||_2, and c ~ N(1, κ² I), against a Monte Carlo
estimate. Also plots the proxy form σ̂(w,x) = z Φ(z/σ) used in Theorem 4.1
of the paper.

Run:
    python -m src.smoothed_relu
or
    python src/smoothed_relu.py

Outputs into ./figures/:
    fig01a_smoothed_relu_proxy.{png,pdf}    -- proxy σ̂
    fig01b_smoothed_relu_exact.{png,pdf}    -- exact σ̃
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm


def relu(z):
    return np.maximum(0.0, z)


def proxy_smoothed_relu(z, sigma):
    """σ̂(w,x) = z Φ(z/σ)  — proxy used in Theorem 4.1 (no φ term)."""
    return z * scipy_norm.cdf(z / sigma)


def exact_smoothed_relu(z, sigma):
    """σ̃(w,x) = z Φ(z/σ) + σ φ(z/σ)  — exact closed form."""
    return z * scipy_norm.cdf(z / sigma) + sigma * scipy_norm.pdf(z / sigma)


def monte_carlo_smoothed_relu(z_grid, sigma, n_samples=200_000, rng=None):
    rng = rng if rng is not None else np.random.default_rng(42)
    eps = sigma * rng.standard_normal(n_samples)
    Z = eps[:, None] + z_grid[None, :]
    return relu(Z).mean(axis=0)


def plot_panel(z_grid, theory, empirical, relu_vals, theory_label, out_stem,
               figsize=(16, 8), fontsize=30):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
    })
    fig = plt.figure(figsize=figsize)
    plt.plot(z_grid, theory,    "o-",  markevery=25, markersize=10, linewidth=4, label=theory_label)
    plt.plot(z_grid, empirical, "s--", markevery=25, markersize=10, linewidth=4,
             label=r"Empirical $\mathbb{E}_{\mathbf{c}}[\sigma(\mathbf{w}^\top(\mathbf{x}\odot\mathbf{c}))]$")
    plt.plot(z_grid, relu_vals, "k:",  linewidth=4,
             label=r"Standard ReLU $\sigma(\mathbf{w}^\top\mathbf{x})$")
    plt.xlabel(r"Input $\mathbf{w}^\top\mathbf{x}$", fontsize=fontsize)
    plt.ylabel("Activation Value",                   fontsize=fontsize)
    plt.xticks(fontsize=fontsize); plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    out_dir = Path(out_stem).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def main(kappa=0.2, fixed_norm_u=1.0, n_samples=200_000, out_dir="figures", seed=42):
    sigma = kappa * fixed_norm_u
    z_grid = np.linspace(-2.1, 2.1, 400)
    rng = np.random.default_rng(seed)

    empirical = monte_carlo_smoothed_relu(z_grid, sigma, n_samples=n_samples, rng=rng)
    relu_vals = relu(z_grid)

    out = Path(out_dir)
    plot_panel(
        z_grid,
        proxy_smoothed_relu(z_grid, sigma),
        empirical, relu_vals,
        rf"Theoretical proxy $\hat{{\sigma}}(\mathbf{{w}},\mathbf{{x}})$ ($\kappa={kappa}$)",
        str(out / "fig01a_smoothed_relu_proxy"),
    )
    plot_panel(
        z_grid,
        exact_smoothed_relu(z_grid, sigma),
        empirical, relu_vals,
        rf"Theoretical exact $\tilde{{\sigma}}(\mathbf{{w}},\mathbf{{x}})$ ($\kappa={kappa}$)",
        str(out / "fig01b_smoothed_relu_exact"),
    )
    print(f"Saved {out}/fig01a_smoothed_relu_proxy.{{png,pdf}}")
    print(f"Saved {out}/fig01b_smoothed_relu_exact.{{png,pdf}}")


if __name__ == "__main__":
    main()
