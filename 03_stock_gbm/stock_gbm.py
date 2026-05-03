"""
Monte Carlo Stock Price Simulation — Geometric Brownian Motion (GBM).

Simulates M stock price paths over T trading days using the discretised GBM
model, then computes the 95 % Value at Risk (VaR) from the distribution of
terminal prices.  Fully vectorised with NumPy — no Python loop over time steps.

Usage:
    python stock_gbm.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

S0 = 100            # Initial stock price (€)
MU = 0.08           # Expected annual return (drift)
SIGMA = 0.20        # Annual volatility
T = 1.0             # Time horizon in years
N_DAYS = 252        # Number of trading days in one year
M = 10_000          # Number of Monte Carlo paths
CONFIDENCE = 0.95   # VaR confidence level
SEED = 42           # RNG seed for reproducibility


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    n_days: int,
    n_paths: int,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate stock price paths via discretised Geometric Brownian Motion.

    Uses the exact-solution discretisation:
        S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0, 1).  All paths are generated simultaneously — no loop
    over time steps.

    Args:
        s0:      Initial stock price.
        mu:      Annualised drift (expected return).
        sigma:   Annualised volatility.
        n_days:  Number of discrete time steps (trading days).
        n_paths: Number of independent simulated paths.
        seed:    RNG seed for reproducibility.

    Returns:
        Price matrix of shape (n_paths, n_days + 1).
        Column 0 is the initial price S0.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_days  # each step = 1 trading day

    # Drift and diffusion per time step
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Draw ALL random shocks at once: (M, N) matrix
    z = rng.standard_normal(size=(n_paths, n_days))

    # Daily log-returns (vectorised)
    log_returns = drift + diffusion * z  # (M, N)

    # Cumulative sum of log-returns → cumulative product of price ratios
    log_paths = np.cumsum(log_returns, axis=1)

    # Prepend zero (log(S0/S0) = 0) and exponentiate
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), log_paths], axis=1,
    )
    prices = s0 * np.exp(log_paths)  # (M, N+1)

    return prices


# ---------------------------------------------------------------------------
# Risk Analysis
# ---------------------------------------------------------------------------

def compute_var(
    prices: np.ndarray,
    s0: float,
    confidence: float = 0.95,
) -> dict:
    """Compute Value at Risk from simulated terminal prices.

    Args:
        prices:     Price matrix of shape (n_paths, n_days + 1).
        s0:         Initial stock price.
        confidence: Confidence level (e.g. 0.95 for 95 % VaR).

    Returns:
        Dict with terminal_prices, var_absolute, var_percent,
        var_price_level, and percentile.
    """
    terminal_prices = prices[:, -1]
    percentile = (1.0 - confidence) * 100  # 5th percentile for 95% VaR

    var_price_level = np.percentile(terminal_prices, percentile)
    var_absolute = s0 - var_price_level          # loss in €
    var_percent = (var_absolute / s0) * 100.0    # loss in %

    return {
        "terminal_prices": terminal_prices,
        "var_absolute": var_absolute,
        "var_percent": var_percent,
        "var_price_level": var_price_level,
        "percentile": percentile,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_gbm_analysis(
    prices: np.ndarray,
    var_result: dict,
    s0: float,
    n_show: int = 100,
    seed: int | None = 0,
) -> plt.Figure:
    """Create a two-panel figure: sample paths + terminal price distribution.

    Args:
        prices:     Full price matrix (n_paths, n_days + 1).
        var_result: Dict from compute_var().
        s0:         Initial stock price.
        n_show:     Number of paths to display in the line plot.
        seed:       Seed for selecting which paths to show.

    Returns:
        The matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(prices.shape[0], size=min(n_show, prices.shape[0]),
                     replace=False)

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("ggplot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel 1: Sample Price Paths ---
    days = np.arange(prices.shape[1])
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(idx)))

    for i, color in zip(idx, colors):
        ax1.plot(days, prices[i], linewidth=0.5, alpha=0.6, color=color)

    ax1.axhline(y=s0, color="#facc15", linewidth=1.5, linestyle="--",
                alpha=0.8, label=f"$S_0 = {s0}$")
    ax1.set_xlabel("Trading Days", fontsize=12)
    ax1.set_ylabel("Price (€)", fontsize=12)
    ax1.set_title(
        f"GBM Simulated Price Paths (showing {len(idx)} of {prices.shape[0]:,})",
        fontsize=13,
    )
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Terminal Price Distribution + VaR ---
    terminal = var_result["terminal_prices"]
    var_level = var_result["var_price_level"]

    ax2.hist(
        terminal, bins=80, color="#3b82f6", alpha=0.7,
        edgecolor="white", linewidth=0.3, density=True,
    )
    ax2.axvline(
        x=var_level, color="#ef4444", linewidth=2.5, linestyle="--",
        label=(
            f"95 % VaR: €{var_result['var_absolute']:.2f} loss "
            f"({var_result['var_percent']:.1f} %)"
        ),
    )
    ax2.axvline(
        x=s0, color="#facc15", linewidth=1.5, linestyle="--",
        alpha=0.8, label=f"$S_0 = {s0}$",
    )

    # Annotation box with statistics
    textstr = (
        f"$\\mu = {terminal.mean():.2f}$\n"
        f"$\\sigma = {terminal.std():.2f}$\n"
        f"$\\mathrm{{VaR}}_{{95\\%}} = €{var_result['var_absolute']:.2f}$"
    )
    ax2.text(
        0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.9, edgecolor="#ccc"),
    )

    ax2.set_xlabel("Terminal Price (€)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(
        f"Distribution of Terminal Prices ($T$ = {prices.shape[1] - 1} days)",
        fontsize=13,
    )
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run GBM simulation, compute VaR, save plot, and print results."""
    print(f"Simulating {M:,} GBM paths × {N_DAYS} days ...")
    prices = simulate_gbm(S0, MU, SIGMA, N_DAYS, M, seed=SEED)
    print(f"Price matrix: {prices.shape}  ({prices.nbytes / 1e6:.1f} MB)")

    # --- Risk analysis ---
    var_result = compute_var(prices, S0, confidence=CONFIDENCE)
    print(
        f"\n{'=' * 50}\n"
        f"  95 % Value at Risk\n"
        f"{'=' * 50}\n"
        f"  Absolute loss : €{var_result['var_absolute']:.2f}\n"
        f"  Percentage    : {var_result['var_percent']:.2f} %\n"
        f"  Price level   : €{var_result['var_price_level']:.2f}\n"
        f"{'=' * 50}"
    )

    # --- Save static figure ---
    fig = plot_gbm_analysis(prices, var_result, S0, n_show=100)
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "gbm_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()
