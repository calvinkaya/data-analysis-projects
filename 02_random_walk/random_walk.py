"""
2D Random Walk — Brownian Motion Simulation.

Simulates N particles performing a 2D Gaussian random walk and validates
the diffusion law MSD(t) = 4Dt via linear regression.  Fully vectorised
with NumPy — no Python loop over time steps.

Usage:
    python random_walk.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_random_walks(
    n_particles: int = 10_000,
    n_steps: int = 1_000,
    sigma: float = 1.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate 2D Gaussian random walks, fully vectorised.

    All steps are drawn at once and accumulated via np.cumsum —
    no Python loop over time steps.

    Args:
        n_particles: Number of independent walkers.
        n_steps:     Number of discrete time steps.
        sigma:       Standard deviation of each Gaussian step.
        seed:        RNG seed for reproducibility.

    Returns:
        Trajectory array of shape (n_particles, n_steps + 1, 2).
        Index 0 along axis 1 is the origin (0, 0).
    """
    rng = np.random.default_rng(seed)

    # Draw ALL steps at once: (N, M, 2)
    steps = rng.normal(loc=0.0, scale=sigma, size=(n_particles, n_steps, 2))

    # Prepend the origin and accumulate
    origin = np.zeros((n_particles, 1, 2))
    trajectories = np.concatenate([origin, steps], axis=1)
    np.cumsum(trajectories, axis=1, out=trajectories)

    return trajectories


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_msd(trajectories: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Displacement averaged over all particles.

    Args:
        trajectories: Array of shape (n_particles, n_steps + 1, 2).

    Returns:
        MSD array of shape (n_steps + 1,).
    """
    squared_distances = np.sum(trajectories ** 2, axis=2)  # (N, M+1)
    return np.mean(squared_distances, axis=0)               # (M+1,)


def fit_diffusion(msd: np.ndarray, sigma: float) -> dict:
    """Fit a line to MSD(t) and extract the diffusion coefficient.

    In 2D Brownian motion: MSD(t) = 4D·t, so D_approx = slope / 4.
    Theoretical: D_theo = sigma² / 2.

    Args:
        msd:   MSD array of shape (n_steps + 1,).
        sigma: Step standard deviation used in simulation.

    Returns:
        Dict with slope, intercept, r_squared, d_approx, d_theo.
    """
    t = np.arange(len(msd))
    slope, intercept = np.polyfit(t, msd, deg=1)

    # R² calculation
    msd_predicted = slope * t + intercept
    ss_res = np.sum((msd - msd_predicted) ** 2)
    ss_tot = np.sum((msd - np.mean(msd)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    d_approx = slope / 4.0
    d_theo = sigma ** 2 / 2.0

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "d_approx": d_approx,
        "d_theo": d_theo,
    }


# ---------------------------------------------------------------------------
# Visualisation — Static Plots
# ---------------------------------------------------------------------------

def plot_analysis(
    trajectories: np.ndarray,
    msd: np.ndarray,
    fit: dict,
    n_highlight: int = 10,
    seed: int | None = 0,
) -> plt.Figure:
    """Create a two-panel figure: sample trajectories + MSD validation.

    Args:
        trajectories: Full trajectory array (n_particles, n_steps+1, 2).
        msd:          MSD array (n_steps+1,).
        fit:          Dict from fit_diffusion().
        n_highlight:  Number of trajectories to plot.
        seed:         Seed for selecting which trajectories to show.

    Returns:
        The matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(trajectories.shape[0], size=n_highlight, replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Trajectories ---
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_highlight))

    for i, color in zip(idx, colors):
        ax1.plot(
            trajectories[i, :, 0],
            trajectories[i, :, 1],
            linewidth=0.5, alpha=0.8, color=color,
        )

    ax1.plot(0, 0, "ko", markersize=8, zorder=5, label="Origin")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Sample Trajectories (n = {n_highlight})", fontsize=13)
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # --- Panel 2: MSD vs. Time ---
    t = np.arange(len(msd))

    ax2.plot(t, msd, ".", color="#3b82f6", markersize=1, alpha=0.4,
             label="MSD data")
    ax2.plot(
        t,
        fit["slope"] * t + fit["intercept"],
        "-", color="#ef4444", linewidth=2,
        label=f"Fit: MSD = {fit['slope']:.2f}·t + {fit['intercept']:.2f}",
    )

    textstr = (
        f"$R^2 = {fit['r_squared']:.6f}$\n"
        f"$D_{{approx}} = {fit['d_approx']:.4f}$\n"
        f"$D_{{theo}}\\;= {fit['d_theo']:.4f}$"
    )
    ax2.text(
        0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.9, edgecolor="#ccc"),
    )

    ax2.set_xlabel("Time step $t$")
    ax2.set_ylabel("MSD$(t)$")
    ax2.set_title("Mean Squared Displacement — Diffusion Validation",
                  fontsize=13)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualisation — Animation
# ---------------------------------------------------------------------------

def create_animation(
    trajectories: np.ndarray,
    n_animate: int = 500,
    interval_ms: int = 50,
    seed: int | None = 1,
) -> tuple[plt.Figure, FuncAnimation]:
    """Animate diffusing particles as an expanding cloud.

    Args:
        trajectories: Full trajectory array (n_particles, n_steps+1, 2).
        n_animate:    Number of particles to show (subset for performance).
        interval_ms:  Delay between frames in milliseconds.
        seed:         Seed for selecting the animated subset.

    Returns:
        Tuple of (Figure, FuncAnimation).
    """
    rng = np.random.default_rng(seed)
    n_show = min(n_animate, trajectories.shape[0])
    idx = rng.choice(trajectories.shape[0], size=n_show, replace=False)
    subset = trajectories[idx]  # (n_animate, M+1, 2)

    n_steps = subset.shape[1] - 1
    limit = 3.0 * np.std(subset[:, -1, :])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Brownian Diffusion", fontsize=14)
    ax.grid(True, alpha=0.15)

    scatter, = ax.plot([], [], ".", color="black", markersize=1, alpha=0.5)
    time_label = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Show every k-th frame to keep animation smooth
    frame_skip = max(1, n_steps // 150)
    frames = list(range(0, n_steps + 1, frame_skip))

    def init():
        scatter.set_data([], [])
        time_label.set_text("")
        return scatter, time_label

    def update(frame):
        scatter.set_data(subset[:, frame, 0], subset[:, frame, 1])
        time_label.set_text(f"t = {frame}")
        return scatter, time_label

    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        blit=True, interval=interval_ms, repeat=True,
    )
    return fig, anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run simulation, analyse, save plots, and show animation."""
    N = 10_000
    M = 1_000
    SIGMA = 1.0

    print(f"Simulating {N:,} particles x {M:,} steps ...")
    trajectories = simulate_random_walks(N, M, SIGMA)
    print(f"Trajectory array: {trajectories.shape}  "
          f"({trajectories.nbytes / 1e6:.1f} MB)")

    # --- Analysis ---
    msd = compute_msd(trajectories)
    fit = fit_diffusion(msd, SIGMA)
    print(
        f"D_approx = {fit['d_approx']:.4f}  "
        f"(D_theo = {fit['d_theo']:.4f},  R^2 = {fit['r_squared']:.6f})"
    )

    # --- Save static figure ---
    fig = plot_analysis(trajectories, msd, fit)
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "random_walk_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Static figure saved to {fig_path}")

    # --- Save animation as GIF ---
    anim_fig, anim = create_animation(trajectories, n_animate=500)
    gif_path = figures_dir / "diffusion_animation.gif"
    try:
        anim.save(str(gif_path), writer="pillow", fps=25)
        print(f"Animation saved to {gif_path}")
    except Exception as e:
        print(f"Could not save GIF ({e}) — showing interactively instead.")

    plt.show()


if __name__ == "__main__":
    main()
