"""
Monte Carlo approximation of pi.

Estimates pi by sampling random points in the unit square [0, 1) x [0, 1)
and counting how many fall inside the quarter unit circle (x^2 + y^2 <= 1).
The ratio of inside points to total points approximates pi/4.

Usage:
    python pi_monte_carlo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def calculate_pi(num_points: int, seed: int | None = 42) -> dict:
    """Estimate pi via Monte Carlo sampling in the unit square.

    Args:
        num_points: Number of random (x, y) samples to draw.
        seed: RNG seed for reproducibility (None for non-deterministic).

    Returns:
        Dictionary with keys:
            - x, y           : coordinate arrays (num_points,)
            - inside_mask     : boolean mask – True when x²+y² ≤ 1
            - pi_estimate     : the Monte Carlo estimate of pi
    """
    rng = np.random.default_rng(seed)
    x = rng.random(num_points)
    y = rng.random(num_points)

    # Vectorised distance check – no Python loop required
    inside_mask = x**2 + y**2 <= 1.0

    pi_estimate = 4.0 * np.sum(inside_mask) / num_points

    return {
        "x": x,
        "y": y,
        "inside_mask": inside_mask,
        "pi_estimate": pi_estimate,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_pi_approximation(
    x: np.ndarray,
    y: np.ndarray,
    inside_mask: np.ndarray,
    pi_estimate: float,
    num_points: int,
) -> plt.Figure:
    """Create a publication-quality scatter plot of the Monte Carlo result.

    Points inside the quarter circle are coloured differently from those
    outside.  The exact quarter-circle arc is overlaid for reference.

    Args:
        x, y          : coordinate arrays of sampled points.
        inside_mask   : boolean array indicating inside / outside.
        pi_estimate   : the calculated approximation of pi.
        num_points    : total number of sampled points (used in title).

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter – inside vs. outside
    point_size = max(0.2, 8.0 - np.log10(num_points))  # auto-scale marker
    ax.scatter(
        x[inside_mask], y[inside_mask],
        s=point_size, color="#3b82f6", alpha=0.6,
        label="Inside circle", edgecolors="none",
    )
    ax.scatter(
        x[~inside_mask], y[~inside_mask],
        s=point_size, color="#ef4444", alpha=0.6,
        label="Outside circle", edgecolors="none",
    )

    # Quarter-circle arc
    arc = patches.Arc(
        (0, 0), 2, 2, angle=0, theta1=0, theta2=90,
        linewidth=2, color="#facc15", linestyle="-",
        label="Unit circle",
    )
    ax.add_patch(arc)

    # Layout & labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        rf"Monte Carlo estimation of $\pi$ — "
        rf"$\pi \approx {pi_estimate:.6f}$ "
        rf"($n = {num_points:,}$)",
        fontsize=13,
        pad=12,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the simulation and save / display the plot."""
    num_points = 50000

    result = calculate_pi(num_points)

    fig = plot_pi_approximation(
        x=result["x"],
        y=result["y"],
        inside_mask=result["inside_mask"],
        pi_estimate=result["pi_estimate"],
        num_points=num_points,
    )

    # Save figure for GitHub README
    from pathlib import Path

    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "pi_approximation.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to {fig_path}")

    plt.show()

    print(
        f"Monte Carlo estimate of pi with {num_points:,} points: "
        f"{result['pi_estimate']:.6f}"
    )


if __name__ == "__main__":
    main()
