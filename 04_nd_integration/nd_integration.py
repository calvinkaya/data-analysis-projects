"""
Monte Carlo Integration — 3D Volume, Center of Mass & Moment of Inertia.

Computes the physical properties of a sphere with an off-center cylindrical
bore using hit-or-miss Monte Carlo integration.  Fully vectorised with NumPy.

Geometry:
    - Base body: sphere of radius R = 5, centred at the origin.
    - Bore: cylinder of radius r = 2, parallel to the z-axis,
      shifted by d = 2 along the positive x-axis.
    - Density: homogeneous, rho = 1.

Usage:
    python nd_integration.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

R_SPHERE = 5.0        # Radius of the main sphere
R_CYLINDER = 2.0      # Radius of the cylindrical bore
D_CYLINDER = 2.0      # Offset of the cylinder centre along x
DENSITY = 1.0         # Homogeneous mass density
N_POINTS = 1_000_000  # Number of Monte Carlo samples
SEED = 42             # RNG seed for reproducibility


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_body_properties(
    n_points: int,
    r_sphere: float,
    r_cylinder: float,
    d_cylinder: float,
    density: float = 1.0,
    seed: int | None = 42,
) -> dict:
    """Compute volume, centre of mass, and moment of inertia via Monte Carlo.

    Generates random points in a cubic bounding box, applies vectorised
    hit-or-miss geometry filters, and evaluates volume integrals from the
    surviving sample.

    Args:
        n_points:   Number of random samples.
        r_sphere:   Radius of the base sphere.
        r_cylinder: Radius of the cylindrical bore.
        d_cylinder: x-offset of the bore axis.
        density:    Constant mass density.
        seed:       RNG seed for reproducibility.

    Returns:
        Dict with volume, mass, com, moment_of_inertia_z,
        coordinate arrays, and hit_ratio.
    """
    rng = np.random.default_rng(seed)

    # Bounding box = cube enclosing the sphere
    box_half = r_sphere
    box_volume = (2 * box_half) ** 3

    # Draw all coordinates at once: shape (3, N)
    coords = rng.uniform(-box_half, box_half, size=(3, n_points))
    x, y, z = coords

    # --- Vectorised geometry filters (no Python loops) ---
    in_sphere = (x**2 + y**2 + z**2) <= r_sphere**2
    in_cylinder = ((x - d_cylinder)**2 + y**2) <= r_cylinder**2
    valid = in_sphere & ~in_cylinder

    x_v, y_v, z_v = x[valid], y[valid], z[valid]
    n_valid = len(x_v)
    hit_ratio = n_valid / n_points

    # --- Physical properties ---
    volume = box_volume * hit_ratio
    mass = volume * density

    # Centre of mass: expectation of position over the body
    com = (np.mean(x_v), np.mean(y_v), np.mean(z_v))

    # Moment of inertia about z-axis: integral of rho*(x²+y²) dV
    #   = mass * <x² + y²>  (where <·> is the sample mean)
    iz = mass * np.mean(x_v**2 + y_v**2)

    return {
        "volume": volume,
        "mass": mass,
        "com": com,
        "moment_of_inertia_z": iz,
        "x_valid": x_v,
        "y_valid": y_v,
        "z_valid": z_v,
        "n_valid": n_valid,
        "hit_ratio": hit_ratio,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_analysis(
    result: dict,
    r_sphere: float,
    r_cylinder: float,
    d_cylinder: float,
) -> plt.Figure:
    """Create a two-panel figure: 3D surface render + 2D cross-section.

    Panel 1 uses matplotlib's plot_surface to render the sphere and
    cylinder bore as proper solid geometry with lighting.

    Panel 2 shows a thin z = 0 slice of Monte Carlo points with the
    analytical geometry boundaries overlaid — analogous to the inside /
    outside visualisation in module 01.

    Args:
        result:     Dict from compute_body_properties().
        r_sphere:   Radius of the sphere.
        r_cylinder: Radius of the cylindrical bore.
        d_cylinder: x-offset of the bore axis.

    Returns:
        The matplotlib Figure.
    """
    fig = plt.figure(figsize=(16, 7))

    # ===== Panel 1: 3D Surface Render =====
    ax1 = fig.add_subplot(121, projection="3d")

    # --- Sphere mesh with hole punched out ---
    n_u, n_v = 300, 150
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    xs = r_sphere * np.outer(np.cos(u), np.sin(v))
    ys = r_sphere * np.outer(np.sin(u), np.sin(v))
    zs = r_sphere * np.outer(np.ones(n_u), np.cos(v))

    # Punch the hole: set vertices inside the cylinder to NaN
    cyl_mask = ((xs - d_cylinder)**2 + ys**2) <= r_cylinder**2
    xs[cyl_mask] = np.nan
    ys[cyl_mask] = np.nan
    zs[cyl_mask] = np.nan

    ax1.plot_surface(
        xs, ys, zs,
        color="#3b82f6", alpha=0.35, edgecolor="none",
        rstride=2, cstride=2, shade=True,
    )

    # --- Cylinder inner wall (only the part inside the sphere) ---
    n_th, n_zh = 200, 200
    theta = np.linspace(0, 2 * np.pi, n_th)
    z_lin = np.linspace(-r_sphere, r_sphere, n_zh)
    th_grid, zc_grid = np.meshgrid(theta, z_lin)

    xc = d_cylinder + r_cylinder * np.cos(th_grid)
    yc = r_cylinder * np.sin(th_grid)

    # Clip cylinder to sphere interior
    outside_sphere = (xc**2 + yc**2 + zc_grid**2) > r_sphere**2
    xc[outside_sphere] = np.nan
    yc[outside_sphere] = np.nan
    zc_grid[outside_sphere] = np.nan

    ax1.plot_surface(
        xc, yc, zc_grid,
        color="#ef4444", alpha=0.55, edgecolor="none",
        rstride=2, cstride=2, shade=True,
    )

    # --- Centre of mass marker ---
    cx, cy, cz = result["com"]
    ax1.scatter(
        cx, cy, cz,
        color="#facc15", marker="*", s=300,
        edgecolor="black", linewidth=0.6, zorder=10,
        label=f"COM ({cx:.2f}, {cy:.2f}, {cz:.2f})",
    )

    ax1.view_init(elev=22, azim=-55)
    ax1.set_xlabel("X", fontsize=10)
    ax1.set_ylabel("Y", fontsize=10)
    ax1.set_zlabel("Z", fontsize=10)
    ax1.set_title("3D Geometry — Sphere with Cylindrical Bore", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Force equal aspect ratio
    lim = r_sphere * 1.1
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_zlim(-lim, lim)
    ax1.set_box_aspect((1, 1, 1))

    # ===== Panel 2: 2D Cross-Section (z ≈ 0) =====
    ax2 = fig.add_subplot(122)

    x_all, y_all, z_all = (
        result["x_valid"], result["y_valid"], result["z_valid"],
    )

    # Take a thin slice around z = 0
    slice_thickness = 0.15
    slice_mask = np.abs(z_all) < slice_thickness
    x_sl = x_all[slice_mask]
    y_sl = y_all[slice_mask]

    # Downsample for plotting
    n_show = min(8000, len(x_sl))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_sl), size=n_show, replace=False)

    # Colour by distance to z-axis (inertia contribution)
    dist = np.sqrt(x_sl[idx]**2 + y_sl[idx]**2)
    sc = ax2.scatter(
        x_sl[idx], y_sl[idx],
        c=dist, cmap="viridis", s=3, alpha=0.7, edgecolors="none",
    )

    # Analytical geometry outlines
    circle_sphere = plt.Circle(
        (0, 0), r_sphere,
        fill=False, linewidth=2, edgecolor="#3b82f6", linestyle="-",
        label="Sphere boundary",
    )
    circle_cyl = plt.Circle(
        (d_cylinder, 0), r_cylinder,
        fill=False, linewidth=2, edgecolor="#ef4444", linestyle="--",
        label="Cylinder bore",
    )
    ax2.add_patch(circle_sphere)
    ax2.add_patch(circle_cyl)

    # COM projection
    ax2.plot(
        cx, cy, "*",
        color="#facc15", markersize=18,
        markeredgecolor="black", markeredgewidth=0.8,
        label="COM projection",
    )

    cbar = fig.colorbar(sc, ax=ax2, shrink=0.85)
    cbar.set_label("Distance to Z-Axis", fontsize=10)

    ax2.set_xlim(-r_sphere * 1.15, r_sphere * 1.15)
    ax2.set_ylim(-r_sphere * 1.15, r_sphere * 1.15)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_title("Cross-Section at $z = 0$", fontsize=13)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    # Summary annotation
    textstr = (
        f"$V \\approx {result['volume']:.1f}$\n"
        f"$I_z \\approx {result['moment_of_inertia_z']:.1f}$\n"
        f"$N = {result['n_valid']:,}$ hits"
    )
    ax2.text(
        0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.9, edgecolor="#ccc"),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run simulation, print results, save figure."""
    print(f"Running 3D Monte Carlo integration with {N_POINTS:,} points ...")

    result = compute_body_properties(
        n_points=N_POINTS,
        r_sphere=R_SPHERE,
        r_cylinder=R_CYLINDER,
        d_cylinder=D_CYLINDER,
        density=DENSITY,
        seed=SEED,
    )

    print(
        f"\n{'=' * 55}\n"
        f"  Physical Properties (Monte Carlo Estimates)\n"
        f"{'=' * 55}\n"
        f"  Volume           : {result['volume']:.4f}\n"
        f"  Mass  (rho=1)    : {result['mass']:.4f}\n"
        f"  Centre of Mass   : ({result['com'][0]:+.4f}, "
        f"{result['com'][1]:+.4f}, {result['com'][2]:+.4f})\n"
        f"  I_z              : {result['moment_of_inertia_z']:.4f}\n"
        f"  Acceptance ratio : {result['hit_ratio'] * 100:.2f} %\n"
        f"{'=' * 55}"
    )

    # --- Save figure ---
    fig = plot_analysis(result, R_SPHERE, R_CYLINDER, D_CYLINDER)

    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / "3d_integration_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()
