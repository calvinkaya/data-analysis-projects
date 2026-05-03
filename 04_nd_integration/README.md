# Monte Carlo Integration — 3D Volume, Center of Mass & Inertia

This module computes the physical properties of a complex, asymmetric 3D body using Monte Carlo hit-or-miss integration: a sphere with an off-center cylindrical bore drilled through it.

## 🧠 How it Works
The geometry consists of a sphere ($R = 5$) centred at the origin with a cylindrical hole ($r = 2$) drilled parallel to the z-axis, offset by $d = 2$ along x.

The algorithm generates 1,000,000 random points in a bounding box and applies vectorised boolean filters to determine which points lie inside the body (inside sphere AND outside cylinder). From the surviving sample, three integrals are evaluated:

- **Volume**: $V = V_{\text{box}} \cdot \frac{N_{\text{hits}}}{N_{\text{total}}}$
- **Centre of Mass**: $\vec{r}_{\text{COM}} = \langle \vec{r} \rangle$ over all valid points
- **Moment of Inertia**: $I_z = M \cdot \langle x^2 + y^2 \rangle$

## 🚀 Technical Highlights
* **Vectorization:** All 1,000,000 geometry checks run as pure NumPy boolean mask operations — zero Python loops. The entire computation finishes in under a second.
* **Modular Architecture:** Strict separation between the Monte Carlo computation (`compute_body_properties`) and visualization (`plot_analysis`).
* **Surface Rendering:** Uses Matplotlib's `plot_surface` to render the sphere and cylinder bore as proper 3D solid geometry with lighting and transparency — not just a point cloud.
* **2D Cross-Section:** A thin $z = 0$ slice shows Monte Carlo sample points coloured by their distance to the z-axis (inertia contribution), with analytical geometry outlines overlaid.
* **Type Hinting:** Fully typed arguments and return values for better code quality.

## 📊 Visualization
The script generates a two-panel figure: a 3D surface render of the geometry and a 2D cross-section with analytical boundaries. Output is saved to `figures/`.

![3D Integration Analysis](./figures/3d_integration_analysis.png)

## 💻 How to Run
Ensure you have the required libraries installed:
`pip install numpy matplotlib`

Run the script from your terminal:
`nd_integration.py`
