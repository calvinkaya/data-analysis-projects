# 2D Random Walk — Brownian Motion Simulation

This module simulates the diffusion of 10,000 independent particles performing a 2D Gaussian random walk and validates the fundamental diffusion law MSD(t) = 4Dt via linear regression.

## 🧠 How it Works
Each particle starts at the origin and takes a series of independent steps drawn from a 2D Gaussian distribution (mean 0, standard deviation σ).

After simulating all trajectories, the Mean Squared Displacement (MSD) is computed by averaging the squared distance from the origin across all particles at each time step. A linear fit to MSD(t) extracts the approximate diffusion coefficient D, which is then compared against the theoretical prediction D = σ² / 2.

## 🚀 Technical Highlights
* **Full Vectorization:** All 10,000 × 1,000 steps are drawn in a single `NumPy` call and accumulated via `np.cumsum` — no Python loop over time steps, making the simulation faster than a naive implementation.
* **Modular Architecture:** Strict separation between simulation (`simulate_random_walks`), analysis (`compute_msd`, `fit_diffusion`), and visualization (`plot_analysis`, `create_animation`).
* **Diffusion Validation:** Automatic linear regression on MSD(t) with R² reporting to quantitatively confirm agreement with the Einstein diffusion relation.
* **Type Hinting:** Fully typed arguments and return values (e.g., `np.ndarray`, `plt.Figure`) for better code quality.

## 📊 Visualization
The script generates a two-panel analysis figure (sample trajectories + MSD validation) and an animated GIF showing the particle cloud expanding over time. All outputs are automatically saved to the `figures/` directory.

![Random Walk Analysis](./figures/random_walk_analysis.png)

![Diffusion Animation](./figures/diffusion_animation.gif)

## 💻 How to Run
Ensure you have the required libraries installed:
`pip install numpy matplotlib pillow`

Run the script from your terminal:
`random_walk.py`
