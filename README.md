# Monte Carlo Simulations 🎲

A portfolio of **Monte Carlo simulations** written in Python, showcasing numerical methods, stochastic modelling, and scientific visualisation.  
Built with a focus on **vectorised NumPy computation**, **clean modular code**, and **publication-quality plots**.

---

## Modules

| # | Module | Topic | Status |
|---|--------|-------|--------|
| 1 | [`01_pi_approximation`](01_pi_approximation/) | Estimating π via random sampling | ✅ Done |
| 2 | [`02_random_walk`](02_random_walk/) | 2D Brownian motion & diffusion | ✅ Done |
| 3 | `03_nd_integration` | High-dimensional volume integration | 🔜 Planned |
| 4 | `04_ising_model` | 2D Ising model (Metropolis algorithm) | 🔜 Planned |

---

## 01 — Monte Carlo Estimation of π

The classic Monte Carlo experiment: sample random points in the unit square and check whether they fall inside the quarter unit circle.

$$\pi \approx 4 \cdot \frac{\text{points inside circle}}{\text{total points}}$$

### Result (n = 50,000)

<p align="center">
  <img src="01_pi_approximation/figures/pi_approximation.png" width="600" alt="Monte Carlo π approximation scatter plot">
</p>

### Key Design Decisions

- **Vectorised computation** — all coordinate generation and distance checks use NumPy array operations, making the simulation ~50–100× faster than a Python loop.
- **Separation of concerns** — `calculate_pi()` returns raw data; `plot_pi_approximation()` handles visualisation. Both can be used independently.
- **Reproducibility** — a fixed RNG seed (`numpy.random.default_rng(42)`) ensures deterministic results.

---

## 02 — 2D Random Walk & Diffusion Validation

Simulates **10,000 Brownian particles** over 1,000 time steps using fully vectorised Gaussian random walks.  The Mean Squared Displacement (MSD) is computed and fitted to validate the theoretical diffusion law $\text{MSD}(t) = 4Dt$.

### Static Analysis

<p align="center">
  <img src="02_random_walk/figures/random_walk_analysis.png" width="800" alt="Random walk trajectories and MSD analysis">
</p>

### Diffusion Animation

<p align="center">
  <img src="02_random_walk/figures/diffusion_animation.gif" width="450" alt="Animated Brownian diffusion of 500 particles">
</p>

### Key Design Decisions

- **Zero loops** — all 10,000 × 1,000 steps are drawn as a single `np.random.normal` call (shape `(N, M, 2)`) and accumulated via `np.cumsum`. The 160 MB trajectory array is computed in under a second.
- **Diffusion proof** — linear regression on MSD yields $D_{approx} \approx 0.4955$ vs. $D_{theo} = 0.5000$ with $R^2 = 0.9999$, confirming correct Brownian dynamics.
- **Animation as GIF** — saved via Pillow for direct embedding in this README.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/calvinkaya/data-analysis-projects.git
cd data-analysis-projects

# Install dependencies
pip install numpy matplotlib

# Run individual modules
python 01_pi_approximation/pi_monte_carlo.py
python 02_random_walk/random_walk.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.12+** | Core language |
| **NumPy** | Vectorised numerical computation |
| **Matplotlib** | Scientific plotting, animation & visualisation |

---

## Project Structure

```
├── README.md
├── 01_pi_approximation/
│   ├── pi_monte_carlo.py
│   └── figures/
│       └── pi_approximation.png
├── 02_random_walk/
│   ├── random_walk.py
│   └── figures/
│       ├── random_walk_analysis.png
│       └── diffusion_animation.gif
├── 03_nd_integration/            (planned)
├── 04_ising_model/               (planned)
└── utils/                        (planned)
```

---

## License

This project is part of a personal portfolio. Feel free to use it as inspiration for your own simulations.
