# Monte Carlo Stock Simulation — Geometric Brownian Motion & VaR

This module simulates 10,000 stock price paths using the Geometric Brownian Motion (GBM) model and quantifies downside risk via Value at Risk (VaR) at the 95 % confidence level.

## 🧠 How it Works
The simulation models daily stock returns using the discretised GBM formula:

$$S(t + \Delta t) = S(t) \cdot \exp\!\Bigl[\bigl(\mu - \tfrac{\sigma^2}{2}\bigr)\Delta t + \sigma\,\sqrt{\Delta t}\;Z\Bigr], \quad Z \sim \mathcal{N}(0,1)$$

All 10,000 paths × 252 trading days are drawn simultaneously as a single NumPy matrix operation. From the resulting distribution of terminal prices, the 5th percentile is extracted to compute the 95 % VaR — the maximum expected loss over one year at the given confidence level.

## 🚀 Technical Highlights
* **Vectorization:** All random shocks are drawn in one `rng.standard_normal((M, N))` call. Log-returns are accumulated via `np.cumsum` and exponentiated — zero Python loops over time steps.
* **Modular Architecture:** Strict separation between simulation (`simulate_gbm`), risk analysis (`compute_var`), and visualization (`plot_gbm_analysis`).
* **Quantitative Finance:** Implements industry-standard GBM discretisation and percentile-based VaR estimation.
* **Type Hinting:** Fully typed arguments and return values (e.g., `np.ndarray`, `plt.Figure`) for better code quality.

## 📊 Visualization
The script generates a two-panel figure: simulated price paths (100 of 10,000) and the terminal price distribution with VaR threshold. The output is automatically saved to the `figures/` directory.

![GBM Analysis](./figures/gbm_analysis.png)

## 📈 Default Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| $S_0$ | 100 € | Initial stock price |
| $\mu$ | 0.08 | Expected annual return (8 %) |
| $\sigma$ | 0.20 | Annual volatility (20 %) |
| $T$ | 1 year | Time horizon (252 trading days) |
| $M$ | 10,000 | Number of simulated paths |

## 💻 How to Run
Ensure you have the required libraries installed:
`pip install numpy matplotlib`

Run the script from your terminal:
`stock_gbm.py`
