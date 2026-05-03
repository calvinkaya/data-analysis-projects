# Monte Carlo Pi Approximation

This module demonstrates a  Monte Carlo simulation to estimate the mathematical constant Pi using random spatial sampling.

## 🧠 How it Works
This algorithm generates a large number of random points (x, y) strictly within the positive unit square, where both x and y range from 0 to 1. 

It then calculates the squared distance of each point from the origin to determine if it falls inside the inscribed quarter circle (x² + y² ≤ 1.0). 

The ratio of points inside this quarter circle to the total number of generated points approximates the area of the quarter circle (Pi/4). By multiplying this ratio by 4, we extract the estimated value of Pi.

## 🚀 Technical Highlights
* **Vectorization:** Replaces slow iterative loops with native `NumPy` array operations and boolean masking. This allows for the simultaneous calculation of thousands of data points.
* **Modular Architecture:** Strict separation of concerns between the mathematical computation (`calculate_pi`) and data visualization (`plot_pi_approximation`).
* **Type Hinting:** Fully typed arguments and return values (e.g., `np.ndarray`, `plt.Figure`) for better code quality.

## 📊 Visualization
The script dynamically scales marker sizes based on the input count and generates a publication-quality scatter plot to visualize the distribution. The output is automatically saved to the `figures/` directory.

![Pi Approximation Plot](./figures/pi_approximation.png)

## 💻 How to Run
Ensure you have the required libraries installed:
`pip install numpy matplotlib`

Run the script from your terminal:
`pi_monte_carlo.py`
