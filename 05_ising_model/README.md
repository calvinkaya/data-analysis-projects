# 05 — 2D Ising Model & Metropolis Algorithm

This module simulates the 2D Ising model using the Metropolis-Hastings Monte Carlo algorithm. It demonstrates object-oriented architecture and NumPy vectorization techniques for Markov Chain Monte Carlo (MCMC) simulations.

## Theory

The Ising model is a mathematical model of ferromagnetism in statistical mechanics. It consists of discrete variables representing magnetic dipole moments of atomic spins. 

Spins $S_i$ take values $+1$ (up) or $-1$ (down). The total energy $E$ (Hamiltonian) of the system is:

$$E = -J \sum_{\langle i, j \rangle} S_i S_j$$

*   $J$ is the interaction constant. $J > 0$ models ferromagnetism (spins align).
*   $\langle i, j \rangle$ denotes summation over nearest-neighbor pairs.

### Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm generates a sequence of states following the Boltzmann distribution. To transition to a new state:
1.  Propose a state change by flipping a single spin.
2.  Calculate the energy difference $\Delta E$ between the proposed state and the current state.
3.  If $\Delta E \le 0$, accept the flip.
4.  If $\Delta E > 0$, accept the flip with probability $P = \exp(-\Delta E / T)$, where $T$ is the temperature (assuming $k_B = 1$).

### Checkerboard Vectorization

MCMC steps are inherently sequential. To utilize NumPy vectorization, this implementation uses the **checkerboard algorithm**. 
The 2D square lattice is divided into two independent subgrids (black and white squares). Spins on the black subgrid only interact with spins on the white subgrid. This structural property allows the algorithm to update all spins of one color simultaneously.

## Output

The code outputs an animated GIF of the spin lattice dynamics near the critical temperature $T_c \approx 2.269$.

<p align="center">
  <img src="figures/ising_simulation.gif" width="400" alt="2D Ising Model Simulation">
</p>
