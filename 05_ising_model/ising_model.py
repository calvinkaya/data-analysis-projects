import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple

class IsingLattice:
    """
    Represents a 2D square lattice for the Ising model simulation.
    Manages the physical state, boundary conditions, and energy calculations.
    """
    def __init__(self, size: int):
        self.size = size
        # Random initial state: hot start (T = infinity)
        self.grid: np.ndarray = np.random.choice([-1, 1], size=(size, size))

    def calculate_energy_per_spin(self) -> float:
        """
        Calculates the total macroscopic energy per spin using periodic boundary conditions.
        """
        # Sum interactions with right and bottom neighbors to avoid double counting
        interactions = self.grid * (np.roll(self.grid, 1, axis=0) + np.roll(self.grid, 1, axis=1))
        total_energy = -float(np.sum(interactions))
        return total_energy / (self.size * self.size)

    def calculate_magnetization_per_spin(self) -> float:
        """
        Calculates the net magnetization per spin of the lattice.
        """
        return float(np.sum(self.grid)) / (self.size * self.size)


class MetropolisSimulator:
    """
    Executes the Monte Carlo Metropolis-Hastings algorithm on an IsingLattice.
    Utilizes a vectorized checkerboard approach for high performance.
    """
    def __init__(self, lattice: IsingLattice, temperature: float):
        self.lattice = lattice
        self.temperature = temperature
        
        # Pre-compute masks for checkerboard update
        x, y = np.indices((self.lattice.size, self.lattice.size))
        self.black_mask: np.ndarray = (x + y) % 2 == 0
        self.white_mask: np.ndarray = (x + y) % 2 == 1

    def step(self) -> None:
        """Performs one complete Monte Carlo step via two checkerboard passes."""
        self._update_subgrid(self.black_mask)
        self._update_subgrid(self.white_mask)

    def _update_subgrid(self, mask: np.ndarray) -> None:
        neighbors = (
            np.roll(self.lattice.grid, 1, axis=0) +
            np.roll(self.lattice.grid, -1, axis=0) +
            np.roll(self.lattice.grid, 1, axis=1) +
            np.roll(self.lattice.grid, -1, axis=1)
        )
        
        dE = 2.0 * self.lattice.grid * neighbors
        accept_prob = np.where(dE > 0, np.exp(-dE / self.temperature), 1.0)
        random_vals = np.random.rand(self.lattice.size, self.lattice.size)
        
        should_flip = (dE <= 0) | (random_vals < accept_prob)
        spins_to_flip = should_flip & mask
        self.lattice.grid[spins_to_flip] *= -1

    def set_temperature(self, temperature: float) -> None:
        """Dynamically update the simulation temperature."""
        self.temperature = temperature


def analyze_phase_transition(size: int = 100, t_min: float = 1.5, t_max: float = 3.5, points: int = 30) -> None:
    """
    Runs simulations across a range of temperatures to observe the thermodynamic phase transition.
    Generates high-quality, publication-ready plots.
    """
    temperatures = np.linspace(t_min, t_max, points)
    
    # Observables arrays
    E_means = np.zeros(points)
    M_means = np.zeros(points)
    C_v = np.zeros(points)
    Chi = np.zeros(points)
    
    # Simulation parameters
    # High number of sweeps to ensure equilibrium
    thermalization_sweeps = 500
    measurement_sweeps = 1000
    N = size * size
    
    print(f"Starting temperature sweep for L={size} ({points} temperatures)...")
    
    # Start with a cold lattice (all spins aligned) and slowly heat it up
    lattice = IsingLattice(size=size)
    lattice.grid = np.ones((size, size), dtype=int)
    simulator = MetropolisSimulator(lattice=lattice, temperature=temperatures[0])
    
    for i, T in enumerate(temperatures):
        simulator.set_temperature(T)
        
        # Thermalize
        for _ in range(thermalization_sweeps):
            simulator.step()
            
        # Measure
        energies = np.zeros(measurement_sweeps)
        magnetizations = np.zeros(measurement_sweeps)
        
        for step in range(measurement_sweeps):
            simulator.step()
            energies[step] = lattice.calculate_energy_per_spin()
            magnetizations[step] = abs(lattice.calculate_magnetization_per_spin())
            
        # Calculate statistics
        E_means[i] = np.mean(energies)
        M_means[i] = np.mean(magnetizations)
        
        # Fluctuation-dissipation theorem formulas
        C_v[i] = (np.var(energies) * N) / (T**2)
        Chi[i] = (np.var(magnetizations) * N) / T
        
        print(f"T = {T:.2f} | <E> = {E_means[i]:.3f} | <|M|> = {M_means[i]:.3f}")

    # Plotting using aesthetic styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Thermodynamics of 2D Ising Model ($L={size}$)", fontsize=20, fontweight='bold', color='#333333')
    
    colors = ['#2b8cbe', '#e34a33', '#31a354', '#756bb1']
    critical_T = 2.269
    
    plot_configs = [
        (0, 0, E_means, "Energy per Spin", r"$\langle E \rangle / N$", colors[0]),
        (0, 1, M_means, "Absolute Magnetization per Spin", r"$\langle |M| \rangle / N$", colors[1]),
        (1, 0, C_v, "Specific Heat Capacity", r"$C_v$", colors[2]),
        (1, 1, Chi, "Magnetic Susceptibility", r"$\chi$", colors[3])
    ]
    
    for row, col, data, title, ylabel, color in plot_configs:
        ax = axs[row, col]
        # Main data line
        ax.plot(temperatures, data, marker='o', markersize=6, linestyle='-', linewidth=2, color=color, alpha=0.9, label='Simulation Data')
        # Critical temperature reference line
        ax.axvline(x=critical_T, color='#555555', linestyle='--', alpha=0.7, linewidth=1.5, label=f'$T_c \\approx {critical_T}$')
        
        # Aesthetics
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel("Temperature ($T$)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        
    # Save the phase transition plot
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    output_path = os.path.join(figures_dir, "phase_transition.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Phase transition plot saved to {output_path}")


def generate_animation() -> None:
    """
    Generates an aesthetic simulation GIF of the spin domains near critical temperature.
    """
    size = 100
    temperature = 2.269
    frames = 100
    sweeps_per_frame = 5
    
    lattice = IsingLattice(size=size)
    simulator = MetropolisSimulator(lattice=lattice, temperature=temperature)
    
    # Thermalize to form some initial domains
    for _ in range(50):
        simulator.step()
        
    # Reset style for the image map to avoid grid lines bleeding in
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.axis('off')
    
    title = ax.set_title(f"Spin Domain Evolution ($T = {temperature}$)", fontsize=16, fontweight='bold', pad=15)
    
    # High-contrast colormap for spins
    img = ax.imshow(lattice.grid, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    
    def update(frame: int) -> Tuple:
        for _ in range(sweeps_per_frame):
            simulator.step()
        img.set_data(lattice.grid)
        return (img,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    output_path = os.path.join(figures_dir, "ising_simulation.gif")
    
    print(f"Generating animation ({frames} frames)...")
    ani.save(output_path, writer='pillow', fps=20)
    print(f"[SUCCESS] Animation saved to {output_path}")


if __name__ == "__main__":
    # Execute the thermodynamic phase transition analysis with full lattice
    analyze_phase_transition(size=100, t_min=1.5, t_max=3.5, points=30)
    
    # Generate the aesthetic visual simulation GIF
    generate_animation()
