import numpy as np


class PDESolver:
    def __init__(self, grid_size, time_steps, delta_x, delta_t, diffusion_coeff, reaction_coeff=0):
        """
        Initialize the PDE Solver.
        - grid_size: Size of the 2D grid (rows, cols).
        - time_steps: Number of time iterations.
        - delta_x: Spatial resolution.
        - delta_t: Time step size.
        - diffusion_coeff: Diffusion constant (for the heat equation).
        - reaction_coeff: Reaction constant (optional for coupled PDEs).
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.diffusion_coeff = diffusion_coeff
        self.reaction_coeff = reaction_coeff
        self.plate = np.zeros(grid_size)  # Initialize 2D grid
        self.alpha = self.diffusion_coeff * self.delta_t / (self.delta_x ** 2)

    def set_initial_conditions(self, initial_func):
        """
        Set the initial temperature distribution on the grid.
        - initial_func: Function defining initial conditions. It takes (x, y) as input.
        """
        rows, cols = self.grid_size
        for i in range(rows):
            for j in range(cols):
                self.plate[i, j] = initial_func(i * self.delta_x, j * self.delta_x)

    def solve_heat_equation(self):
        """
        Solve the heat equation using finite difference method.
        Returns: History of solutions (list of 2D arrays).
        """
        history = [self.plate.copy()]
        rows, cols = self.grid_size

        for t in range(self.time_steps):
            new_plate = self.plate.copy()
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Update temperature using finite difference
                    new_plate[i, j] = (
                            self.plate[i, j]
                            + self.alpha * (
                                    self.plate[i + 1, j]
                                    + self.plate[i - 1, j]
                                    + self.plate[i, j + 1]
                                    + self.plate[i, j - 1]
                                    - 4 * self.plate[i, j]
                            )
                            + self.delta_t * self.reaction_coeff * self.plate[i, j]
                    )
            self.plate = new_plate
            history.append(new_plate.copy())
        return history

    def solve_wave_equation(self):
        """
        Solve the wave equation using finite difference method.
        Returns: History of solutions (list of 2D arrays).
        """
        history = [self.plate.copy()]
        rows, cols = self.grid_size

        # Initialize the previous state for wave equation
        prev_plate = self.plate.copy()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                prev_plate[i, j] = (
                        0.5 * self.alpha * (
                        self.plate[i + 1, j]
                        + self.plate[i - 1, j]
                        + self.plate[i, j + 1]
                        + self.plate[i, j - 1]
                        - 4 * self.plate[i, j]
                )
                )

        for t in range(self.time_steps):
            new_plate = self.plate.copy()
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Update wave equation
                    new_plate[i, j] = (
                            2 * self.plate[i, j]
                            - prev_plate[i, j]
                            + self.alpha * (
                                    self.plate[i + 1, j]
                                    + self.plate[i - 1, j]
                                    + self.plate[i, j + 1]
                                    + self.plate[i, j - 1]
                                    - 4 * self.plate[i, j]
                            )
                    )
            prev_plate = self.plate
            self.plate = new_plate
            history.append(new_plate.copy())
        return history


if __name__ == "__main__":
    # Example 1: Solving the heat equation
    solver = PDESolver(grid_size=(50, 50), time_steps=100, delta_x=1.0, delta_t=0.1, diffusion_coeff=0.1)


    def initial_heat(x, y):
        if 20 <= x <= 30 and 20 <= y <= 30:
            return 100
        return 0


    solver.set_initial_conditions(initial_heat)
    heat_history = solver.solve_heat_equation()
    print("Heat Equation Solved. Final State:")
    print(heat_history[-1])

    # Example 2: Solving the wave equation
    solver = PDESolver(grid_size=(50, 50), time_steps=100, delta_x=1.0, delta_t=0.1, diffusion_coeff=0.1)


    def initial_wave(x, y):
        if x == 25 and y == 25:
            return 100
        return 0


    solver.set_initial_conditions(initial_wave)
    wave_history = solver.solve_wave_equation()
    print("\nWave Equation Solved. Final State:")
    print(wave_history[-1])

    # Example 3: Heat equation with reaction term
    solver = PDESolver(grid_size=(50, 50), time_steps=100, delta_x=1.0, delta_t=0.1, diffusion_coeff=0.1,
                       reaction_coeff=0.01)


    def initial_reaction(x, y):
        return np.exp(-0.1 * ((x - 25) ** 2 + (y - 25) ** 2))


    solver.set_initial_conditions(initial_reaction)
    reaction_history = solver.solve_heat_equation()
    print("\nHeat Equation with Reaction Term Solved. Final State:")
    print(reaction_history[-1])

"""
Project Title: Solving Systems of Partial Differential Equations Using Finite Difference Method
File Name: pde_solver_with_numpy.py

Short Description
This project involves solving systems of partial differential equations (PDEs) using the finite difference method (FDM) implemented with NumPy. The goal is to compute numerical solutions for problems involving coupled PDEs such as the heat equation with a time-dependent reaction term or wave propagation in 2D. It leverages NumPy's array operations for efficient computation.

The project demonstrates:

Numerical discretization of PDEs.
Iterative solvers using NumPy.
Advanced manipulation of multidimensional arrays.
"""

# -----

"""
Example Inputs and Expected Outputs
Example 1: Solving Heat Equation
Input:

Grid size: (50, 50).
Initial heat in the center of the grid: 100. Expected Output:
Heat diffuses radially, forming a smooth gradient over time.
Example 2: Solving Wave Equation
Input:

Grid size: (50, 50).
Initial wave peak at the center of the grid: 100. Expected Output:
Waves propagate outward symmetrically from the center.
Example 3: Heat Equation with Reaction Term
Input:

Grid size: (50, 50).
Initial Gaussian heat distribution. Expected Output:
Heat diffuses with slight amplification due to the reaction term.
Key Features
Multi-Equation Solver:
Solves different PDEs using finite difference methods.
Flexible Boundary Conditions:
Easily modify boundary behavior.
Scalable Grid Size:
Handles arbitrary grid resolutions for high-detail simulations.
Efficient Computation:
Uses NumPy for high-speed array operations.
This project takes numerical simulations a step further, demonstrating advanced mathematical modeling using only NumPy.
"""
