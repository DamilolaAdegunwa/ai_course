import numpy as np
import matplotlib.pyplot as plt


class HeatDiffusionSimulator:
    def __init__(self, grid_size, alpha, time_step, total_time, boundary_temp=0):
        """
        Initialize the heat diffusion simulator.
        - grid_size: Tuple (rows, cols) for the simulation grid.
        - alpha: Thermal diffusivity constant.
        - time_step: Simulation time step.
        - total_time: Total time to run the simulation.
        - boundary_temp: Fixed temperature at the boundaries.
        """
        self.grid_size = grid_size
        self.alpha = alpha
        self.time_step = time_step
        self.total_time = total_time
        self.boundary_temp = boundary_temp
        self.plate = np.zeros(grid_size)  # Initialize plate with zero temperature
        self.time_steps = int(total_time / time_step)

    def set_initial_conditions(self, heat_sources):
        """
        Set initial heat sources on the plate.
        - heat_sources: List of tuples (row, col, temp) indicating heat sources.
        """
        for row, col, temp in heat_sources:
            self.plate[row, col] = temp

    def simulate(self):
        """
        Run the heat diffusion simulation.
        Returns:
            - history: List of 2D NumPy arrays representing plate state at each time step.
        """
        rows, cols = self.grid_size
        history = [self.plate.copy()]
        for _ in range(self.time_steps):
            # Create a copy to compute the next state
            new_plate = self.plate.copy()
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Compute heat diffusion based on neighbors
                    new_plate[i, j] = (
                            self.plate[i, j]
                            + self.alpha * self.time_step * (
                                    self.plate[i + 1, j]
                                    + self.plate[i - 1, j]
                                    + self.plate[i, j + 1]
                                    + self.plate[i, j - 1]
                                    - 4 * self.plate[i, j]
                            )
                    )
            # Apply boundary conditions
            new_plate[0, :] = self.boundary_temp
            new_plate[-1, :] = self.boundary_temp
            new_plate[:, 0] = self.boundary_temp
            new_plate[:, -1] = self.boundary_temp
            self.plate = new_plate
            history.append(new_plate.copy())
        return history

    def visualize(self, history):
        """
        Visualize the heat diffusion process.
        - history: List of 2D NumPy arrays representing plate state at each time step.
        """
        for step, plate_state in enumerate(history):
            plt.imshow(plate_state, cmap='hot', interpolation='nearest')
            plt.title(f"Time Step {step}")
            plt.colorbar(label="Temperature")
            plt.pause(0.1)
        plt.show()


if __name__ == "__main__":
    # Parameters
    grid_size = (50, 50)  # 50x50 grid
    alpha = 0.1  # Thermal diffusivity
    time_step = 0.1  # Time step in seconds
    total_time = 5.0  # Total simulation time in seconds
    boundary_temp = 0  # Fixed boundary temperature

    # Initialize simulator
    simulator = HeatDiffusionSimulator(grid_size, alpha, time_step, total_time, boundary_temp)

    # Set initial heat sources
    heat_sources = [(25, 25, 100), (10, 10, 75), (40, 40, 50)]  # (row, col, temperature)
    simulator.set_initial_conditions(heat_sources)

    # Run simulation
    history = simulator.simulate()

    # Visualize the heat diffusion
    simulator.visualize(history)
