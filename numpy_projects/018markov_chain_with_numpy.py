import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix):
        """
        Initialize the Markov Chain.
        - transition_matrix: A square matrix of shape (n_states, n_states) representing
          transition probabilities between states.
        """
        self.transition_matrix = np.array(transition_matrix)
        self.n_states = self.transition_matrix.shape[0]
        self.validate_transition_matrix()

    def validate_transition_matrix(self):
        """
        Validate the transition matrix to ensure rows sum to 1.
        """
        if not np.allclose(self.transition_matrix.sum(axis=1), 1):
            raise ValueError("Each row of the transition matrix must sum to 1.")

    def next_state(self, current_state):
        """
        Compute the next state based on the current state.
        - current_state: An integer representing the current state index.
        Returns: An integer representing the next state index.
        """
        return np.random.choice(self.n_states, p=self.transition_matrix[current_state])

    def simulate(self, start_state, n_steps):
        """
        Simulate the Markov Chain over a given number of steps.
        - start_state: The initial state index.
        - n_steps: Number of steps to simulate.
        Returns: A list of states visited during the simulation.
        """
        states = [start_state]
        current_state = start_state
        for _ in range(n_steps):
            current_state = self.next_state(current_state)
            states.append(current_state)
        return states

    def stationary_distribution(self, tol=1e-8, max_iterations=1000):
        """
        Compute the stationary distribution of the Markov Chain.
        - tol: Tolerance for convergence.
        - max_iterations: Maximum number of iterations.
        Returns: A 1D array representing the stationary distribution.
        """
        dist = np.random.rand(self.n_states)
        dist /= dist.sum()  # Normalize to sum to 1
        for _ in range(max_iterations):
            new_dist = np.dot(dist, self.transition_matrix)
            if np.allclose(new_dist, dist, atol=tol):
                break
            dist = new_dist
        return dist


if __name__ == "__main__":
    # Example 1: Weather states (Sunny, Rainy)
    transition_matrix = [
        [0.8, 0.2],  # Sunny to Sunny, Sunny to Rainy
        [0.4, 0.6]   # Rainy to Sunny, Rainy to Rainy
    ]
    mc = MarkovChain(transition_matrix)
    simulation = mc.simulate(start_state=0, n_steps=10)
    print("Simulation (Example 1):", simulation)
    print("Stationary Distribution (Example 1):", mc.stationary_distribution())

    # Example 2: Market states (Bull, Bear, Stagnant)
    transition_matrix = [
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.3, 0.4]
    ]
    mc = MarkovChain(transition_matrix)
    simulation = mc.simulate(start_state=1, n_steps=15)
    print("\nSimulation (Example 2):", simulation)
    print("Stationary Distribution (Example 2):", mc.stationary_distribution())

    # Example 3: Traffic states (Light, Moderate, Heavy)
    transition_matrix = [
        [0.5, 0.4, 0.1],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ]
    mc = MarkovChain(transition_matrix)
    simulation = mc.simulate(start_state=2, n_steps=20)
    print("\nSimulation (Example 3):", simulation)
    print("Stationary Distribution (Example 3):", mc.stationary_distribution())

# Markov Chain Simulation Using NumPy
"""
Project Title: Markov Chain Simulation Using NumPy
File Name: markov_chain_with_numpy.py

Short Description
This project involves implementing a Markov Chain, a stochastic model used to predict the behavior of systems that move between states with certain probabilities. Markov Chains are widely used in fields such as finance, genetics, and text generation.

The implementation uses NumPy for matrix manipulations to simulate the transition between states over multiple steps. The user can define a transition probability matrix and simulate how the states evolve over time. The project also includes functionality to compute the stationary distribution of the Markov Chain.

"""

# Example Inputs and Expected Outputs
"""
Example 1: Weather Simulation
Input:

python
Copy code
transition_matrix = [
    [0.8, 0.2],
    [0.4, 0.6]
]
start_state = 0  # Start in Sunny
n_steps = 10
Expected Output:

plaintext
Copy code
Simulation: A sequence of 11 states such as [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1].
Stationary Distribution: A 1D array, e.g., [0.6667, 0.3333], indicating long-term probabilities of Sunny and Rainy.
Example 2: Market States
Input:

python
Copy code
transition_matrix = [
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.3, 0.4]
]
start_state = 1  # Start in Bear market
n_steps = 15
Expected Output:

plaintext
Copy code
Simulation: A sequence of 16 states, e.g., [1, 1, 2, 1, 1, 0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0].
Stationary Distribution: A 1D array, e.g., [0.385, 0.487, 0.128], indicating long-term probabilities of Bull, Bear, and Stagnant markets.
Example 3: Traffic Simulation
Input:

python
Copy code
transition_matrix = [
    [0.5, 0.4, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
]
start_state = 2  # Start in Heavy traffic
n_steps = 20
Expected Output:

plaintext
Copy code
Simulation: A sequence of 21 states, e.g., [2, 1, 2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 1, 0, 1].
Stationary Distribution: A 1D array, e.g., [0.315, 0.453, 0.232], indicating long-term probabilities of Light, Moderate, and Heavy traffic.
Key Features
State Transition Simulation:
Models how systems evolve over time based on given probabilities.
Stationary Distribution:
Calculates the long-term behavior of the system.
Custom Transition Matrices:
Supports any number of states and transitions.
Robust Validation:
Ensures the transition matrix is well-formed with rows summing to 1.
This project is more advanced and delves into stochastic processes and probabilistic modeling. It demonstrates NumPy's power in handling matrix operations and random sampling effectively.
"""