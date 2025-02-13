import numpy as np


# 1a. Generate 10 evenly spaced points between 0 and 1
def example_1a():
    points = np.linspace(0, 1, 10)
    print("1a. Evenly spaced points between 0 and 1:", points)
    # Output: [0.  0.111...  0.222...  ... 1.]


# 1b. Generate 5 points between -π and π, used for sine computation
def example_1b():
    angles = np.linspace(-np.pi, np.pi, 5)
    sin_values = np.sin(angles)
    print("1b. Angles:", angles)
    print("1b. Sine Values:", sin_values)
    # Output: Angles: [-3.14... -1.57... 0. ... 3.14...]
    #         Sine Values: [0. ... -1.  0.  1.  0.]


# 1c. Generate time intervals for animation (0 to 2 seconds, 100 frames)
def example_1c():
    time_intervals = np.linspace(0, 2, 100)
    print("1c. Time Intervals:", time_intervals)
    # Output: [0.  0.020... 0.040... ... 2.]


# 2a. Generate logarithmically spaced numbers for plotting
def example_2a():
    log_space = np.logspace(1, 3, 5)
    print("2a. Logarithmically spaced values (10^1 to 10^3):", log_space)
    # Output: [10. 31.62... 100. 316.22... 1000.]


# 2b. Generate frequencies for audio wave processing
def example_2b():
    frequencies = np.logspace(2, 4, 6, base=2)
    print("2b. Logarithmic frequencies (2^2 to 2^4):", frequencies)
    # Output: [4. 5.04... 6.35... 8. 10.07... 16.]


# 2c. Generate decades of numbers for scaling purposes
def example_2c():
    decades = np.logspace(0, 2, 3, dtype=int)
    print("2c. Decade values:", decades)
    # Output: [1 10 100]


# 3a. Generate geometric progression values
def example_3a():
    geom_values = np.geomspace(1, 1000, 4)
    print("3a. Geometric progression values (1 to 1000):", geom_values)
    # Output: [1. 10. 100. 1000.]


# 3b. Generate values for compounding interest over time
def example_3b():
    compounding_values = np.geomspace(100, 10000, 5)
    print("3b. Compounding values:", compounding_values)
    # Output: [100. 316.22... 1000. 3162.27... 10000.]


# 3c. Generate radii values for circular scaling
def example_3c():
    radii = np.geomspace(0.1, 10, 5)
    print("3c. Radii values:", radii)
    # Output: [0.1 0.316... 1. 3.162... 10.]


# 4a. Create a 2D grid for a mathematical function
def example_4a():
    x = np.linspace(-2, 2, 5)
    y = np.linspace(-2, 2, 5)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    print("4a. Meshgrid X:", X)
    print("4a. Meshgrid Y:", Y)
    print("4a. Computed Z (X^2 + Y^2):", Z)
    # Output: X, Y grids and Z values as the sum of squares


# 4b. Simulate a 2D terrain grid
def example_4b():
    x = np.arange(0, 5)
    y = np.arange(0, 5)
    X, Y = np.meshgrid(x, y)
    print("4b. Terrain Grid X:", X)
    print("4b. Terrain Grid Y:", Y)
    # Output: X and Y arrays of integers for the terrain grid


# 4c. Create a grid for a vector field visualization
def example_4c():
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    U, V = np.meshgrid(x, y)
    print("4c. Vector Field U:", U)
    print("4c. Vector Field V:", V)
    # Output: U, V values for the vector field


# 5a. Use mgrid to create a 2D array of coordinates
def example_5a():
    grid = np.mgrid[0:3, 0:3]
    print("5a. 2D Coordinate Grid:", grid)
    # Output: Two arrays representing a grid of coordinates


# 5b. Create a 3D grid of points
def example_5b():
    grid = np.mgrid[0:2, 0:2, 0:2]
    print("5b. 3D Grid Points:", grid)
    # Output: 3D coordinate arrays for x, y, z


# 5c. Create ranges with a step size using mgrid
def example_5c():
    grid = np.mgrid[0:5:2, 0:5:2]
    print("5c. Stepped Grid:", grid)
    # Output: Grid arrays with steps


# 6a. Create a 1D grid using ogrid
def example_6a():
    grid = np.ogrid[0:5:2]
    print("6a. 1D Grid:", grid)
    # Output: One-dimensional array with steps


# 6b. Generate a 2D range using ogrid
def example_6b():
    x, y = np.ogrid[0:5:1, 0:5:1]
    print("6b. OGrid X:", x)
    print("6b. OGrid Y:", y)
    # Output: Sparse 2D grid


# 6c. Use ogrid for efficient grid computation
def example_6c():
    x, y = np.ogrid[-1:1:3j, -1:1:3j]
    print("6c. Sparse OGrid X:", x)
    print("6c. Sparse OGrid Y:", y)
    # Output: Sparse arrays for efficient computation


# Run all examples
example_1a()
example_1b()
example_1c()
example_2a()
example_2b()
example_2c()
example_3a()
example_3b()
example_3c()
example_4a()
example_4b()
example_4c()
example_5a()
example_5b()
example_5c()
example_6a()
example_6b()
example_6c()
