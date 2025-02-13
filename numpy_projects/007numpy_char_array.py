import numpy as np


# 1a. Creating a character array for a list of names and standardizing their case
def example_1a():
    names = ["Alice", "BOB", "Charlie", "diana"]
    char_array = np.char.array(names)
    standardized = char_array.capitalize()  # Capitalize all names
    print("Standardized Names:", standardized)
    # Output: ['Alice', 'Bob', 'Charlie', 'Diana']


# 1b. Combining first and last names in a character array
def example_1b():
    first_names = np.char.array(["John", "Jane", "Jim", "Jill"])
    last_names = np.char.array(["Doe", "Smith", "Brown", "Taylor"])
    full_names = np.char.add(first_names, " ") + last_names
    print("Full Names:", full_names)
    # Output: ['John Doe', 'Jane Smith', 'Jim Brown', 'Jill Taylor']


# 1c. Creating repeated patterns in a character array
def example_1c():
    pattern = np.char.array(["Hi", "Bye"])
    repeated = np.char.multiply(pattern, 3)
    print("Repeated Patterns:", repeated)
    # Output: ['HiHiHi', 'ByeByeBye']


# 1d. Removing prefixes from strings in a character array
def example_1d():
    emails = np.char.array(["team_john@example.com", "team_jane@example.com"])
    cleaned_emails = np.char.replace(emails, "team_", "")
    print("Cleaned Emails:", cleaned_emails)
    # Output: ['john@example.com', 'jane@example.com']


# 2a. Converting a Python list of strings to a character array
def example_2a():
    list_data = ["alpha", "beta", "gamma"]
    char_array = np.char.asarray(list_data)
    print("Character Array:", char_array)
    # Output: ['alpha' 'beta' 'gamma']


# 2b. Converting a NumPy array of mixed data to a string representation
def example_2b():
    numeric_array = np.array([1, 2, 3, 4])
    string_array = np.char.asarray(numeric_array.astype(str))
    print("String Representation:", string_array)
    # Output: ['1' '2' '3' '4']


# 2c. Using asarray for a uniform character array
def example_2c():
    data = [["one", "two"], ["three", "four"]]
    char_array = np.char.asarray(data)
    print("Character Array (2D):", char_array)
    # Output: [['one' 'two']
    #          ['three' 'four']]


# 2d. Converting integers to padded strings
def example_2d():
    numbers = np.array([1, 20, 300])
    padded = np.char.asarray([str(x).zfill(4) for x in numbers])
    print("Padded Strings:", padded)
    # Output: ['0001', '0020', '0300']


# 3a. Generate an arithmetic sequence
def example_3a():
    sequence = np.arange(0, 20, 5)
    print("Arithmetic Sequence:", sequence)
    # Output: [0 5 10 15]


# 3b. Generate a grid of points for a 2D plane
def example_3b():
    x = np.arange(-5, 6, 1)
    y = np.arange(-5, 6, 1)
    grid = np.meshgrid(x, y)
    print("2D Grid (X):", grid[0])
    print("2D Grid (Y):", grid[1])
    # Output: A 2D grid with X and Y coordinates from -5 to 5.


# 3c. Create a time array for evenly spaced intervals in a simulation
def example_3c():
    time_points = np.arange(0, 10, 0.1)
    print("Time Points:", time_points)
    # Output: [0. 0.1 0.2 ... 9.8 9.9]


# 3d. Generate indices for slicing a data array
def example_3d():
    data = np.arange(1, 101)
    indices = np.arange(0, 100, 10)
    print("Data:", data[indices])
    # Output: Every 10th element from the data array.


# Run all examples
example_1a()
example_1b()
example_1c()
example_1d()
example_2a()
example_2b()
example_2c()
example_2d()
example_3a()
example_3b()
example_3c()
example_3d()
