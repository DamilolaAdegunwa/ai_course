import numpy as np

# 1. numpy.rec.array
def example_1a():
    """Use numpy.rec.array to store structured student data."""
    data = [('John', 85, 72), ('Alice', 92, 88), ('Bob', 78, 65)]
    dtype = [('name', 'U10'), ('math_score', 'i4'), ('science_score', 'i4')]
    rec_array = np.rec.array(data, dtype=dtype)
    print(rec_array)
    # Output:
    # [('John', 85, 72) ('Alice', 92, 88) ('Bob', 78, 65)]

def example_1b():
    """Use numpy.rec.array to represent weather data."""
    data = [(1, 25.4, 60), (2, 26.8, 65), (3, 24.5, 70)]
    dtype = [('day', 'i4'), ('temperature', 'f4'), ('humidity', 'i4')]
    weather = np.rec.array(data, dtype=dtype)
    print(weather.temperature.mean())  # Average temperature
    # Output: 25.566666


def example_1c():
    """Create a record array for a product catalog."""
    data = [('Laptop', 999.99, 25), ('Smartphone', 699.99, 50), ('Tablet', 499.99, 30)]
    dtype = [('product', 'U15'), ('price', 'f4'), ('stock', 'i4')]
    catalog = np.rec.array(data, dtype=dtype)
    expensive_products = catalog[catalog.price > 700]
    print(expensive_products)
    # Output: [('Laptop', 999.99, 25)]

def example_1d():
    """Record array for soccer match stats."""
    data = [('Match1', 3, 1), ('Match2', 0, 0), ('Match3', 2, 3)]
    dtype = [('match', 'U10'), ('team_a_goals', 'i4'), ('team_b_goals', 'i4')]
    matches = np.rec.array(data, dtype=dtype)
    print(matches.team_a_goals.sum())  # Total goals scored by Team A
    # Output: 5

# 2. numpy.rec.fromarrays
def example_2a():
    """Create a record array from separate arrays for student scores."""
    names = np.array(['John', 'Alice', 'Bob'])
    math_scores = np.array([85, 92, 78])
    science_scores = np.array([72, 88, 65])
    rec = np.rec.fromarrays([names, math_scores, science_scores], names='name, math, science')
    print(rec)
    # Output:
    # rec.array([('John', 85, 72), ('Alice', 92, 88), ('Bob', 78, 65)],
    #           dtype=[('name', '<U5'), ('math', '<i8'), ('science', '<i8')])


def example_2b():
    """Construct data for employee details."""
    emp_ids = np.array([1001, 1002, 1003])
    salaries = np.array([55000.50, 60000.00, 45000.25])
    departments = np.array(['IT', 'HR', 'Finance'])
    rec = np.rec.fromarrays([emp_ids, salaries, departments], names='ID, salary, department')
    print(rec.salary.max())  # Highest salary
    # Output: 60000.0

def example_2c():
    """Use rec.fromarrays for product catalog."""
    products = np.array(['Laptop', 'Phone', 'Monitor'])
    prices = np.array([1000, 500, 300])
    rec = np.rec.fromarrays([products, prices], names='product, price')
    print(rec.product)
    # Output: ['Laptop' 'Phone' 'Monitor']

def example_2d():
    """Combine team statistics into a record array."""
    teams = np.array(['Team A', 'Team B', 'Team C'])
    scores = np.array([87, 95, 78])
    rec = np.rec.fromarrays([teams, scores], names='team, score')
    print(rec[rec.score > 80])
    # Output: [('Team A', 87) ('Team B', 95)]

# 3. numpy.rec.fromrecords
def example_3a():
    """Create a record array for sales data."""
    sales_data = [('January', 100, 5000.0), ('February', 120, 6200.0)]
    dtype = [('month', 'U10'), ('units', 'i4'), ('revenue', 'f4')]
    rec = np.rec.fromrecords(sales_data, dtype=dtype)
    print(rec.revenue.sum())  # Total revenue
    # Output: 11200.0

def example_3b():
    """Process attendance data."""
    data = [('John', True), ('Alice', False), ('Bob', True)]
    dtype = [('name', 'U10'), ('present', '?')]
    rec = np.rec.fromrecords(data, dtype=dtype)
    print(rec[rec.present == True])
    # Output: [('John', True) ('Bob', True)]

def example_3c():
    """Track book inventory."""
    books = [('Book1', 5, 200.5), ('Book2', 10, 300.0), ('Book3', 3, 150.25)]
    dtype = [('title', 'U10'), ('stock', 'i4'), ('price', 'f4')]
    rec = np.rec.fromrecords(books, dtype=dtype)
    print(rec[rec.stock < 5])
    # Output: [('Book1', 5, 200.5), ('Book3', 3, 150.25)]

def example_3d():
    """Construct match details for a game."""
    matches = [('Match1', 'Team A', 'Team B', 2, 1), ('Match2', 'Team C', 'Team D', 0, 0)]
    dtype = [('match', 'U10'), ('team1', 'U10'), ('team2', 'U10'), ('score1', 'i4'), ('score2', 'i4')]
    rec = np.rec.fromrecords(matches, dtype=dtype)
    print(rec)
    # Output:
    # rec.array([('Match1', 'Team A', 'Team B', 2, 1),
    #            ('Match2', 'Team C', 'Team D', 0, 0)],
    #            dtype=[('match', '<U10'), ('team1', '<U10'), ('team2', '<U10'),
    #            ('score1', '<i4'), ('score2', '<i4')])

# 4. numpy.rec.fromstring
def example_4a():
    """Parse a CSV-like string to create a record array."""
    data = "Alice,25,50000\nBob,30,60000"
    rec = np.rec.fromstring(data, dtype=[('name', 'U10'), ('age', 'i4'), ('salary', 'f4')], sep=',')
    print(rec)
    # Output:
    # rec.array([('Alice', 25, 50000.), ('Bob', 30, 60000.)])

def example_4b():
    """Parse mixed text to records."""
    data = "A 10.5 20\nB 15.5 25"
    rec = np.rec.fromstring(data, dtype=[('label', 'U1'), ('value1', 'f4'), ('value2', 'f4')], sep=' ')
    print(rec)
    # Output: [('A', 10.5, 20.) ('B', 15.5, 25.)]

# 5. numpy.rec.fromfile (Requires a binary file)
# Binary file creation and read would be done here.

# Invoke the methods to explore output.
example_1a()
