import numpy as np


# array()
def array_exercises():
    print("Exercise 1: Create a 1D array from a list")
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)
    """
    [1 2 3 4 5]
    """

    print("\nExercise 2: Create a 2D array from a nested list")
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    print(arr)
    """
    [[1 2]
     [3 4]
     [5 6]]
    """

    print("\nExercise 3: Create an array with specified dtype")
    arr = np.array([1, 2, 3], dtype=float)
    print(arr)
    """
    [1. 2. 3.]
    """


# asarray()
def asarray_exercises():
    print("\nExercise 1: Convert a list to an array")
    lst = [1, 2, 3, 4]
    arr = np.asarray(lst)
    print(arr)
    """
    [1 2 3 4]
    """

    print("\nExercise 2: Convert a tuple to an array")
    tpl = (10, 20, 30)
    arr = np.asarray(tpl)
    print(arr)
    """
    [10 20 30]
    """

    print("\nExercise 3: Use 'copy' parameter")
    x = np.array([1, 2, 3])
    arr = np.asarray(x)
    print(arr)
    """
    [1, 2, 3]
    """


# asanyarray()
def asanyarray_exercises():
    print("\nExercise 1: Convert a list to an array")
    lst = [10, 20, 30]
    arr = np.asanyarray(lst)
    print(arr)
    """
    [10 20 30]
    """

    print("\nExercise 2: Pass a numpy subclass through")
    class MyArray(np.ndarray): pass

    x = np.arange(4).view(MyArray)
    arr = np.asanyarray(x)
    print(arr)
    """
    [0 1 2 3]
    """


# ascontiguousarray()
def ascontiguousarray_exercises():
    print("\nExercise 1: Convert a non-contiguous array")
    arr = np.arange(12).reshape(3, 4).T
    print("Original array (non-contiguous):")
    print(arr)
    """
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    """

    contiguous = np.ascontiguousarray(arr)
    print("Contiguous array:")
    print(contiguous)
    """
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    """


# asmatrix()
def asmatrix_exercises():
    print("\nExercise 1: Convert an array to a matrix")
    arr = np.array([[1, 2], [3, 4]])
    matrix = np.asmatrix(arr)
    print(matrix)
    """
    [[1 2]
    [3 4]]
    """


# astype()
def astype_exercises():
    print("\nExercise 1: Change dtype of an array")
    arr = np.array([1.5, 2.7, 3.2])
    new_arr = arr.astype(int)
    print(new_arr)
    """
    [1 2 3]
    """


# copy()
def copy_exercises():
    print("\nExercise 1: Create a copy of an array")
    arr = np.array([1, 2, 3])
    copy_arr = np.copy(arr)
    print(copy_arr)
    """
    [1 2 3]
    """


# frombuffer()
def frombuffer_exercises():
    print("\nExercise 1: Interpret a buffer as an array")
    buf = b'hello world'
    arr = np.frombuffer(buf, dtype='S1')
    print(arr)
    """
    [b'h' b'e' b'l' b'l' b'o' b' ' b'w' b'o' b'r' b'l' b'd']
    """


# from_dlpack()
def from_dlpack_exercises():
    print("\nExercise 1: Create a NumPy array from a __dlpack__ object (Example not runnable here)")


# fromfile()
def fromfile_exercises():
    print("\nExercise 1: Load data from a binary file (Requires file creation)")
    # Uncomment if you have a file ready to test
    arr = np.fromfile('data.bin', dtype=np.int32)
    print(arr)
    """
    [539767131 857746482 741613612]
    """


# fromfunction()
def fromfunction_exercises():
    print("\nExercise 1: Generate an array with a custom function")
    func = lambda x, y: x + y
    arr = np.fromfunction(func, (3, 3))
    print(arr)
    """
    [[0. 1. 2.]
     [1. 2. 3.]
     [2. 3. 4.]]
    """


# fromiter()
def fromiter_exercises():
    print("\nExercise 1: Create an array from an iterable")
    iterable = range(5)
    arr = np.fromiter(iterable, dtype=int)
    print(arr)
    """
    [0 1 2 3 4]
    """


# fromstring()
def fromstring_exercises():
    print("\nExercise 1: Create an array from a string")
    s = "1 2 3 4"
    arr = np.fromstring(s, dtype=int, sep=' ')
    print(arr)
    """
    [1 2 3 4]
    """


# loadtxt()
def loadtxt_exercises():
    print("\nExercise 1: Load data from a text file (Requires file creation)")
    # Uncomment if you have a file ready to test
    arr = np.loadtxt('data.txt', delimiter=',')
    print(arr)
    """
    [[1. 2. 3.]
     [4. 5. 6.]
     [7. 8. 9.]]
    """


# Run all exercises
if __name__ == "__main__":
    print("=== NumPy Array Creation Exercises ===\n")

    array_exercises()
    asarray_exercises()
    asanyarray_exercises()
    ascontiguousarray_exercises()
    asmatrix_exercises()
    astype_exercises()
    copy_exercises()
    frombuffer_exercises()
    from_dlpack_exercises()
    fromfile_exercises()
    fromfunction_exercises()
    fromiter_exercises()
    fromstring_exercises()
    loadtxt_exercises()
