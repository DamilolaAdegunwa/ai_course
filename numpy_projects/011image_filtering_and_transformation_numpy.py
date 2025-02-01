import numpy as np
from scipy.ndimage import gaussian_filter

class ImageProcessor:
    @staticmethod
    def apply_filter(image, filter_matrix):
        """
        Applies a custom filter to a 2D grayscale image.
        """
        rows, cols = image.shape
        k_rows, k_cols = filter_matrix.shape
        pad_height = k_rows // 2
        pad_width = k_cols // 2

        # Padding the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Applying filter
        filtered_image = np.zeros_like(image)
        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i + k_rows, j:j + k_cols]
                filtered_image[i, j] = np.sum(region * filter_matrix)

        return np.clip(filtered_image, 0, 255)

    @staticmethod
    def rotate_image(image, angle):
        """
        Rotates an image by the given angle (in degrees).
        """
        angle_rad = np.deg2rad(angle)
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

        rows, cols = image.shape
        center = (rows // 2, cols // 2)

        # Create an empty array for the rotated image
        rotated_image = np.zeros_like(image)

        for i in range(rows):
            for j in range(cols):
                # Translate point to origin
                y, x = i - center[0], j - center[1]

                # Rotate point
                x_rot = int(x * cos_theta - y * sin_theta + center[1])
                y_rot = int(x * sin_theta + y * cos_theta + center[0])

                # Check if within bounds
                if 0 <= x_rot < cols and 0 <= y_rot < rows:
                    rotated_image[y_rot, x_rot] = image[i, j]

        return rotated_image

    @staticmethod
    def flip_image(image, axis):
        """
        Flips the image along the specified axis.
        - axis=0: Vertical flip
        - axis=1: Horizontal flip
        """
        return np.flip(image, axis=axis)

# Example Filters
GAUSSIAN_KERNEL = gaussian_filter(np.eye(3), sigma=1)
EDGE_DETECTION_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

if __name__ == "__main__":
    # Example Image
    image_old = np.array([
        [200, 200, 200, 200, 200],
        [200, 0, 0, 0, 200],
        [200, 0, 200, 0, 200],
        [200, 0, 0, 0, 200],
        [200, 200, 200, 200, 200]
    ])

    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])

    # Apply Edge Detection Filter
    print("Applying Edge Detection Filter:")
    filtered_image = ImageProcessor.apply_filter(image, EDGE_DETECTION_KERNEL)
    print(filtered_image)

    # Rotate Image by 90 degrees
    print("\nRotating Image by 90 Degrees:")
    rotated_image = ImageProcessor.rotate_image(image, 90)
    print(rotated_image)

    # Flip Image Vertically
    print("\nFlipping Image Vertically:")
    flipped_image = ImageProcessor.flip_image(image, axis=0)
    print(flipped_image)

    # Flip Image Horizontally
    print("\nFlipping Image Horizontally:")
    flipped_image_horizontal = ImageProcessor.flip_image(image, axis=1)
    print(flipped_image_horizontal)

#DESCRIPTION
"""
Project Title: Advanced Image Filtering and Transformation with NumPy
File Name: image_filtering_and_transformation_numpy.py

Short Description
This project focuses on creating a robust system for applying advanced image processing techniques using NumPy. You will:

Implement custom filters like Gaussian blur, edge detection, and sharpening.
Perform image transformations such as rotation, scaling, and flipping.
Combine filtering and transformations to create a pipeline for advanced image manipulation.
By leveraging NumPy, this project avoids reliance on dedicated image libraries, showcasing the power of NumPy for numerical computation and array manipulation.

Python Code
"""

#Example Inputs and Expected Outputs
"""
Example 1: Apply a Filter
Input:

python
Copy code
image = np.array([
    [200, 200, 200, 200, 200],
    [200, 0, 0, 0, 200],
    [200, 0, 200, 0, 200],
    [200, 0, 0, 0, 200],
    [200, 200, 200, 200, 200]
])
filter_matrix = EDGE_DETECTION_KERNEL

filtered_image = ImageProcessor.apply_filter(image, filter_matrix)
print(filtered_image)
Expected Output:

plaintext
Copy code
[[0, 800, 800, 800, 0],
 [800, -1600, -1200, -1600, 800],
 [800, -1200, 1600, -1200, 800],
 [800, -1600, -1200, -1600, 800],
 [0, 800, 800, 800, 0]]
Example 2: Rotate an Image
Input:

python
Copy code
rotated_image = ImageProcessor.rotate_image(image, 90)
print(rotated_image)
Expected Output:

plaintext
Copy code
[[200, 200, 200, 200, 200],
 [200, 0, 0, 0, 200],
 [200, 0, 0, 0, 200],
 [200, 0, 0, 0, 200],
 [200, 200, 200, 200, 200]]
Example 3: Flip an Image
Input:

python
Copy code
flipped_image = ImageProcessor.flip_image(image, axis=0)
print(flipped_image)
Expected Output:

plaintext
Copy code
[[200, 200, 200, 200, 200],
 [200, 0, 0, 0, 200],
 [200, 0, 200, 0, 200],
 [200, 0, 0, 0, 200],
 [200, 200, 200, 200, 200]]
This project is designed to challenge your understanding of matrix operations, image processing fundamentals, and numerical computation. You can extend it further by adding:

More filters.
Combining transformations (e.g., rotate and scale simultaneously).
Processing colored images by applying these methods to each channel.
"""