import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

class ImageCompressor:
    def __init__(self, image_path):
        """
        Initialize the ImageCompressor with an image.
        - image_path: Path to the image file.
        """
        self.image_path = image_path
        self.original_image = None
        self.grayscale_image = None

    def load_image(self):
        """
        Load and preprocess the image. Convert to grayscale.
        """
        self.original_image = io.imread(self.image_path)
        self.grayscale_image = color.rgb2gray(self.original_image)

    def compress_image(self, num_singular_values):
        """
        Compress the image using SVD.
        - num_singular_values: Number of singular values to retain.
        Returns:
            - Compressed image matrix.
        """
        # Perform SVD
        U, S, Vt = np.linalg.svd(self.grayscale_image, full_matrices=False)
        # Retain only the specified number of singular values
        S_reduced = np.zeros((num_singular_values, num_singular_values))
        S_reduced[:num_singular_values, :num_singular_values] = np.diag(S[:num_singular_values])
        # Reconstruct the image
        U_reduced = U[:, :num_singular_values]
        Vt_reduced = Vt[:num_singular_values, :]
        compressed_image = U_reduced @ S_reduced @ Vt_reduced
        return compressed_image

    def display_images(self, compressed_images, singular_values_list):
        """
        Display the original image and the compressed versions.
        - compressed_images: List of compressed images.
        - singular_values_list: List of singular values used for each compression.
        """
        plt.figure(figsize=(12, 6))
        # Original Image
        plt.subplot(1, len(compressed_images) + 1, 1)
        plt.title("Original Image")
        plt.imshow(self.grayscale_image, cmap='gray')
        plt.axis('off')

        # Compressed Images
        for i, (compressed_image, num_sv) in enumerate(zip(compressed_images, singular_values_list), start=2):
            plt.subplot(1, len(compressed_images) + 1, i)
            plt.title(f"SV: {num_sv}")
            plt.imshow(compressed_image, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    # image_path = "example_image.jpg"  # Replace with your image file path
    image_path = "../images/things.png"
    compressor = ImageCompressor(image_path)
    compressor.load_image()

    # Compress image with different numbers of singular values
    sv_list = [5, 20, 50]
    compressed_images = [compressor.compress_image(num_sv) for num_sv in sv_list]

    # Display results
    compressor.display_images(compressed_images, sv_list)


# Project Title: Image Compression Using Singular Value Decomposition (SVD)
"""
File Name: image_compression_with_svd.py

Short Description
This project uses the mathematical technique of Singular Value Decomposition (SVD) for image compression. The idea is to approximate an image matrix using a reduced number of singular values, significantly lowering the storage size while maintaining acceptable visual quality. This project demonstrates advanced NumPy techniques, including matrix decomposition and reconstruction, and provides flexibility in compression levels.
"""

# Example Inputs and Expected Outputs
"""
Example 1
Input:

plaintext
Copy code
Image: 100x100 grayscale image with distinct patterns.
Number of Singular Values: 5, 20, 50.
Expected Output:

Original Image: High-quality grayscale image.
5 Singular Values: Blurry representation with basic patterns visible.
20 Singular Values: Improved clarity, noticeable but acceptable loss of detail.
50 Singular Values: Near-original quality with slight compression artifacts.
Example 2
Input:

plaintext
Copy code
Image: 256x256 grayscale portrait.
Number of Singular Values: 10, 50, 100.
Expected Output:

Original Image: High-resolution grayscale portrait.
10 Singular Values: Very blurry with minimal recognizable features.
50 Singular Values: Recognizable portrait with minor blurring.
100 Singular Values: Excellent quality, closely resembling the original.
Example 3
Input:

plaintext
Copy code
Image: 512x512 grayscale satellite photo.
Number of Singular Values: 25, 100, 200.
Expected Output:

Original Image: Crisp grayscale satellite image.
25 Singular Values: Coarse representation with major features visible.
100 Singular Values: Clear image with moderate compression artifacts.
200 Singular Values: High-quality image, visually similar to the original.
Key Features
SVD-Based Compression:
Demonstrates advanced matrix operations using NumPy.
Dynamic Compression:
Allows for adjustable levels of compression based on singular value count.
Visualization:
Clearly shows the trade-offs between compression and image quality.
Versatility:
Applicable to diverse grayscale images.
This project not only showcases the computational power of NumPy but also bridges the gap to real-world applications like data compression and image processing.
"""