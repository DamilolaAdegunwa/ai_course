"""
Project Title: Dynamic Image Collage Generator using DALL·E
Description:
In this project, you will create a dynamic image collage generator that takes in multiple text prompts, generates images based on those prompts using OpenAI’s image generation API (DALL·E 3), and then combines these generated images into a single collage. This project helps in learning how to handle multiple image generation requests and how to process and merge images programmatically using Python and OpenAI's API.

Key Features:
Generate multiple images based on user-provided prompts.
Combine the generated images into a collage.
Dynamically handle different numbers of prompts (e.g., a 2x2 or 3x3 grid depending on how many prompts are given).
Allow the user to specify the output collage size (e.g., "1024x1024", "512x512").

Key Learning Points:
Dynamic Image Generation: You will learn how to handle multiple API requests and store images dynamically based on user input.
Image Manipulation: You’ll use the Pillow (PIL) library to handle image resizing and combining images into a collage.
Collage Layout Logic: The project introduces logic to automatically determine how to layout images in a grid based on the number of prompts.
Custom Collage Sizes: You’ll learn how to allow flexibility with the collage size while keeping the images within proportion.
Explanation:
Image Generation: We use client.images.generate() for each user-specified prompt, fetching a unique image URL for each and then downloading the image.
Collage Creation: We calculate how to arrange the images in a square grid based on the number of images (i.e., 2x2 for 4 images, 3x3 for 9 images, etc.). The images are resized to fit perfectly within the overall collage.
Dynamic Layout: The number of prompts defines the grid layout and how the images are arranged on the collage canvas. This method ensures that whether the user enters 4, 9, or 16 prompts, the collage is appropriately arranged.
Additional Challenges:
Custom Grid Layouts: You could extend the project by allowing the user to define a custom number of rows and columns instead of an automatic grid.
Interactive Prompt Input: Modify the code to allow users to input prompts dynamically via the command line or a simple GUI.
Higher-Resolution Images: Modify the project to handle larger images like "1024x1024" and optimize the collage generation process to handle higher-resolution output.
This project will help you gain more control over generating and manipulating images and lead to more complex projects, such as integrating this with user interfaces or creating fully automated art galleries!
"""

import os
from openai import OpenAI
from apikey import apikey
from PIL import Image, ImageOps
import requests
from io import BytesIO
import certifi
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_image_from_prompt(prompt):
    """
    Generate an image based on a text prompt using OpenAI's DALL·E 3 model.
    """
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="512x512",  # Change this to 1024x1024 if you want higher resolution
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Image generated for prompt: '{prompt}' - URL: {image_url}")
    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


def create_image_collage(prompts, collage_size=(1024, 1024)):
    """
    Generate a collage of images generated from a list of prompts.
    :param prompts: List of text prompts to generate images for.
    :param collage_size: Size of the final collage (width, height).
    """
    images = []
    num_images = len(prompts)

    # Generate images for each prompt
    for prompt in prompts:
        image = generate_image_from_prompt(prompt)
        images.append(image)

    # Create a blank image for the collage
    collage_width, collage_height = collage_size
    num_columns = num_rows = int(num_images ** 0.5)  # Approximate a square grid

    # Resize images to fit within the collage
    single_width = collage_width // num_columns
    single_height = collage_height // num_rows
    resized_images = [ImageOps.fit(image, (single_width, single_height), method=Image.Resampling.LANCZOS) for image in images]

    # Create a blank canvas for the collage
    collage_image = Image.new('RGB', collage_size)

    # Paste the images onto the collage
    for index, image in enumerate(resized_images):
        x = (index % num_columns) * single_width
        y = (index // num_columns) * single_height
        collage_image.paste(image, (x, y))

    return collage_image


def main():
    # User-defined prompts for generating images
    prompts = [
        "A sunset over the mountains",
        "A futuristic city with neon lights",
        "A serene beach with clear blue water",
        "A medieval castle under the moonlight"
    ]

    # Generate the collage
    collage = create_image_collage(prompts, collage_size=(1024, 1024))

    # Save and show the final collage
    collage.save("collage.png")
    collage.show()


if __name__ == "__main__":
    main()
