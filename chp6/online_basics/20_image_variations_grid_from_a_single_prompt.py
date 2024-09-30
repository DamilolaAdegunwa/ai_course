"""
Project Title: Image Variations Grid from a Single Prompt
Description:
In this exercise, we will take a single prompt and generate multiple variations of images from that prompt. The goal is to build a grid layout (collage) that displays different variations of the image generated from the same prompt. This will help you learn how to work with variations, multiple API calls, and arranging images into a grid, while handling their retrieval and processing efficiently.

This project advances your previous one by introducing image variation generation and grid arrangement of similar but distinct images based on a single prompt. Instead of handling different prompts for each image, you will work with variations and structure them into a grid.
"""

import os
import requests
from PIL import Image
from io import BytesIO
import certifi
from openai import OpenAI
from apikey import apikey  # Importing the API key from your apikey.py file

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate multiple variations of an image from a single prompt
def generate_image_variations(prompt, num_variations=4):
    """
    Generate multiple variations of an image based on a single prompt using OpenAI's DALL·E model.
    :param prompt: The text prompt to generate images.
    :param num_variations: Number of image variations to generate.
    :return: A list of generated images.
    """
    images = []

    for _ in range(num_variations):
        # Generate image for the prompt
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="512x512",  # Fixed size for each image variation
            response_format="url"
        )

        image_url = response.data[0].url
        print(f"Generated image variation: {image_url}")

        # Fetch the image from the URL
        image_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(image_response.content))
        images.append(img)

    return images


# Function to create a grid collage of the image variations
def create_variations_collage(images, grid_size):
    """
    Create a grid collage of image variations.
    :param images: A list of PIL images.
    :param grid_size: A tuple indicating the grid size (rows, cols).
    :return: The final collage image.
    """
    rows, cols = grid_size
    num_images = len(images)

    if num_images != rows * cols:
        raise ValueError(f"Number of images ({num_images}) doesn't match the grid size ({rows}x{cols}).")

    # Set up the size for each image in the collage
    image_width, image_height = 512, 512

    # Create a blank canvas for the collage
    collage_width = cols * image_width
    collage_height = rows * image_height
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    # Place each image in the grid
    for i, img in enumerate(images):
        # Resize the image to ensure consistency
        img = img.resize((image_width, image_height))

        # Calculate the position in the grid
        row = i // cols
        col = i % cols

        # Calculate the position where the image will be pasted
        x = col * image_width
        y = row * image_height

        # Paste the image into the collage
        collage_image.paste(img, (x, y))

    return collage_image


# Main function to run the variation collage generation
def main():
    # Get the prompt from the user
    prompt = input("Enter the prompt for generating image variations: ")

    # Set grid size (for example: 2x2 grid of variations)
    rows = 2
    cols = 2

    # Generate image variations from the prompt
    images = generate_image_variations(prompt, num_variations=rows * cols)

    # Create the collage from the generated variations
    collage_image = create_variations_collage(images, grid_size=(rows, cols))
    collage_image.show()

    # Save the collage image
    output_name = "image_variations_collage.png"
    collage_image.save(output_name)
    print(f"Collage image saved as {output_name}")


if __name__ == "__main__":
    main()

"""
Key Learning Points:
Image Variations: You will learn how to generate multiple image variations from a single prompt using OpenAI's image generation capabilities.
Collage Construction: Handling multiple variations and arranging them into a grid is a step up in complexity compared to simply generating a grid of unrelated images.
Efficient API Calls: This exercise emphasizes making multiple API calls, retrieving the images, and processing them effectively into a final composition.
Example Use Case:
Use a prompt like "A futuristic robot in a neon city", and the script will generate multiple variations of this concept and arrange them into a collage. Each image will have its own slight differences while staying within the theme.

This could be applied to explore variations of an artistic idea or create a collage that showcases different interpretations of the same concept.

Next Challenge:
After completing this project, you can try adding features such as:

Dynamic Grid Size: Let the user decide how many rows and columns they want in the grid.
Advanced Prompt Modifications: Instead of repeating the same prompt for each variation, introduce slight modifications to the prompt for more diverse results.
Image Filters: Apply post-processing filters (e.g., grayscale, sepia) to the generated images to give each variation a different effect.
This project is noticeably more advanced than the previous one as you handle variations of the same image, emphasizing creativity, repetition, and manipulation. This offers a deeper dive into image handling and manipulation but keeps the complexity manageable.
"""