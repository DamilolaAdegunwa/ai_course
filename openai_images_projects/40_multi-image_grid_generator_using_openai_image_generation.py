"""
Project Title: Multi-Image Grid Generator Using OpenAI Image Generation
In this project, we will use OpenAI's image generation capabilities to create multiple images from different prompts and combine them into a single grid image. The project will generate a series of images based on various prompts and then assemble them into a composite grid. This project advances your skills by incorporating image processing and manipulating multiple image outputs into a unified visual display.

Python Code
"""
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_image_from_prompt(prompt):
    """
    Generate an image based on a description prompt.

    :param prompt: A string containing the image description.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"
    )
    return response.data[0].url  # Returns the URL of the generated image


def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def create_image_grid(prompts, grid_size=(2, 2), image_size=(1024, 1024)):
    """
    Generate a grid of images based on prompts and assemble them into a single image.

    :param prompts: A list of description prompts for generating images.
    :param grid_size: A tuple indicating the grid layout (rows, cols).
    :param image_size: The size of each generated image.
    :return: A composite grid image (PIL Image).
    """
    images = []

    # Generate images for each prompt
    for prompt in prompts:
        print(f"Generating image for prompt: {prompt}")
        image_url = generate_image_from_prompt(prompt)
        image = download_image(image_url)
        image = image.resize(image_size)
        images.append(image)

    # Create an empty grid canvas
    grid_width = grid_size[1] * image_size[0]
    grid_height = grid_size[0] * image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste the images into the grid
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid_image.paste(img, (col * image_size[0], row * image_size[1]))

    return grid_image


def save_grid_image(grid_image, output_path="grid_image.jpg"):
    """
    Save the final grid image to a specified path.

    :param grid_image: The composite grid image.
    :param output_path: The path to save the image.
    :return: None
    """
    grid_image.save(output_path)
    print(f"Saved the grid image to {output_path}")


# Example use cases
if __name__ == "__main__":
    # Prompts for generating images
    prompts = [
        "A futuristic cityscape",
        "A dragon flying over a medieval castle",
        "A peaceful forest at sunset",
        "A cyberpunk neon-lit street"
    ]

    # Create a 2x2 grid image
    grid_image = create_image_grid(prompts, grid_size=(2, 2), image_size=(512, 512))

    # Save the resulting image
    save_grid_image(grid_image, "futuristic_city_grid.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompts: ["A futuristic cityscape", "A dragon flying over a medieval castle", "A peaceful forest at sunset", "A cyberpunk neon-lit street"]
Grid Size: (2, 2)
Expected Output:
A 2x2 image grid combining:
A futuristic cityscape
A dragon flying over a medieval castle
A forest at sunset
A neon-lit cyberpunk street
Input:

Prompts: ["A spaceship flying through an asteroid belt", "An underwater city", "A volcano erupting", "A serene mountain range"]
Grid Size: (2, 2)
Expected Output:
A 2x2 grid with:
A spaceship flying through an asteroid belt
An underwater city
A volcano erupting
A serene mountain range
Input:

Prompts: ["A magical forest", "A futuristic robot", "An ancient pyramid", "A city on the moon", "A deep sea creature"]
Grid Size: (3, 2)
Expected Output:
A 3x2 grid of:
A magical forest
A futuristic robot
An ancient pyramid
A city on the moon
A deep sea creature
Input:

Prompts: ["A knight battling a dragon", "A futuristic cityscape", "A cozy winter cabin", "A bright beach during summer"]
Grid Size: (2, 2)
Expected Output:
A 2x2 grid combining:
A knight battling a dragon
A futuristic cityscape
A cozy winter cabin
A bright summer beach
Input:

Prompts: ["A waterfall in a tropical forest", "A car driving in the desert", "An astronaut on Mars", "A vibrant coral reef"]
Grid Size: (2, 2)
Expected Output:
A 2x2 image grid showing:
A waterfall in a tropical forest
A car driving in the desert
An astronaut exploring Mars
A vibrant coral reef underwater
Project Overview
In this advanced project, you will generate multiple images based on different prompts and organize them into a single grid-like composite image. You'll practice working with multiple outputs from the OpenAI image API and manipulating these images using Python’s PIL library to form a visually appealing grid. This project will build your expertise in handling API requests, manipulating image outputs, and integrating them into a single cohesive design.
"""