"""
Project Title: Generating Thematic Digital Posters with Dynamic Composition
In this project, you will generate digital posters that include multiple thematic elements (such as characters, environments, and visual motifs) using OpenAI’s image generation API. The focus is on creating dynamic, visually striking compositions suitable for digital posters. The exercise will help you improve at combining different artistic styles and elements into a cohesive, advanced visual output.

You will also be able to experiment with variations in themes, such as a sci-fi setting, historical artwork, or abstract design posters. Each poster will be generated based on a complex, detailed prompt.

Python File Name: thematic_digital_poster_generation.py
Python Code:
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

# Function to generate a thematic poster based on a complex prompt
def generate_thematic_poster(prompt):
    """
    Generate a thematic digital poster using OpenAI's image generation API.

    :param prompt: The prompt describing the theme and style for the digital poster.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Using 1024x1024 for high-resolution poster-like images
    )

    return response.data[0].url  # Returns the URL of the generated poster

# Function to download the image from a URL and return it as a PIL Image object
def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to save the image to a file
def save_image(image, filename):
    """
    Save the downloaded image to a file.

    :param image: PIL Image object.
    :param filename: The path and name of the file to save the image to.
    """
    image.save(filename)
    print(f"Image saved as {filename}")

# Example use cases
if __name__ == "__main__":
    # Define multiple thematic prompts to test
    poster_prompts = [
        "A cyberpunk city skyline at sunset with neon lights and flying cars",
        "A medieval castle on a hill with knights in armor and dragons flying in the background",
        "A futuristic robot in a dystopian landscape, with glowing red eyes and shattered buildings",
        "A retro-style 80s movie poster with vibrant colors, palm trees, and a sports car",
        "An abstract digital art poster with colorful geometric shapes and a minimalist design"
    ]

    # Generate, download, and save each poster image
    for i, prompt in enumerate(poster_prompts):
        print(f"Generating poster for prompt: '{prompt}'...")
        poster_url = generate_thematic_poster(prompt)
        poster_image = download_image(poster_url)
        save_image(poster_image, f"thematic_poster_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A cyberpunk city skyline at sunset with neon lights and flying cars"
Expected Output:
A vibrant, futuristic cityscape with towering skyscrapers bathed in neon lights. Flying cars zoom through the sky, and the setting sun casts a dramatic orange glow over the skyline.
Input:

Prompt: "A medieval castle on a hill with knights in armor and dragons flying in the background"
Expected Output:
An epic fantasy scene featuring a majestic medieval castle atop a hill. Knights in shining armor stand guard, while dragons soar through the sky in the background, adding an air of mystery and adventure.
Input:

Prompt: "A futuristic robot in a dystopian landscape, with glowing red eyes and shattered buildings"
Expected Output:
A dark and moody scene with a menacing robot in a post-apocalyptic world. The robot has glowing red eyes, and the environment is filled with crumbling buildings, signifying destruction and chaos.
Input:

Prompt: "A retro-style 80s movie poster with vibrant colors, palm trees, and a sports car"
Expected Output:
A nostalgic throwback to the 1980s, featuring bold, vibrant colors and a classic sports car speeding down a road lined with palm trees. The poster has a vintage movie vibe with bold fonts and dramatic lighting.
Input:

Prompt: "An abstract digital art poster with colorful geometric shapes and a minimalist design"
Expected Output:
A visually striking abstract design with various colorful geometric shapes arranged in a clean, minimalist layout. The poster has a modern feel with smooth lines and minimal text.
Project Overview:
This exercise is designed to help you work with more intricate prompts that involve specific styles, themes, and artistic compositions. By creating visually dynamic posters, you will explore how the AI interprets various artistic directions (e.g., cyberpunk, medieval, retro, abstract). The examples above are meant to showcase a variety of themes, helping you push the boundaries of what OpenAI’s image generation can accomplish in terms of complex thematic visualization.
"""