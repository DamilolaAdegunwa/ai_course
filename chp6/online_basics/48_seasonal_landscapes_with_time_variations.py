"""
Project Title: AI-Generated Landscapes with Seasonal and Time-of-Day Variations
In this advanced exercise, you will generate highly detailed and atmospheric landscapes with dynamic adjustments based on seasons and time of day. The goal of the project is to produce stunning landscapes that vary depending on the specified season (e.g., winter, spring) and time (e.g., sunrise, sunset, night). This exercise will help you explore how changes in these environmental factors impact the artistic rendering of nature, cities, and fantasy settings.

Python File Name: seasonal_landscapes_with_time_variations.py
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


# Function to generate landscape images based on season and time of day
def generate_landscape_image(prompt):
    """
    Generate a detailed landscape image using OpenAI's image generation API.

    :param prompt: The prompt describing the landscape, season, and time of day.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High-resolution for detailed landscapes
    )

    return response.data[0].url  # Returns the URL of the generated landscape


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
    # Define multiple seasonal landscape prompts to test
    landscape_prompts = [
        "A winter landscape at sunrise with snow-covered mountains and pine trees",
        "A spring meadow at sunset with colorful wildflowers and a stream flowing through",
        "A tropical beach at dusk with palm trees swaying in the wind and the ocean reflecting the last light of the sun",
        "An autumn forest in the morning mist, with golden leaves falling and a peaceful atmosphere",
        "A fantasy landscape at night, with glowing mushrooms and a star-filled sky over distant mountains"
    ]

    # Generate, download, and save each landscape image
    for i, prompt in enumerate(landscape_prompts):
        print(f"Generating landscape for prompt: '{prompt}'...")
        landscape_url = generate_landscape_image(prompt)
        landscape_image = download_image(landscape_url)
        save_image(landscape_image, f"landscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A winter landscape at sunrise with snow-covered mountains and pine trees"
Expected Output:
A crisp, cold scene with snow-capped mountains in the distance. The sun is just rising, casting a golden glow over the white landscape, and tall pine trees stand majestically in the foreground.
Input:

Prompt: "A spring meadow at sunset with colorful wildflowers and a stream flowing through"
Expected Output:
A vibrant spring scene with a meadow filled with wildflowers in various colors. The setting sun bathes the scene in warm hues, and a small stream meanders through the field, reflecting the colors of the sky.
Input:

Prompt: "A tropical beach at dusk with palm trees swaying in the wind and the ocean reflecting the last light of the sun"
Expected Output:
A serene tropical beach with tall palm trees bending slightly in the evening breeze. The sky is tinged with the last rays of sunlight, which shimmer across the calm ocean waters.
Input:

Prompt: "An autumn forest in the morning mist, with golden leaves falling and a peaceful atmosphere"
Expected Output:
A calm autumnal forest scene with trees covered in golden and orange leaves. There’s a light mist in the air, and the scene feels tranquil, with leaves slowly drifting to the ground in the stillness of the morning.
Input:

Prompt: "A fantasy landscape at night, with glowing mushrooms and a star-filled sky over distant mountains"
Expected Output:
A mystical, otherworldly landscape with bioluminescent mushrooms scattered across the ground. The dark night sky is full of stars, and faintly visible in the distance are tall, jagged mountains, giving a sense of wonder and adventure.
Project Overview:
This project allows you to generate detailed, atmospheric landscape images by incorporating seasonal changes and variations in the time of day. These factors greatly affect the resulting visual compositions, giving you the ability to experiment with natural elements and fantasy-inspired settings. By specifying seasons like winter or spring and times of day such as sunrise or night, you'll create highly varied landscape artwork, which can be useful for generating background art, posters, or creative inspiration.
"""