"""
Project Title: AI-Generated Cities of the Future
In this advanced exercise, you will generate futuristic cityscapes with highly detailed architectural elements based on specified themes like technology, nature integration, or sci-fi dystopia. This exercise will explore how futuristic design concepts, such as urban environments blending with nature, can be visualized using AI. The project will help you learn to prompt for complex environments, handling specific stylistic elements to generate a variety of cityscape scenes.

Python File Name: futuristic_cities_generation.py
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


# Function to generate futuristic cityscape images based on themes
def generate_futuristic_city_image(prompt):
    """
    Generate a futuristic city image using OpenAI's image generation API.

    :param prompt: The prompt describing the futuristic city and theme.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed cityscapes
    )

    return response.data[0].url  # Returns the URL of the generated cityscape


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
    # Define multiple futuristic cityscape prompts to test
    city_prompts = [
        "A futuristic city with floating buildings and flying cars, glowing in neon colors at night",
        "A high-tech city where nature has integrated with the architecture, with trees growing on skyscrapers and green parks suspended in the sky",
        "A sci-fi dystopian city with towering structures, dark skies, and mechanical drones flying above",
        "A futuristic city underwater, with transparent domes housing people and large sea creatures swimming outside",
        "A futuristic eco-friendly city in a desert, where solar panels cover buildings and vertical farms are built into the architecture"
    ]

    # Generate, download, and save each futuristic cityscape image
    for i, prompt in enumerate(city_prompts):
        print(f"Generating futuristic city for prompt: '{prompt}'...")
        city_url = generate_futuristic_city_image(prompt)
        city_image = download_image(city_url)
        save_image(city_image, f"futuristic_city_{i + 1}.jpg")

"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A futuristic city with floating buildings and flying cars, glowing in neon colors at night"
Expected Output:
A vibrant, cyberpunk-inspired city with buildings floating in the air. The night sky is filled with neon hues of pink, blue, and purple, with flying cars speeding through the air.
Input:

Prompt: "A high-tech city where nature has integrated with the architecture, with trees growing on skyscrapers and green parks suspended in the sky"
Expected Output:
A futuristic city with skyscrapers covered in greenery, trees and plants growing on balconies and rooftops. Green parks hover in mid-air, connected by suspension bridges, showing a blend of nature and modern technology.
Input:

Prompt: "A sci-fi dystopian city with towering structures, dark skies, and mechanical drones flying above"
Expected Output:
A dark and ominous cityscape with towering, angular skyscrapers. The sky is clouded and gray, with mechanical drones patrolling the air, hinting at a dystopian future with advanced technology.
Input:

Prompt: "A futuristic city underwater, with transparent domes housing people and large sea creatures swimming outside"
Expected Output:
An underwater city encased in glass domes. The ocean outside teems with marine life, while within the domes, buildings stretch upwards, showing people living in a futuristic aquatic environment.
Input:

Prompt: "A futuristic eco-friendly city in a desert, where solar panels cover buildings and vertical farms are built into the architecture"
Expected Output:
A desert-based city with towering buildings covered in solar panels. Vertical farms provide greenery amidst the arid environment, showing a highly sustainable, eco-friendly futuristic design.
Project Overview:
This project focuses on generating imaginative, detailed futuristic cities based on different themes and environmental conditions. From floating cities and neon-lit cyberpunk worlds to eco-friendly designs integrating nature into urban environments, the exercise will push your creativity in crafting diverse environments. This will allow you to create dynamic, high-quality futuristic cityscapes, which can serve as concept art, inspiration, or assets for creative projects.

"""