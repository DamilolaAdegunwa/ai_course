"""
Project Title: Futuristic Cityscapes with Sci-Fi Elements
File Name: futuristic_cityscapes_with_sci_fi_elements.py
Description:
This project focuses on generating highly detailed and visually complex images of futuristic cityscapes. The aim is to create diverse urban environments, enhanced with sci-fi elements such as floating vehicles, towering neon-lit skyscrapers, holograms, and intergalactic objects. By combining several visual features, the generated images will display advanced architecture and vibrant colors reminiscent of sci-fi films or games.

The goal is to push your ability to handle intricate multi-layered prompts that result in visually rich imagery. Additionally, you'll be able to refine your control over detail, composition, and atmosphere, which are critical for generating compelling artwork.

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


# Function to generate images of futuristic cityscapes with sci-fi elements
def generate_cityscape(prompt):
    """
    Generate a futuristic cityscape image based on a prompt describing sci-fi elements.

    :param prompt: The prompt describing the futuristic cityscape and sci-fi features.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed and complex cityscape
    )

    return response.data[0].url  # Returns the URL of the generated image


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
    # Define multiple prompts for futuristic cityscapes
    cityscape_prompts = [
        "A futuristic metropolis at night with glowing neon skyscrapers, flying cars, and floating billboards under a dark purple sky",
        "A utopian city with towering glass structures, green spaces with robotic gardeners, and holographic ads lighting the streets",
        "A massive cyberpunk city with interconnected buildings, giant robot statues, and an alien ship hovering in the sky",
        "A sprawling megacity with a mix of ancient temples and ultra-modern skyscrapers, filled with drones and floating trains",
        "A dystopian city where towering industrial structures dominate, red neon signs glow, and storm clouds gather in the sky"
    ]

    # Generate, download, and save each cityscape image
    for i, prompt in enumerate(cityscape_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_cityscape(prompt)
        image = download_image(image_url)
        save_image(image, f"futuristic_cityscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A futuristic metropolis at night with glowing neon skyscrapers, flying cars, and floating billboards under a dark purple sky"
Expected Output:
A vibrant cityscape at night, featuring towering neon-lit skyscrapers, cars flying between the buildings, and glowing floating billboards. The sky has a dark purple hue, giving a sci-fi movie feel to the scene.
Input:

Prompt: "A utopian city with towering glass structures, green spaces with robotic gardeners, and holographic ads lighting the streets"
Expected Output:
A clean and futuristic utopian city, with glass buildings towering into the sky. Robotic gardeners tend to lush green spaces, while holographic ads and street signs illuminate the surroundings.
Input:

Prompt: "A massive cyberpunk city with interconnected buildings, giant robot statues, and an alien ship hovering in the sky"
Expected Output:
A dark cyberpunk city, with buildings connected by sky bridges. Giant robot statues overlook the streets, while an ominous alien ship hovers above the city, casting shadows over the streets below.
Input:

Prompt: "A sprawling megacity with a mix of ancient temples and ultra-modern skyscrapers, filled with drones and floating trains"
Expected Output:
A unique city that blends ancient temples with sleek modern skyscrapers. Drones zip through the air, and floating trains hover along tracks suspended between buildings, creating a contrast of old and new.
Input:

Prompt: "A dystopian city where towering industrial structures dominate, red neon signs glow, and storm clouds gather in the sky"
Expected Output:
A bleak dystopian city dominated by dark industrial structures. Red neon signs provide the only light in the darkened streets, while storm clouds gather ominously in the sky, creating a sense of foreboding.
Project Overview:
This project enhances your skills in prompt engineering for more elaborate and visually detailed imagery. You’ll explore a range of sci-fi and futuristic themes, learning to create complex compositions, atmospheres, and contrasting urban settings. The combination of visual elements will allow you to refine how you describe detailed environments and scenes in a cohesive and imaginative way.







"""