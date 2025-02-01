"""
Project Title: Architectural Wonders in Futuristic Cities
File Name: architectural_wonders_in_futuristic_cities.py
Description:
This project focuses on generating images of stunning architectural wonders set in futuristic cityscapes. The aim is to create visuals that depict towering, innovative buildings, advanced transportation systems, and utopian or dystopian settings. You will explore the complexity of architectural design by incorporating elements like skyscrapers made of glass, gravity-defying structures, or futuristic floating cities.

This project introduces you to generating multi-faceted and complex imagery based on urban design concepts, blending creativity with precision in describing the city’s futuristic aesthetic.

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


# Function to generate futuristic cityscapes and architectural wonders
def generate_architectural_image(prompt):
    """
    Generate an image based on futuristic architecture and cityscape using OpenAI's image generation API.

    :param prompt: The prompt describing the architectural wonder and futuristic city.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed urban imagery
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
    # Define multiple prompts involving futuristic architecture in modern cityscapes
    city_prompts = [
        "A futuristic city with towering glass skyscrapers, flying cars, and vertical gardens",
        "A city of floating buildings connected by transparent bridges in a neon-lit skyline",
        "An underground city illuminated by bioluminescent plants, with buildings carved from crystal",
        "A utopian city with sleek, gravity-defying structures surrounded by waterfalls and lush green parks",
        "A dystopian city with decaying skyscrapers, hovering drones, and dark, stormy skies"
    ]

    # Generate, download, and save each architectural wonder image
    for i, prompt in enumerate(city_prompts):
        print(f"Generating architectural image for prompt: '{prompt}'...")
        city_url = generate_architectural_image(prompt)
        city_image = download_image(city_url)
        save_image(city_image, f"futuristic_city_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A futuristic city with towering glass skyscrapers, flying cars, and vertical gardens"
Expected Output:
An image of a vibrant futuristic city with tall glass skyscrapers reflecting sunlight, cars flying between buildings, and lush vertical gardens climbing up the sides of the towers.
Input:

Prompt: "A city of floating buildings connected by transparent bridges in a neon-lit skyline"
Expected Output:
A surreal city with buildings floating in the air, interconnected by transparent bridges. The skyline is lit up with neon lights in shades of purple, pink, and blue.
Input:

Prompt: "An underground city illuminated by bioluminescent plants, with buildings carved from crystal"
Expected Output:
An underground city with structures made of glowing crystals, illuminated by bioluminescent plants growing throughout the streets. The atmosphere is serene yet otherworldly.
Input:

Prompt: "A utopian city with sleek, gravity-defying structures surrounded by waterfalls and lush green parks"
Expected Output:
A pristine utopian city with futuristic, gravity-defying buildings that curve and float. The city is surrounded by natural waterfalls and parks full of vibrant green vegetation.
Input:

Prompt: "A dystopian city with decaying skyscrapers, hovering drones, and dark, stormy skies"
Expected Output:
A grim, dystopian city with decaying buildings, menacing drones flying overhead, and an ominous sky filled with dark clouds and lightning. The mood is tense and apocalyptic.
Project Overview:
This project is a step up in complexity as it emphasizes intricate cityscapes and futuristic architecture. It challenges you to think about spatial elements, design, and how to incorporate urban aesthetics into your prompt engineering. Through multiple use cases, you’ll gain experience generating detailed, awe-inspiring images of both utopian and dystopian cities.







"""