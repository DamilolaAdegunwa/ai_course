"""
Project Title: Futuristic Cityscapes with Advanced Transportation Systems
File Name: futuristic_cityscapes_with_advanced_transportation.py
Description:
This project focuses on generating futuristic cityscapes that feature advanced transportation systems, such as flying cars, hyperloop trains, and futuristic architecture. The goal is to visualize bustling, high-tech urban environments filled with modern innovations and design elements. The images generated will be detailed city scenes, showcasing the complexity of infrastructure, transportation hubs, skyscrapers, and overall technological advancements in the urban landscape.

You will craft prompts that blend futuristic architecture with modern transit technologies, pushing the AI to generate intricate and vibrant cityscapes. This exercise will challenge you to create dynamic and multi-layered compositions that highlight the interplay between technology and urban design.

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


# Function to generate an image of a futuristic cityscape with advanced transportation
def generate_cityscape_image(prompt):
    """
    Generate an image of a futuristic cityscape with advanced transportation systems using OpenAI's image generation API.

    :param prompt: The prompt describing the cityscape and transportation systems.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed cityscapes
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
    # Define multiple prompts involving futuristic cityscapes and transportation systems
    cityscape_prompts = [
        "A futuristic city with towering skyscrapers, flying cars zooming through the air, and neon lights illuminating the streets",
        "A hyperloop train station in the middle of a modern city with people boarding sleek, high-speed pods",
        "A sprawling city at sunset with massive glass buildings, autonomous buses, and drones delivering packages",
        "A futuristic metropolis featuring levitating trains passing through towering arcology structures with green terraces",
        "An aerial view of a futuristic city with circular highways, hovercrafts, and vertical gardens growing on skyscrapers"
    ]

    # Generate, download, and save each cityscape image
    for i, prompt in enumerate(cityscape_prompts):
        print(f"Generating futuristic cityscape image for prompt: '{prompt}'...")
        cityscape_url = generate_cityscape_image(prompt)
        cityscape_image = download_image(cityscape_url)
        save_image(cityscape_image, f"futuristic_cityscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A futuristic city with towering skyscrapers, flying cars zooming through the air, and neon lights illuminating the streets"
Expected Output:
A vibrant cityscape filled with towering glass skyscrapers. Flying cars soar between the buildings, and the streets below are lit by bright neon lights, adding a cyberpunk aesthetic.
Input:

Prompt: "A hyperloop train station in the middle of a modern city with people boarding sleek, high-speed pods"
Expected Output:
A hyperloop station surrounded by futuristic buildings. Passengers board streamlined, tube-like pods designed for high-speed travel. The station has a sleek, modern design with minimalistic architecture.
Input:

Prompt: "A sprawling city at sunset with massive glass buildings, autonomous buses, and drones delivering packages"
Expected Output:
A sprawling cityscape during sunset, featuring large glass buildings that reflect the golden light. Autonomous buses travel along the streets while drones fly between buildings, delivering packages.
Input:

Prompt: "A futuristic metropolis featuring levitating trains passing through towering arcology structures with green terraces"
Expected Output:
A futuristic city where levitating trains travel through massive arcology structures—self-contained habitats with green terraces and sustainable designs. The trains glide smoothly through the air without tracks.
Input:

Prompt: "An aerial view of a futuristic city with circular highways, hovercrafts, and vertical gardens growing on skyscrapers"
Expected Output:
A bird’s-eye view of a futuristic city. Circular highways are filled with hovercrafts, and many skyscrapers are adorned with vertical gardens, blending nature with urban living in a high-tech environment.
Project Overview:
This project explores how to generate detailed and imaginative images of futuristic cityscapes, focusing on advanced transportation systems such as flying cars, hyperloop stations, and levitating trains. The visuals combine modern technological advancements with innovative urban design, creating dynamic and visually stunning representations of future cities. By experimenting with a range of prompts, you will create diverse and richly detailed cityscapes that showcase different aspects of futuristic urban life.
"""