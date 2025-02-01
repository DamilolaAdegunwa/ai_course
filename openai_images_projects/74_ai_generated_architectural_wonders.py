"""
Project Title: AI-Generated Architectural Wonders
File Name: ai_generated_architectural_wonders.py
Description:
In this project, we explore the creation of architectural masterpieces that blend futuristic design with natural elements. You'll generate images of imaginative and surreal architectural structures, such as floating cities, glass pyramids embedded in waterfalls, or skyscrapers intertwined with forests. This exercise will improve your ability to compose complex architectural and environmental prompts to generate stunning visual representations that seem plausible yet fantastical.

This exercise introduces the challenge of combining hard, geometric shapes with soft, natural forms, while keeping the focus on imaginative architecture.

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


# Function to generate images of architectural wonders
def generate_architecture_image(prompt):
    """
    Generate an image based on a futuristic or surreal architectural design.

    :param prompt: The prompt describing the architecture and environment.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Higher resolution to capture architectural details
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
    # Define multiple prompts involving futuristic architectural designs
    architecture_prompts = [
        "A floating city above a waterfall surrounded by dense tropical forest, with buildings made of glass and steel",
        "A modern skyscraper intertwined with a giant ancient oak tree, with transparent walkways connecting the branches",
        "A massive underground temple built from glowing crystals, with water flowing through intricate channels on the floor",
        "A futuristic dome city on Mars, surrounded by red desert landscapes and shielded by a translucent energy field",
        "A glass pyramid standing in the middle of a waterfall, with sunlight reflecting off its surface and creating rainbows"
    ]

    # Generate, download, and save each architectural wonder image
    for i, prompt in enumerate(architecture_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_architecture_image(prompt)
        image = download_image(image_url)
        save_image(image, f"architectural_wonder_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A floating city above a waterfall surrounded by dense tropical forest, with buildings made of glass and steel"
Expected Output:
An image of a futuristic city floating high above a giant waterfall, with sleek, glass and steel buildings. The surrounding environment is lush with tropical trees, and mist from the waterfall creates an ethereal effect.
Input:

Prompt: "A modern skyscraper intertwined with a giant ancient oak tree, with transparent walkways connecting the branches"
Expected Output:
A stunning fusion of nature and architecture, with a towering skyscraper wrapped around the massive branches of an ancient oak tree. Transparent walkways connect various parts of the tree to the building, blending modern and organic elements.
Input:

Prompt: "A massive underground temple built from glowing crystals, with water flowing through intricate channels on the floor"
Expected Output:
A visually striking underground temple made entirely from luminescent crystals. The floor is intricately designed with channels carrying streams of water, creating a serene and mystical ambiance.
Input:

Prompt: "A futuristic dome city on Mars, surrounded by red desert landscapes and shielded by a translucent energy field"
Expected Output:
A high-tech dome city on the surface of Mars, encapsulated by a shimmering energy shield. The barren red desert surrounds the dome, contrasting sharply with the sleek futuristic design inside.
Input:

Prompt: "A glass pyramid standing in the middle of a waterfall, with sunlight reflecting off its surface and creating rainbows"
Expected Output:
A surreal image of a large glass pyramid at the center of a powerful waterfall. Sunlight hits the pyramid, causing vivid rainbow reflections across the misty surroundings.
Project Overview:
This project pushes your creative limits by encouraging you to blend architectural elements with natural environments in unique and futuristic ways. By generating complex prompts that feature imaginative, surreal structures, you’ll further develop your ability to express architectural vision through AI image generation. The exercise enhances your prompt engineering skills for creating not only fantastical environments but also detailed and realistic architectural wonders.







"""