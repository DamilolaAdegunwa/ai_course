"""
Project Title: Mythical Creature Generations in Unique Environments
File Name: mythical_creature_generations_in_unique_environments.py
Description:
In this advanced project, you will generate images of mythical creatures, each placed in distinct environments. The project will focus on creating detailed and imaginative creatures such as dragons, phoenixes, or unicorns, combined with different settings like ancient forests, volcanic landscapes, or snowy mountains. The purpose of this exercise is to improve the ability to compose highly creative prompts that blend creature details and environmental settings, pushing the complexity and diversity of images you can generate.

This project will give you experience in working with multi-layered prompts that need to combine creature descriptions, environmental elements, and artistic styles to produce stunning results.

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


# Function to generate images of mythical creatures in unique environments
def generate_mythical_creature(prompt):
    """
    Generate an image based on a prompt describing a mythical creature in a unique environment.

    :param prompt: The prompt describing the creature and its environment.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for capturing details of both creature and environment
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
    # Define multiple prompts for mythical creatures in unique environments
    creature_prompts = [
        "A majestic phoenix soaring through a stormy sky, surrounded by lightning and dark clouds",
        "A giant dragon curled around a volcanic mountain, with lava flowing and fire lighting the sky",
        "A unicorn standing in a moonlit enchanted forest, with glowing mushrooms and sparkling streams",
        "A sea serpent emerging from the ocean during a sunset, with waves crashing and glowing coral reefs beneath",
        "A griffin perched atop a snowy mountain peak, with auroras lighting the sky and distant icy landscapes"
    ]

    # Generate, download, and save each mythical creature image
    for i, prompt in enumerate(creature_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_mythical_creature(prompt)
        image = download_image(image_url)
        save_image(image, f"mythical_creature_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A majestic phoenix soaring through a stormy sky, surrounded by lightning and dark clouds"
Expected Output:
A stunning image of a glowing phoenix with wings outstretched, soaring through a dark sky filled with bolts of lightning. Dark clouds swirl in the background, adding a dramatic atmosphere to the scene.
Input:

Prompt: "A giant dragon curled around a volcanic mountain, with lava flowing and fire lighting the sky"
Expected Output:
A fearsome dragon with scales glistening in the fiery light. It curls around a volcanic peak with rivers of lava flowing down the mountainside, while the sky is ablaze with flames and smoke.
Input:

Prompt: "A unicorn standing in a moonlit enchanted forest, with glowing mushrooms and sparkling streams"
Expected Output:
A magical unicorn standing elegantly in a serene forest. The scene is illuminated by moonlight, with glowing mushrooms scattered around, and a sparkling stream flowing gently nearby, evoking a peaceful fantasy landscape.
Input:

Prompt: "A sea serpent emerging from the ocean during a sunset, with waves crashing and glowing coral reefs beneath"
Expected Output:
A massive sea serpent rising from the ocean, its scales reflecting the golden light of the setting sun. Waves crash around it, and beneath the water, vibrant coral reefs glow faintly, adding color and texture to the underwater scenery.
Input:

Prompt: "A griffin perched atop a snowy mountain peak, with auroras lighting the sky and distant icy landscapes"
Expected Output:
A regal griffin stands proudly on a snowy peak, its wings spread slightly as if ready to take flight. The sky above is illuminated with colorful auroras, casting a soft glow over the distant icy landscapes below.
Project Overview:
This project takes your OpenAI image generation skills to the next level by focusing on combining fantastical elements with detailed natural environments. By working with mythical creatures and complex settings, you will learn how to craft more intricate prompts and control the balance between character design and atmospheric details. The project allows for creativity while challenging you to create visually rich and imaginative artwork.
"""