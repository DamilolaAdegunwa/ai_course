"""
Project Title: Mythical Creature Fusion Generator
File Name: mythical_creature_fusion_generator.py
Description:
In this project, you will create AI-generated images that blend features of two or more mythical creatures. You’ll develop complex prompts that mix the characteristics of creatures such as dragons, phoenixes, griffins, and mermaids to create entirely new and original hybrids. The project will allow you to explore multi-concept prompts, where you combine different visual ideas, textures, and patterns into a cohesive creation.

This exercise will help you practice working with more complex combinations of ideas, enhancing your ability to generate rich, detailed images from intricate prompts.

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


# Function to generate images of fused mythical creatures
def generate_mythical_creature(prompt):
    """
    Generate an image based on a prompt combining features of multiple mythical creatures.

    :param prompt: The prompt describing the fused mythical creatures.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed creature design
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
    # Define multiple prompts for hybrid mythical creatures
    creature_prompts = [
        "A dragon with the wings of a phoenix and the tail of a griffin, breathing fire in a stormy sky",
        "A mermaid with the body of a unicorn and the horns of a minotaur, swimming in an ethereal ocean",
        "A griffin with the scales of a dragon, standing majestically on a mountain peak with lava flowing in the background",
        "A centaur with the wings of a Pegasus and the claws of a basilisk, racing across a desert landscape under a purple sky",
        "A kraken with the body of a hydra, wrapping its tentacles around ships in a stormy sea with lightning illuminating the scene"
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

Prompt: "A dragon with the wings of a phoenix and the tail of a griffin, breathing fire in a stormy sky"
Expected Output:
A fierce dragon with fiery phoenix wings, a griffin-like tail, and a background of dark clouds and lightning. The dragon is shown breathing fire into the stormy sky, creating a dramatic and intense visual.
Input:

Prompt: "A mermaid with the body of a unicorn and the horns of a minotaur, swimming in an ethereal ocean"
Expected Output:
A surreal hybrid of a mermaid and a unicorn, with the horned head of a minotaur. The creature swims gracefully in a glowing, ethereal ocean, its mane flowing in the water and blending with magical lights.
Input:

Prompt: "A griffin with the scales of a dragon, standing majestically on a mountain peak with lava flowing in the background"
Expected Output:
A majestic griffin with the scaled body of a dragon stands proudly atop a craggy mountain. Lava flows ominously in the background, and the sky is filled with ash and clouds, creating a volcanic scene.
Input:

Prompt: "A centaur with the wings of a Pegasus and the claws of a basilisk, racing across a desert landscape under a purple sky"
Expected Output:
A powerful centaur with large Pegasus wings and terrifying basilisk claws runs across a vast desert. The purple sky above creates an otherworldly atmosphere as the creature moves swiftly through the barren landscape.
Input:

Prompt: "A kraken with the body of a hydra, wrapping its tentacles around ships in a stormy sea with lightning illuminating the scene"
Expected Output:
A terrifying kraken with the multiple heads of a hydra, emerging from a stormy sea. Its tentacles are wrapped around several ships, and flashes of lightning light up the chaotic, high-seas scene.
Project Overview:
This project enhances your skills in handling multi-faceted and complex prompts by blending various elements of mythical creatures. You will learn to work with descriptions that involve intricate details, textures, and unique combinations of different concepts, elevating your prompt-engineering skills for generating elaborate, imaginative images.







"""