"""
Project Title: Mythical Creatures in Surreal Landscapes
File Name: mythical_creatures_in_surreal_landscapes.py
Description:
This project is designed to generate surreal landscapes inhabited by mythical creatures, combining detailed elements from nature with fantastical beings like dragons, phoenixes, and more. The goal is to push your skills further by experimenting with highly imaginative prompts that focus on creating visually extraordinary scenes with rich color contrasts and mythical themes.

The exercise introduces a more advanced approach to image generation by focusing on scene complexity and mood, requiring you to describe the mythical creatures, their environment, and the atmosphere that ties the entire scene together.

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


# Function to generate surreal landscape images with mythical creatures
def generate_mythical_image(prompt):
    """
    Generate an image of a surreal landscape populated with mythical creatures based on a prompt.

    :param prompt: The prompt describing the mythical creatures and surreal landscapes.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed mythical landscapes
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
    # Define multiple prompts for surreal landscapes with mythical creatures
    mythical_prompts = [
        "A phoenix rising from glowing embers in a twilight forest with purple and orange skies",
        "A dragon perched on a floating island above crystal-clear waters under a green aurora",
        "A unicorn galloping through a misty enchanted forest with glowing flowers and golden trees",
        "A griffin flying over a waterfall that cascades into a bottomless pit surrounded by floating rocks",
        "A giant sea serpent weaving through colossal waves in an ocean lit by a blood-red moon"
    ]

    # Generate, download, and save each mythical image
    for i, prompt in enumerate(mythical_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_mythical_image(prompt)
        image = download_image(image_url)
        save_image(image, f"mythical_landscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A phoenix rising from glowing embers in a twilight forest with purple and orange skies"
Expected Output:
A vivid scene featuring a majestic phoenix, its wings ablaze, emerging from glowing embers. The backdrop includes a twilight forest with vibrant purple and orange hues in the sky.
Input:

Prompt: "A dragon perched on a floating island above crystal-clear waters under a green aurora"
Expected Output:
A massive dragon sitting majestically on a floating island, its scales reflecting the green aurora that lights up the night sky. The waters below are crystal-clear, creating a mystical and serene atmosphere.
Input:

Prompt: "A unicorn galloping through a misty enchanted forest with glowing flowers and golden trees"
Expected Output:
A graceful unicorn running through an enchanted forest filled with glowing flowers. The trees are golden, and a light mist covers the forest floor, creating a magical and otherworldly vibe.
Input:

Prompt: "A griffin flying over a waterfall that cascades into a bottomless pit surrounded by floating rocks"
Expected Output:
A powerful griffin soaring above a massive waterfall that plunges into a bottomless pit. Floating rocks are scattered around, and the atmosphere is filled with mist and mystery.
Input:

Prompt: "A giant sea serpent weaving through colossal waves in an ocean lit by a blood-red moon"
Expected Output:
A terrifying sea serpent weaving through gigantic waves in a dark, stormy ocean. Overhead, a blood-red moon illuminates the scene, casting an eerie glow on the water.
Project Overview:
This project will help you dive deeper into surreal and fantasy-themed image generation, focusing on the combination of mythical creatures and intricate environments. It encourages you to create rich, multi-layered descriptions that lead to visually compelling images. You'll work with both dynamic and serene scenes, allowing you to explore a wide range of visual styles.
"""