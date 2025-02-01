"""
Project Title: Multi-Concept Image Generation Using Text Descriptors
This project focuses on generating images that combine multiple distinct concepts or subjects within a single prompt. The idea is to create complex imagery by layering multiple elements, such as animals, objects, and environments, to test how well the AI can integrate diverse ideas into a single cohesive image.

For instance, you can prompt the AI to generate an image that combines "a futuristic city skyline with a dragon flying over it" or "a robot playing chess in a forest."

Python File Name: multi_concept_image_generation.py
Python Code
python
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


# Function to generate images based on multiple concepts
def generate_multi_concept_image(prompt):
    """
    Generate an image based on a complex prompt that includes multiple concepts.

    :param prompt: The prompt that describes multiple concepts or elements (e.g., 'a robot playing chess in a forest').
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Fixed size for high-quality outputs
    )

    return response.data[0].url  # Returns the URL of the generated image


def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def save_image(image, filename):
    """
    Save the image to the specified file.

    :param image: PIL Image object.
    :param filename: The path and name of the file to save the image to.
    """
    image.save(filename)
    print(f"Image saved as {filename}")


# Example use cases
if __name__ == "__main__":
    # Define a set of multi-concept prompts to test
    multi_concept_prompts = [
        "A futuristic city skyline with a dragon flying over it",
        "A robot playing chess in a dense forest",
        "A surreal landscape with floating islands and giant mushrooms",
        "A tiger sitting in a café reading a newspaper",
        "An astronaut walking on the moon with Earth in the background and a dog by their side"
    ]

    # Loop through each prompt and generate images
    for i, prompt in enumerate(multi_concept_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_multi_concept_image(prompt)
        image = download_image(image_url)
        save_image(image, f"multi_concept_image_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompt: "A futuristic city skyline with a dragon flying over it"
Expected Output:
A highly detailed image of a futuristic city with neon lights and tall skyscrapers, and a large, majestic dragon soaring through the sky above.
Input:

Prompt: "A robot playing chess in a dense forest"
Expected Output:
An image showing a robotic figure seated at a chessboard in a lush, green forest, with rays of sunlight filtering through the trees.
Input:

Prompt: "A surreal landscape with floating islands and giant mushrooms"
Expected Output:
A fantastical image featuring floating islands in the sky, with enormous mushrooms dotting the landscape, creating an otherworldly scene.
Input:

Prompt: "A tiger sitting in a café reading a newspaper"
Expected Output:
A whimsical scene of a tiger sitting at a table in a modern café, casually reading a newspaper, with other café elements such as coffee cups and pastries.
Input:

Prompt: "An astronaut walking on the moon with Earth in the background and a dog by their side"
Expected Output:
A space-themed image with an astronaut in a space suit walking across the lunar surface, Earth visible in the distant sky, and a friendly dog trotting alongside them.
Project Overview
This project challenges you to explore how OpenAI can handle the integration of multiple, diverse concepts within a single image generation request. By experimenting with creative combinations of subjects, objects, and environments, you’ll be able to push the boundaries of image generation beyond simple descriptions. You can test various themes like futuristic cities, animals in unexpected scenarios, or entirely surreal landscapes. This exercise not only improves your ability to craft complex prompts but also helps you understand how to handle multiple subjects in AI-generated art.
"""