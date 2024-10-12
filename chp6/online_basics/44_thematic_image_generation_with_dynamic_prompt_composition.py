"""
https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833
Project Title: Thematic Image Generation with Dynamic Prompt Composition
In this exercise, you will expand your image generation skills by using dynamic themes to create images that adapt based on the day of the week or specific events. The goal is to generate images that vary based on contextual information like themes (e.g., nature, urban, fantasy) and user-defined elements. The project focuses on dynamically adjusting prompts to suit varying contexts and creating visually rich images that can fit multiple scenarios, with different styles and themes.

Python Code

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


# Function to generate images based on dynamic themes and contextual elements
def generate_dynamic_image(theme, custom_elements):
    """
    Generate an image based on a specific theme and user-provided custom elements.

    :param theme: The main theme of the image (e.g., 'nature', 'urban', 'fantasy').
    :param custom_elements: A list of additional elements to include in the image.
    :return: URL of the generated image.
    """
    prompt = f"A beautiful {theme} scene with " + ', '.join(custom_elements)

    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"
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
    # Define a set of dynamic themes and elements to generate different images
    theme_set = [
        ("nature", ["majestic mountains", "a flowing river", "wild animals"]),
        ("urban", ["skyscrapers", "busy streets", "neon signs"]),
        ("fantasy", ["a floating castle", "dragons flying", "a magical forest"]),
        ("futuristic", ["hovering cars", "giant holograms", "a digital skyline"]),
        ("historical", ["ancient ruins", "stone temples", "a bustling marketplace"])
    ]

    # Loop through each theme and generate images
    for i, (theme, elements) in enumerate(theme_set):
        print(f"Generating image with theme '{theme}' and elements {elements}...")
        image_url = generate_dynamic_image(theme, elements)
        image = download_image(image_url)
        save_image(image, f"dynamic_image_{i + 1}_{theme}.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Theme: "nature"
Elements: ["majestic mountains", "a flowing river", "wild animals"]
Expected Output:
A visually stunning nature scene with tall, majestic mountains, a winding river cutting through the landscape, and various wild animals roaming freely.
Input:

Theme: "urban"
Elements: ["skyscrapers", "busy streets", "neon signs"]
Expected Output:
A bustling cityscape featuring tall skyscrapers, busy streets full of people and cars, and bright neon signs illuminating the scene.
Input:

Theme: "fantasy"
Elements: ["a floating castle", "dragons flying", "a magical forest"]
Expected Output:
An epic fantasy scene with a grand floating castle in the sky, dragons soaring above, and a mystical forest below with glowing trees.
Input:

Theme: "futuristic"
Elements: ["hovering cars", "giant holograms", "a digital skyline"]
Expected Output:
A futuristic city with advanced technology, including hovering cars zooming through the air, large holographic displays, and a skyline full of futuristic architecture.
Input:

Theme: "historical"
Elements: ["ancient ruins", "stone temples", "a bustling marketplace"]
Expected Output:
A scene from the past, showcasing ancient ruins, stone temples, and a busy marketplace with people trading and interacting in a historical setting.
Project Overview
This project allows you to generate images dynamically by combining different themes with custom elements. By exploring different thematic concepts such as nature, urban landscapes, fantasy worlds, futuristic settings, and historical moments, you will improve your ability to generate rich and varied imagery based on contextual inputs.
"""