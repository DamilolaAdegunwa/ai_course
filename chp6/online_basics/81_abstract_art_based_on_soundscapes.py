"""
Project Title: Abstract Art Based on Soundscapes
File Name: abstract_art_based_on_soundscapes.py
Description:
This project takes image generation to the next level by using abstract interpretations of soundscapes as the basis for creating unique artworks. The prompts will describe specific audio environments, such as a bustling city, a tranquil forest, or an intense thunderstorm, and these audio cues will be translated into abstract visual representations. The exercise encourages you to push your creativity by describing sound in visual terms (colors, shapes, and movement) and imagining how those concepts could be turned into vibrant and artistic abstract images.

The focus will be on blending different sensory experiences (sound and sight) to create truly imaginative and conceptual visual outputs.

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


# Function to generate abstract art based on soundscapes
def generate_abstract_art(prompt):
    """
    Generate an abstract art image based on soundscapes described in the prompt.

    :param prompt: The prompt describing a soundscape and how it translates into an abstract visual.
    :return: URL of the generated abstract image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed abstract artwork
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
    # Define multiple prompts for abstract art based on soundscapes
    soundscape_prompts = [
        "The sound of ocean waves crashing on a rocky shore under a full moon, with silver and blue tones swirling together in abstract patterns.",
        "The chaotic sounds of a bustling city at rush hour, represented as jagged lines and vibrant bursts of color in a crowded, abstract cityscape.",
        "A quiet forest at dawn, where birds chirp softly and the wind rustles through the leaves, shown through calm pastel colors and smooth, flowing shapes.",
        "A thunderstorm with booming thunder and sharp lightning strikes, depicted through bold, dark strokes and flashes of electric yellow and white.",
        "A jazz band performing live in a smoky lounge, where the music's rhythm is visualized as smooth curves and spirals of deep reds and purples."
    ]

    # Generate, download, and save each abstract art image
    for i, prompt in enumerate(soundscape_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_abstract_art(prompt)
        image = download_image(image_url)
        save_image(image, f"abstract_art_soundscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "The sound of ocean waves crashing on a rocky shore under a full moon, with silver and blue tones swirling together in abstract patterns."
Expected Output:
An abstract image featuring swirling silver and blue patterns that reflect the fluidity of ocean waves under a moonlit sky, with dynamic but serene shapes.
Input:

Prompt: "The chaotic sounds of a bustling city at rush hour, represented as jagged lines and vibrant bursts of color in a crowded, abstract cityscape."
Expected Output:
A vibrant, chaotic abstract cityscape filled with jagged lines and sharp color contrasts, representing the energy and noise of a busy city.
Input:

Prompt: "A quiet forest at dawn, where birds chirp softly and the wind rustles through the leaves, shown through calm pastel colors and smooth, flowing shapes."
Expected Output:
A peaceful abstract scene using soft pastel colors and smooth, flowing shapes that convey the calm and quiet of a forest at dawn.
Input:

Prompt: "A thunderstorm with booming thunder and sharp lightning strikes, depicted through bold, dark strokes and flashes of electric yellow and white."
Expected Output:
A bold abstract image with dark, intense strokes and flashes of yellow and white that represent the power and suddenness of a thunderstorm.
Input:

Prompt: "A jazz band performing live in a smoky lounge, where the music's rhythm is visualized as smooth curves and spirals of deep reds and purples."
Expected Output:
An abstract image with smooth, rhythmic curves and spirals in rich reds and purples, capturing the flow and soul of a live jazz performance.
Project Overview:
In this project, you will focus on translating auditory experiences into visually compelling abstract art. The prompts describe soundscapes—environments created by sound—which are then turned into visually interesting abstract representations. This project not only exercises your image generation skills but also challenges your ability to think abstractly about how different sensory experiences can be visually depicted.
"""