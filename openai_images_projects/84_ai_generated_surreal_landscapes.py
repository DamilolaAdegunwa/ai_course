"""
Project Title: AI-Generated Surreal Landscapes
File Name: ai_generated_surreal_landscapes.py
Description:
In this advanced project, you'll generate surreal landscapes that blend the boundaries of imagination and reality. You will specify various elements like environment type, lighting, color palette, and abstract qualities to produce dream-like, otherworldly scenes. The AI will combine these traits to create visuals that evoke emotion and curiosity, exploring a wide range of unique, surreal compositions.

This exercise will help you improve your ability to design creative prompts for generating complex scenes and push your skills in AI-driven image generation.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key
import requests
from PIL.Image import Image
from io import BytesIO
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate surreal landscape images based on user-specified elements
def generate_surreal_landscape(environment, lighting, color_palette, abstract_quality):
    """
    Generate an image of a surreal landscape based on the given elements.

    :param environment: The type of environment (e.g., desert, forest, ocean).
    :param lighting: The lighting condition (e.g., sunset, twilight, moonlit).
    :param color_palette: The dominant colors (e.g., neon, pastel, monochrome).
    :param abstract_quality: The abstract element to add (e.g., floating islands, melting sky).
    :return: URL of the generated surreal landscape image.
    """
    prompt = (f"A surreal {environment} landscape during {lighting} with a {color_palette} color palette, "
              f"featuring {abstract_quality}.")

    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed surreal imagery
    )

    return response.data[0].url  # Returns the URL of the generated image


# Function to display and save the generated landscape image
def display_and_save_image(image_url, filename):
    """
    Download and save the surreal landscape image to a file.

    :param image_url: The URL of the generated surreal landscape image.
    :param filename: The path and name of the file to save the image to.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.save(filename)
    print(f"Image saved as {filename}")


# Example use cases
if __name__ == "__main__":
    # Define surreal landscape characteristics
    landscape_specs = [
        ("desert", "sunset", "neon", "floating pyramids"),
        ("forest", "moonlit", "pastel", "trees growing upside down"),
        ("ocean", "twilight", "monochrome", "giant jellyfish in the sky"),
        ("mountains", "golden hour", "vivid", "fractals in the clouds"),
        ("city", "dawn", "rainbow", "buildings made of glass and light")
    ]

    # Generate and save images for each landscape specification
    for specs in landscape_specs:
        environment, lighting, color_palette, abstract_quality = specs
        print(f"Generating image for a surreal {environment} landscape...")
        image_url = generate_surreal_landscape(environment, lighting, color_palette, abstract_quality)
        display_and_save_image(image_url, f"surreal_landscape_{environment}_{lighting}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Environment: "desert"
Lighting: "sunset"
Color Palette: "neon"
Abstract Quality: "floating pyramids"
Expected Output:
An image of a vast desert bathed in neon hues during a glowing sunset, with massive pyramids hovering in midair, defying gravity.
Input:

Environment: "forest"
Lighting: "moonlit"
Color Palette: "pastel"
Abstract Quality: "trees growing upside down"
Expected Output:
A serene moonlit forest with soft pastel colors, where the trees grow from the sky toward the ground, creating a surreal, dreamlike effect.
Input:

Environment: "ocean"
Lighting: "twilight"
Color Palette: "monochrome"
Abstract Quality: "giant jellyfish in the sky"
Expected Output:
A monochrome twilight ocean scene, with enormous translucent jellyfish floating in the sky, giving an eerie and otherworldly vibe.
Input:

Environment: "mountains"
Lighting: "golden hour"
Color Palette: "vivid"
Abstract Quality: "fractals in the clouds"
Expected Output:
A vivid mountain range bathed in the golden hour, with intricate fractal patterns forming within the clouds, creating a surreal and mathematical sky.
Input:

Environment: "city"
Lighting: "dawn"
Color Palette: "rainbow"
Abstract Quality: "buildings made of glass and light"
Expected Output:
A futuristic cityscape at dawn, with rainbow-colored buildings constructed entirely of shimmering glass and beams of light, creating an ethereal, holographic city.
Project Overview:
This project challenges you to generate surreal landscapes that capture the imagination. By using combinations of environments, lighting, colors, and abstract elements, you will create unique and visually striking imagery that blurs the line between reality and fantasy. This exercise pushes your skills in prompt creation and image generation, allowing you to explore highly artistic and surreal themes.
"""