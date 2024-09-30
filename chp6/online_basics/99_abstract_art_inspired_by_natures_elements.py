"""
Project Title: Abstract Art Inspired by Nature's Elements
File Name: abstract_art_inspired_by_natures_elements.py
Description:
In this project, we will generate abstract art that draws inspiration from the elements of nature—water, fire, air, and earth. Each image will capture the essence of one element in an abstract, imaginative form. The goal is to use AI to interpret natural elements and create visually compelling pieces of abstract art, allowing room for creative experimentation.

The project challenges you to craft descriptive prompts that creatively depict nature’s forces in abstract ways. You will explore how the AI interprets these descriptions and generates captivating visual outputs. This project will push your prompt engineering skills while leveraging artistic abstraction as the central theme.

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


# Class to generate abstract art based on natural elements
class AbstractNatureArt:
    def __init__(self, prompt):
        self.prompt = prompt

    def generate_art(self):
        """
        Generate an abstract art image based on the given prompt using OpenAI's image generation API.
        :return: URL of the generated abstract art.
        """
        response = client.images.generate(
            prompt=self.prompt,
            size="1024x1024"  # Using 1024x1024 for higher resolution abstract art
        )
        return response.data[0].url  # Return the URL of the generated image


# Function to download and return the image as a PIL Image object
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
    # Define multiple abstract prompts inspired by nature's elements
    nature_prompts = [
        "Abstract representation of flowing water, rippling in a dark blue ocean with streaks of silver light",
        "A swirling vortex of flames in bright orange and red, mixed with smoke and ashes in an abstract form",
        "Abstract air currents floating through a transparent sky, with soft pastel hues of pink, purple, and blue",
        "An abstract depiction of the earth element, using rugged textures and earthy tones of brown and green",
        "A fusion of all four elements: fire, water, air, and earth, swirling together in an abstract cosmic dance"
    ]

    # Generate, download, and save each abstract nature art piece
    for i, prompt in enumerate(nature_prompts):
        print(f"Generating abstract art for prompt: '{prompt}'...")
        art_generator = AbstractNatureArt(prompt)
        art_url = art_generator.generate_art()
        art_image = download_image(art_url)
        save_image(art_image, f"abstract_art_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "Abstract representation of flowing water, rippling in a dark blue ocean with streaks of silver light"
Expected Output:
An abstract image of deep blue waves in a dark ocean, with silver streaks that represent light reflecting off the surface. The waves are flowing, creating a surreal, dream-like effect.
Input:

Prompt: "A swirling vortex of flames in bright orange and red, mixed with smoke and ashes in an abstract form"
Expected Output:
An image depicting swirling fiery shapes in bright orange and red. The flames twist and curl, interspersed with wisps of dark smoke and ash, creating a chaotic yet visually engaging abstract fire.
Input:

Prompt: "Abstract air currents floating through a transparent sky, with soft pastel hues of pink, purple, and blue"
Expected Output:
A soft and light abstract image representing air currents. The pastel colors of pink, purple, and blue blend together, forming flowing shapes that give the impression of wind moving through a clear sky.
Input:

Prompt: "An abstract depiction of the earth element, using rugged textures and earthy tones of brown and green"
Expected Output:
A textured image with abstract formations that represent the earth. The colors range from rich browns to dark greens, with rough, uneven patterns that give the appearance of a natural, rugged landscape.
Input:

Prompt: "A fusion of all four elements: fire, water, air, and earth, swirling together in an abstract cosmic dance"
Expected Output:
An image that combines abstract representations of fire, water, air, and earth. Fiery reds, deep blues, earthy browns, and light pastels swirl together in a mesmerizing cosmic dance, blending yet distinct in form and color.
Project Overview:
In this project, you’ll focus on generating abstract images inspired by the natural elements—water, fire, air, and earth. Through creative prompts, you will explore how abstract shapes, colors, and textures can represent the forces of nature in a surreal and artistic manner. The final result will be a collection of beautiful abstract art pieces that capture the essence of nature in a non-literal form. This exercise helps improve your prompt design skills, especially in generating imaginative and non-figurative concepts.
"""