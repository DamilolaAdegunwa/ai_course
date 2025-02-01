"""
Project Title: Sci-Fi Alien Worlds with Advanced Technology Structures
File Name: sci_fi_alien_worlds_with_advanced_technology.py
Description:
This project will focus on generating visually stunning images of alien worlds with futuristic technology. You will craft prompts to create scenes that depict extraterrestrial landscapes featuring advanced civilizations and technological marvels such as flying ships, hovering cities, and towering structures made of unknown materials. The goal is to push the creative limits of AI-generated sci-fi art by blending natural alien terrains with advanced technological designs.

By the end of this project, you'll be able to generate immersive sci-fi worlds that merge creativity and futuristic elements. The scenes may range from desolate alien deserts with gleaming technology to lush forests housing advanced alien machinery.

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


# Factory Pattern to generate image based on theme
class SciFiWorldFactory:
    def __init__(self, prompt):
        self.prompt = prompt

    def create_world_image(self):
        """
        Generate an image based on a sci-fi alien world prompt using OpenAI's image generation API.
        :return: URL of the generated image.
        """
        response = client.images.generate(
            prompt=self.prompt,
            size="1024x1024"  # High resolution for detailed sci-fi landscapes
        )
        return response.data[0].url  # Return the URL of the generated image


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
    # Define multiple prompts involving sci-fi alien worlds with advanced technology structures
    sci_fi_prompts = [
        "An alien desert with hovering cities made of glass and light, under a sky filled with colorful nebulae",
        "A dense alien jungle where massive tree-like structures hold floating mechanical platforms",
        "A futuristic alien ocean world with flying ships and technological islands rising from the water",
        "An icy alien planet with tall crystal structures, glowing with alien symbols and energy fields",
        "A volcanic alien world with giant energy towers drawing power from lava flows, surrounded by advanced alien machinery"
    ]

    # Generate, download, and save each sci-fi world image
    for i, prompt in enumerate(sci_fi_prompts):
        print(f"Generating sci-fi world for prompt: '{prompt}'...")
        world_factory = SciFiWorldFactory(prompt)
        world_url = world_factory.create_world_image()
        world_image = download_image(world_url)
        save_image(world_image, f"sci_fi_world_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "An alien desert with hovering cities made of glass and light, under a sky filled with colorful nebulae"
Expected Output:
A bright, surreal desert landscape. In the sky, cities hover, made of transparent glass-like structures illuminated by internal glowing lights. The sky is filled with nebulae of vibrant colors, making the scene feel both futuristic and otherworldly.
Input:

Prompt: "A dense alien jungle where massive tree-like structures hold floating mechanical platforms"
Expected Output:
A lush jungle scene with gigantic, otherworldly tree-like structures. Between the trees, mechanical platforms hover, connected by shimmering pathways. The foliage glows faintly, suggesting a bioluminescent environment enhanced by alien technology.
Input:

Prompt: "A futuristic alien ocean world with flying ships and technological islands rising from the water"
Expected Output:
A vast alien ocean with flying ships cruising through the air. Floating technological islands with sleek, metallic surfaces rise from the ocean, surrounded by mist and futuristic energy waves.
Input:

Prompt: "An icy alien planet with tall crystal structures, glowing with alien symbols and energy fields"
Expected Output:
A cold, icy planet where tall, translucent crystal structures glow with alien symbols. The crystals emit energy fields that shimmer in the frigid air, adding an ethereal feel to the otherwise barren landscape.
Input:

Prompt: "A volcanic alien world with giant energy towers drawing power from lava flows, surrounded by advanced alien machinery"
Expected Output:
A fiery landscape filled with rivers of lava. Towering energy structures stand amidst the heat, drawing power from the molten flows. Surrounding these towers are intricate alien machines that hum with futuristic energy, forming a highly advanced industrial setting.
Project Overview:
This project explores the creation of sci-fi worlds with advanced technology, where alien environments are transformed through futuristic designs. From cities that hover in the sky to mysterious planets with glowing crystal structures, this exercise blends imaginative world-building with AI-generated visuals. You will use OpenAI's image generation to craft immersive alien worlds that capture both the awe-inspiring beauty of other planets and the technological advancement of alien civilizations.
"""