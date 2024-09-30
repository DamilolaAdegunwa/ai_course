"""
Project Title: Dynamic Surrealism: Imagining Impossible Worlds
File Name: dynamic_surrealism_imagining_impossible_worlds.py
Description:
In this project, we’ll explore the world of surrealism by generating dynamic and imaginative impossible worlds. The concept of surrealism thrives on unusual juxtapositions and fantastical elements that seem to defy reality. This exercise challenges you to create prompts that envision worlds where the impossible becomes possible: floating cities, hybrid landscapes, and unconventional natural elements.

You’ll generate unique surreal images by combining different themes like nature, technology, and abstract forms. The complexity here lies in using surreal concepts, forcing the AI to blend elements that don’t traditionally belong together, pushing the limits of creativity and visualization.

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


# Class to generate surrealist images based on creative prompts
class SurrealismImageGenerator:
    def __init__(self, prompt):
        self.prompt = prompt

    def generate_image(self):
        """
        Generate a surrealist image based on the given prompt using OpenAI's image generation API.
        :return: URL of the generated surrealist image.
        """
        response = client.images.generate(
            prompt=self.prompt,
            size="1024x1024"  # Large size to capture surreal details
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
    # Define surrealism-inspired prompts that describe impossible worlds
    surreal_prompts = [
        "A city made of glass, floating above a desert with green sand and orange skies",
        "A giant clock made of clouds, ticking over a vast ocean filled with luminous jellyfish",
        "A waterfall that flows upwards, with mountains suspended in the sky, under a pink and turquoise sunset",
        "A robot forest where metallic trees grow, their leaves made of circuit boards and gears",
        "A world where the sun is a giant diamond, casting rainbow light on a field of black roses"
    ]

    # Generate, download, and save each surrealist image
    for i, prompt in enumerate(surreal_prompts):
        print(f"Generating surrealist image for prompt: '{prompt}'...")
        image_generator = SurrealismImageGenerator(prompt)
        image_url = image_generator.generate_image()
        surreal_image = download_image(image_url)
        save_image(surreal_image, f"surreal_image_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A city made of glass, floating above a desert with green sand and orange skies"
Expected Output:
An image of a transparent city made of glass towers floating mid-air. Below, an expansive desert with bright green sand stretches out, contrasting against an orange sky with soft clouds.
Input:

Prompt: "A giant clock made of clouds, ticking over a vast ocean filled with luminous jellyfish"
Expected Output:
A massive cloud-shaped clock floating in the sky, its hands made of cloud wisps. Below, the ocean is filled with glowing jellyfish, casting a soft light on the surface of the water.
Input:

Prompt: "A waterfall that flows upwards, with mountains suspended in the sky, under a pink and turquoise sunset"
Expected Output:
An abstract scene of a waterfall flowing in reverse, with water streaming upward into the sky. Floating mountains hover in the air under a dramatic pink and turquoise sky at sunset.
Input:

Prompt: "A robot forest where metallic trees grow, their leaves made of circuit boards and gears"
Expected Output:
An image of a forest where the trees have metallic trunks, with branches covered in leaves made of tiny circuit boards and gears. The forest feels mechanical, but oddly organic at the same time.
Input:

Prompt: "A world where the sun is a giant diamond, casting rainbow light on a field of black roses"
Expected Output:
A surreal landscape featuring a giant, glittering diamond in the sky, radiating rainbow colors over a vast field of pitch-black roses. The light refracts, creating beautiful rainbow shadows across the ground.
Project Overview:
In this project, you’ll expand your understanding of visual surrealism by generating images that merge the fantastical and the impossible. You’ll create detailed prompts that challenge the AI to visualize strange and otherworldly landscapes, focusing on elements that defy the laws of nature and logic.

This project will help you push boundaries in AI-generated art by focusing on surreal and dynamic compositions. It also encourages the exploration of unconventional settings and imagery, helping you become more adept at crafting complex prompts.
"""