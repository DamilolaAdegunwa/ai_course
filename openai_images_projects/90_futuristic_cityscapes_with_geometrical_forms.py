"""
Project Title: Futuristic Cityscapes with Abstract Geometrical Forms
File Name: futuristic_cityscapes_with_geometrical_forms.py
Description:
This project will involve generating futuristic cityscapes combined with abstract geometrical forms. The goal is to create a striking visual contrast between hyper-modern urban environments and surreal geometrical shapes floating or integrated into the environment. This exercise will test your skills in blending structured, sci-fi architecture with abstract art, using prompts to merge these elements into compelling, AI-generated images.

You will be generating highly detailed images that depict futuristic cities with skyscrapers, neon lights, and strange, otherworldly geometric forms—floating or towering structures that defy physics. The geometrical shapes may be massive cubes, pyramids, or spheres made of unknown materials, appearing in unexpected places in the scene.

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


# Function to generate a futuristic cityscape with geometrical forms
def generate_cityscape_image(prompt):
    """
    Generate an image based on a futuristic cityscape combined with abstract geometrical forms using OpenAI's image generation API.

    :param prompt: The prompt describing the futuristic cityscape and the geometric shapes.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed futuristic cityscapes
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
    # Define multiple prompts involving futuristic cityscapes with abstract geometrical forms
    cityscape_prompts = [
        "A futuristic city with towering skyscrapers and a giant floating cube casting a shadow over the skyline",
        "A neon-lit city at night with massive glowing spheres levitating between buildings",
        "A hyper-modern cityscape with geometric pyramids suspended in the sky, with light beams emitting from their peaks",
        "A futuristic metropolis surrounded by giant transparent geometrical structures, glowing softly with neon light",
        "A sprawling sci-fi city with intricate skyscrapers and an enormous floating dodecahedron reflecting the city's lights"
    ]

    # Generate, download, and save each cityscape image
    for i, prompt in enumerate(cityscape_prompts):
        print(f"Generating futuristic cityscape image for prompt: '{prompt}'...")
        cityscape_url = generate_cityscape_image(prompt)
        cityscape_image = download_image(cityscape_url)
        save_image(cityscape_image, f"futuristic_cityscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A futuristic city with towering skyscrapers and a giant floating cube casting a shadow over the skyline"
Expected Output:
A cityscape with tall, metallic skyscrapers. In the sky, a massive floating cube hovers ominously, casting a shadow over the buildings and streets below.
Input:

Prompt: "A neon-lit city at night with massive glowing spheres levitating between buildings"
Expected Output:
A vibrant night scene of a futuristic city bathed in neon lights. Giant glowing spheres hover between the skyscrapers, casting reflections on the glass surfaces of the buildings.
Input:

Prompt: "A hyper-modern cityscape with geometric pyramids suspended in the sky, with light beams emitting from their peaks"
Expected Output:
A sleek cityscape with sharp, angular skyscrapers. In the sky, pyramids float, and light beams shoot out from their peaks, illuminating the city below with a futuristic glow.
Input:

Prompt: "A futuristic metropolis surrounded by giant transparent geometrical structures, glowing softly with neon light"
Expected Output:
A sprawling metropolis surrounded by semi-transparent geometrical shapes like spheres and cubes, softly glowing in neon hues. The shapes seem to be made of an ethereal material that reflects the city’s lights.
Input:

Prompt: "A sprawling sci-fi city with intricate skyscrapers and an enormous floating dodecahedron reflecting the city's lights"
Expected Output:
An expansive sci-fi cityscape with intricately designed skyscrapers. Above the city hovers a gigantic dodecahedron, its metallic surfaces reflecting the city’s neon lights in all directions.
Project Overview:
This project focuses on the intersection of structured, futuristic city designs with abstract and surreal geometrical forms, creating highly imaginative and complex scenes. You will explore creative prompt-engineering techniques to generate unique visual compositions, combining urban environments with artistic geometry. This exercise pushes the boundaries of using AI-generated imagery for surrealistic and sci-fi-themed art creation.







"""