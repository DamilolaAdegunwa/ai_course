"""
Project Title: Text-to-Image Art Variations with Style Control
This project builds on your previous experience by introducing style control to the image generation process. Instead of generating a single image, this project allows you to experiment with various artistic styles (e.g., watercolor, 3D render, anime) for the same prompt. You’ll generate multiple variations of an image from the same text description but apply different art styles to create a diverse set of images.

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


def generate_image_from_prompt(prompt, style):
    """
    Generate an image based on a description prompt and style.

    :param prompt: A string containing the image description.
    :param style: A string containing the artistic style (e.g., 'watercolor', '3D render', 'anime').
    :return: URL of the generated image.
    """
    styled_prompt = f"{prompt} in {style} style"
    response = client.images.generate(
        prompt=styled_prompt,
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
    # Define a prompt and multiple styles
    prompt = "A futuristic city at sunset"
    styles = ["watercolor", "3D render", "anime", "digital painting", "oil painting"]

    for i, style in enumerate(styles):
        print(f"Generating image in {style} style...")
        image_url = generate_image_from_prompt(prompt, style)
        image = download_image(image_url)
        save_image(image, f"futuristic_city_{style}.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompt: "A futuristic city at sunset"
Style: "watercolor"
Expected Output:
An image of a futuristic city painted in a soft watercolor style, with delicate blending of colors typical of watercolor painting.
Input:

Prompt: "A futuristic city at sunset"
Style: "3D render"
Expected Output:
A hyper-realistic 3D render of the city, with clean lines, detailed reflections, and lighting effects, mimicking a 3D animation or video game scene.
Input:

Prompt: "A futuristic city at sunset"
Style: "anime"
Expected Output:
An anime-style drawing of the city, with vibrant colors, exaggerated perspectives, and sharp lines characteristic of anime art.
Input:

Prompt: "A futuristic city at sunset"
Style: "digital painting"
Expected Output:
A digital painting version of the city, with smooth brushstrokes, a vivid color palette, and a more stylized, artistic interpretation of the scene.
Input:

Prompt: "A futuristic city at sunset"
Style: "oil painting"
Expected Output:
An oil painting style rendition, with bold brushstrokes, rich textures, and deep, contrasting colors, as if painted on canvas using traditional oil paints.
Project Overview
This project advances your skills by focusing on style-based image variations. You are not only generating images from prompts but also controlling the artistic style, giving you the ability to create visually distinct outputs from the same base prompt. This approach helps you explore and manipulate creative possibilities with more depth while building on your image generation expertise.
"""