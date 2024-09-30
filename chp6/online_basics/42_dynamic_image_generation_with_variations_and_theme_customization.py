"""
https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833
Project Title: Dynamic Image Generation with Variations and Theme Customization
In this project, you’ll focus on creating dynamic image variations using different themes. Instead of simple prompt-to-image generation, this exercise allows you to use a combination of a description prompt and a theme to generate images. The goal is to improve your understanding of how different prompts and themes impact image generation, leading to the creation of unique image outputs.

This project is designed to be noticeably more advanced than your previous image generation task by incorporating more detailed prompts and diverse thematic categories.

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


def generate_image_from_prompt(prompt, theme):
    """
    Generate an image based on a description prompt and a selected theme.

    :param prompt: A string containing the image description.
    :param theme: A string containing the theme for the image (e.g., 'fantasy', 'sci-fi', 'abstract').
    :return: URL of the generated image.
    """
    themed_prompt = f"{prompt} in {theme} theme"
    response = client.images.generate(
        prompt=themed_prompt,
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
    # Define a prompt and multiple themes
    prompt = "A castle on a mountain during sunset"
    themes = ["fantasy", "sci-fi", "cyberpunk", "steampunk", "abstract"]

    for i, theme in enumerate(themes):
        print(f"Generating image with {theme} theme...")
        image_url = generate_image_from_prompt(prompt, theme)
        image = download_image(image_url)
        save_image(image, f"castle_mountain_{theme}.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompt: "A castle on a mountain during sunset"
Theme: "fantasy"
Expected Output:
A beautifully rendered castle set on a towering mountain, with a mystical and ethereal glow, possibly including magical elements like floating islands or dragons.
Input:

Prompt: "A castle on a mountain during sunset"
Theme: "sci-fi"
Expected Output:
A futuristic interpretation of a castle, perhaps a high-tech structure surrounded by energy shields, against a colorful otherworldly sky.
Input:

Prompt: "A castle on a mountain during sunset"
Theme: "cyberpunk"
Expected Output:
An industrial, neon-lit castle with a dystopian feel, reflecting the essence of the cyberpunk aesthetic—dark tones, glowing lights, and mechanical elements.
Input:

Prompt: "A castle on a mountain during sunset"
Theme: "steampunk"
Expected Output:
A castle adorned with gears, steam pipes, and Victorian-style architecture, set against a hazy sunset filled with airships.
Input:

Prompt: "A castle on a mountain during sunset"
Theme: "abstract"
Expected Output:
A highly stylized, abstract representation of a castle and mountain, using geometric shapes, bold colors, and unusual compositions that leave room for interpretation.
Project Overview
This project expands your experience by allowing you to explore thematic variations in your image generation. You will focus on how the same base prompt can yield very different visual outcomes depending on the artistic theme applied, pushing your understanding of prompt-tuning and creative control further.
"""