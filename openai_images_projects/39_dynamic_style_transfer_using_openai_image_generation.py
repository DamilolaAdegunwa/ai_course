"""
https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833
Project Title: Dynamic Style Transfer Using OpenAI Image Generation
In this project, we will use OpenAI’s image generation capabilities to create images based on prompts, but with a twist—each generated image will adopt a different art style or visual treatment (e.g., watercolor, pixel art, realistic). The complexity comes from generating multiple variations of the same image content with different artistic interpretations, making this project a step up from previous image generation tasks.

The goal is to understand how prompts affect stylistic variation and combine them into a multi-frame presentation (like a collage or slideshow).

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


def generate_image_from_prompt(prompt):
    """
    Generate an image based on a description prompt.

    :param prompt: A string containing the image description.
    :return: URL of the generated image.
    """
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


def generate_image_with_styles(content_prompt, styles):
    """
    Generate images based on a content description and different artistic styles.

    :param content_prompt: A string describing the main content of the image.
    :param styles: A list of strings describing different artistic styles.
    :return: A list of PIL Image objects
    """
    images = []
    for style in styles:
        styled_prompt = f"{content_prompt} in {style} style"
        print(f"Generating image for: {styled_prompt}")
        image_url = generate_image_from_prompt(styled_prompt)
        image = download_image(image_url)
        images.append(image)
    return images


def save_images(images, output_dir="styled_images"):
    """
    Save multiple images to a specified directory.

    :param images: A list of PIL Image objects to save.
    :param output_dir: Directory where images will be saved.
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(images):
        img.save(f"{output_dir}/image_{i + 1}.jpg")
        print(f"Saved image_{i + 1}.jpg in {output_dir}")


# Example use cases
if __name__ == "__main__":
    # Main content of the image
    content_prompt = "A futuristic city skyline"

    # Artistic styles to apply
    styles = [
        "watercolor",
        "pixel art",
        "realistic",
        "cyberpunk",
        "abstract"
    ]

    # Generate images with different styles
    styled_images = generate_image_with_styles(content_prompt, styles)

    # Save the generated images
    save_images(styled_images)
"""
Multiple Example Inputs and Expected Outputs
Input:

Content Prompt: "A futuristic city skyline"
Styles: ["watercolor", "pixel art", "realistic", "cyberpunk", "abstract"]
Expected Output:
Five different images of a futuristic city skyline, each in a distinct style:
Watercolor painting of a city skyline
Pixel-art rendering of a city skyline
A highly realistic, detailed city skyline
A neon-lit, cyberpunk-themed city
Abstract interpretation of a city skyline using geometric shapes
Input:

Content Prompt: "A forest in autumn"
Styles: ["oil painting", "impressionist", "low poly", "sketch", "vaporwave"]
Expected Output:
Five images of an autumn forest:
An oil painting of a colorful autumn forest
Impressionist depiction of a forest with soft brush strokes
Low-poly 3D-rendered forest
A sketch-like hand-drawn forest
A vaporwave-themed forest with neon colors and retro style
Input:

Content Prompt: "A dragon flying over a medieval castle"
Styles: ["cartoon", "surrealist", "realistic", "fantasy", "minimalist"]
Expected Output:
Five distinct images of a dragon and a castle:
Cartoon-style dragon over a castle
Surrealist dreamlike version of the scene
A highly detailed, realistic dragon and castle
Fantasy art with magical elements
Minimalist, clean lines and shapes representing the dragon and castle
Input:

Content Prompt: "A space station orbiting Earth"
Styles: ["line art", "photorealistic", "anime", "low poly", "cyberpunk"]
Expected Output:
Five different representations of a space station:
A line-drawn space station with simple, clean lines
Photorealistic depiction of a futuristic space station orbiting Earth
Anime-style space station with bold, vibrant colors
A low-poly 3D version of the space station
A neon-lit, cyberpunk-style space station
Input:

Content Prompt: "A snow-covered mountain range"
Styles: ["impressionist", "hyperrealistic", "low poly", "watercolor", "sketch"]
Expected Output:
Five interpretations of a snow-covered mountain range:
Impressionist-style painting with soft brush strokes
Hyperrealistic mountain range with crisp details
Low-poly 3D mountain range with simple geometry
Watercolor painting with flowing colors
Hand-drawn sketch of the mountain range
Project Overview
This project allows you to explore how different artistic styles can transform the same content prompt into various visual forms. It will deepen your understanding of how to create diverse visual representations using AI and refine your skills in managing multiple prompt variations, downloading images, and saving them efficiently.

You'll generate multiple stylistic versions of a single image, learn how to adapt prompts for different visual interpretations, and practice handling multiple API calls and managing the resulting images. This exercise also introduces file handling and organization, which are important for managing multiple outputs effectively.
"""