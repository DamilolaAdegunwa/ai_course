"""
Project Title: Image Generation with Layered Prompts and Style Transfer
In this exercise, you will explore generating images based on layered prompts and apply specific artistic styles. Instead of a single prompt, you'll combine multiple prompts and then specify a style for the output, allowing the generation of complex images. This project is designed to build on your prior knowledge while introducing new techniques like combining prompts and applying artistic filters to create richer, more varied outputs.

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


def generate_image_from_layered_prompts(prompt_list, style):
    """
    Generate an image based on multiple prompts and an artistic style.

    :param prompt_list: A list of strings containing the description prompts.
    :param style: A string containing the artistic style for the image (e.g., 'impressionist', 'modern', 'realistic').
    :return: URL of the generated image.
    """
    combined_prompt = ', '.join(prompt_list) + f" in {style} style"
    response = client.images.generate(
        prompt=combined_prompt,
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
    # Define multiple prompts and artistic styles
    prompts = [
        ["A dragon flying over a forest", "a medieval city in the background"],
        ["A futuristic city", "spaceships flying above", "a sunset on the horizon"],
        ["A serene beach with palm trees", "a boat sailing in the distance"],
        ["A snowy mountain landscape", "a cozy cabin", "northern lights"],
        ["An enchanted forest", "mysterious glowing mushrooms", "a deer standing in the clearing"]
    ]

    styles = ["impressionist", "realistic", "cyberpunk", "watercolor", "fantasy"]

    for i, prompt_set in enumerate(prompts):
        style = styles[i]
        print(f"Generating image with prompts {prompt_set} in {style} style...")
        image_url = generate_image_from_layered_prompts(prompt_set, style)
        image = download_image(image_url)
        save_image(image, f"layered_image_{i + 1}_{style}.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompts: ["A dragon flying over a forest", "a medieval city in the background"]
Style: "fantasy"
Expected Output:
A mystical scene featuring a dragon soaring above a dense, green forest with a grand medieval city visible in the distance. The fantasy style adds an epic, otherworldly feel to the image.
Input:

Prompts: ["A futuristic city", "spaceships flying above", "a sunset on the horizon"]
Style: "cyberpunk"
Expected Output:
A neon-lit futuristic cityscape with large, hovering spaceships and a beautiful sunset in the background. The cyberpunk style introduces a gritty, high-tech look with glowing signs and industrial elements.
Input:

Prompts: ["A serene beach with palm trees", "a boat sailing in the distance"]
Style: "watercolor"
Expected Output:
A peaceful beach scene with soft palm trees and a boat on the horizon, depicted in a watercolor style, where the colors blend seamlessly, giving a calm and dreamy feel.
Input:

Prompts: ["A snowy mountain landscape", "a cozy cabin", "northern lights"]
Style: "realistic"
Expected Output:
A breathtaking image of a snow-covered mountain with a warm, inviting cabin nestled near the base, illuminated by the northern lights in the sky, rendered in a photorealistic style.
Input:

Prompts: ["An enchanted forest", "mysterious glowing mushrooms", "a deer standing in the clearing"]
Style: "impressionist"
Expected Output:
A magical forest filled with softly glowing mushrooms and a peaceful deer standing in the clearing. The impressionist style gives the image a painterly look with soft brushstrokes and vibrant colors.
Project Overview
In this project, you’ll explore combining multiple prompts to create complex and detailed images with specific artistic styles. The exercise will enhance your ability to control the visual output through layered input and predefined styles, giving you more creative power.
"""