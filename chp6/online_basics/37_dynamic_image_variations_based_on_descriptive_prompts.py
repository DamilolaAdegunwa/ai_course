"""
Project Title: Dynamic Image Variations Based on Descriptive Prompts
This project focuses on creating multiple variations of images based on different descriptive prompts. You will generate several unique images using OpenAI’s image generation API. This is more advanced than a simple single-image generation project, as you’ll create a function to handle dynamic image generation for different themes, and also test variations for multiple use cases. Additionally, this project will include saving each generated image with an appropriate name based on the prompt.

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


def download_and_save_image(image_url, prompt):
    """
    Download an image from a URL and save it with a filename based on the prompt.

    :param image_url: The URL of the image.
    :param prompt: The prompt used to generate the image, used for naming the file.
    :return: None
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Simplify the prompt to use it as a filename
    file_name = "_".join(prompt.split())[:50] + ".jpg"
    img.save(file_name)
    print(f"Image saved as {file_name}")


def generate_images_with_variations(prompts):
    """
    Generate multiple images from different descriptive prompts.

    :param prompts: A list of descriptive prompts for generating images.
    :return: None
    """
    for prompt in prompts:
        image_url = generate_image_from_prompt(prompt)
        download_and_save_image(image_url, prompt)


# Example use cases
if __name__ == "__main__":
    # List of prompts for generating variations of images
    prompts = [
        "A futuristic city at sunset",
        "A serene beach with palm trees swaying in the wind",
        "A cat wearing a spacesuit walking on the moon",
        "A fantasy kingdom with dragons flying in the sky",
        "An astronaut floating in deep space near a nebula"
    ]

    # Generate and save images for each prompt
    generate_images_with_variations(prompts)
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompt: "A futuristic city at sunset"
Expected Output:
Image of a city skyline with futuristic skyscrapers and flying vehicles, painted in warm, sunset colors. File saved as A_futuristic_city_at_sunset.jpg.
Input:

Prompt: "A serene beach with palm trees swaying in the wind"
Expected Output:
A peaceful beach scene with soft waves, palm trees gently swaying, and a clear blue sky. File saved as A_serene_beach_with_palm_trees_swaying_in_the_wind.jpg.
Input:

Prompt: "A cat wearing a spacesuit walking on the moon"
Expected Output:
A playful image of a cat in an astronaut suit walking on a barren moon surface, with Earth visible in the background. File saved as A_cat_wearing_a_spacesuit_walking_on_the_moon.jpg.
Input:

Prompt: "A fantasy kingdom with dragons flying in the sky"
Expected Output:
A grand, medieval-style kingdom with castles and mountains, dragons soaring majestically through the sky. File saved as A_fantasy_kingdom_with_dragons_flying_in_the_sky.jpg.
Input:

Prompt: "An astronaut floating in deep space near a nebula"
Expected Output:
An image of a lone astronaut drifting in the vastness of space, with colorful nebula clouds swirling in the distance. File saved as An_astronaut_floating_in_deep_space_near_a_nebula.jpg.
Project Overview
This project allows you to explore generating images based on a wide range of prompts, testing the API’s capabilities in interpreting various themes like fantasy, futuristic settings, animals, and landscapes. You will dynamically create and save each image to reflect the specific input prompt, ensuring versatility in the generated outputs. The exercise builds on your experience with image generation by introducing dynamic variations, scaling up in complexity while being easy to test with different types of input.

Through this project, you'll gain the ability to handle multiple inputs, optimize image saving with proper naming, and explore how the API responds to diverse descriptions, making it an ideal stepping stone toward more advanced AI-driven image manipulation tasks.
"""
