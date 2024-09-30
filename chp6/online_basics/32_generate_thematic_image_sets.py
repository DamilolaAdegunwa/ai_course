"""
https://chatgpt.com/c/66f99f0b-8548-800c-a6a2-40e8732f7833 (from my dammy account)
Project Description:
In this advanced project, you'll build a tool to generate thematic image sets based on a user's keyword or concept. Instead of generating a single image, you'll create a set of related images, exploring variations on a given prompt. This project will allow for testing the creative boundaries of the OpenAI API by producing a set of different but cohesive images tied to a common theme.

The main idea is to explore variations in the generated images, such as changes in style, perspective, or mood, based on the same core concept. This could be useful for generating artwork collections, storyboarding, or brainstorming design ideas.

Example Use Cases:
Generate a set of landscape images: Provide a concept like "mountain sunset," and the system will produce a series of mountain sunset images in different styles or lighting conditions.
Generate character variations: Input "futuristic warrior," and the tool will return different representations of futuristic warriors—some with different armor designs, weapons, and colors.
Generate mood-based imagery: Provide an emotional concept like "peaceful forest," and the system will create a series of forest images, each with different lighting, colors, and vibes.
Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_thematic_image_set(prompt, num_images=3):
    """
    Generate a series of images based on a single prompt with variations in style and mood.

    Parameters:
        prompt (str): The core idea or concept for the images.
        num_images (int): Number of images to generate in the series.

    Returns:
        List of image URLs.
    """
    image_urls = []

    # Iterate to create variations
    for i in range(num_images):
        response = client.images.generate(
            prompt=f"{prompt}, variation {i + 1}",
            size="1024x1024"
        )
        image_urls.append(response.data[0].url)

    return image_urls


# Example Use Cases

# Generate variations on a landscape prompt
landscape_images = generate_thematic_image_set("mountain sunset", num_images=5)
print("Mountain Sunset Variations:", landscape_images)

# Generate different versions of a futuristic character
character_images = generate_thematic_image_set("futuristic warrior", num_images=4)
print("Futuristic Warrior Variations:", character_images)

# Generate a series of peaceful mood-based images
mood_images = generate_thematic_image_set("peaceful forest", num_images=3)
print("Peaceful Forest Variations:", mood_images)
"""
Project Summary:
This project encourages you to push the boundaries of image generation by exploring variations. You can experiment with different prompts and see how creative you can get with series-based image creation. Since the generated images come from the same core concept but with different variations, the exercise demonstrates how flexible the API can be in producing diverse yet cohesive results.

This tool can be extended by allowing users to select specific attributes for each variation (e.g., lighting, color scheme, etc.), but the basic structure provides an excellent start for creative image generation projects.
"""