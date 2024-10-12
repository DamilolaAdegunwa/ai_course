"""
Project Title:
AI-Generated Fantasy Landscapes

File Name:
ai_generated_fantasy_landscapes.py

Project Description:
In this project, you'll create a Python script that generates various fantasy landscapes based on user-defined themes and creative inputs. The script allows users to input a wide range of imaginative themes such as "floating cities," "magical forests," or "post-apocalyptic deserts." The AI will then generate an image representing the landscape described in the prompt. This project will help you develop the ability to generate highly creative images using diverse and abstract prompts.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The file containing the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_image_from_prompt(prompt):
    """Generates an image from a user-defined theme for a fantasy landscape."""

    # Generate the image based on the user-defined theme
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # Adjust size as needed (e.g., "512x512", "1024x1024")
        n=1  # Number of images to generate per prompt
    )

    # Extract and return the image URL
    return response.data[0].url


def generate_fantasy_landscapes(themes):
    """Generates fantasy landscapes based on a list of user-defined themes."""
    landscape_images = []
    for i, theme in enumerate(themes):
        print(f"Generating image for theme {i + 1}: {theme}")
        image_url = generate_image_from_prompt(theme)
        landscape_images.append(image_url)
        print(f"Image {i + 1} URL: {image_url}")
    return landscape_images


# Example usage:
if __name__ == "__main__":
    # A list of user-defined themes for fantasy landscapes
    fantasy_landscape_themes = [
        "floating city in the sky with glowing waterfalls",
        "enchanted forest with glowing trees and mysterious creatures",
        "desert with giant crystals rising from the sand",
        "volcanic island surrounded by black sand beaches",
        "ancient ruins of a forgotten civilization in a jungle"
    ]

    # Generate the fantasy landscape images
    fantasy_landscape_images = generate_fantasy_landscapes(fantasy_landscape_themes)

    # Print out the URLs of the generated images
    for i, img_url in enumerate(fantasy_landscape_images):
        print(f"Fantasy Landscape Image {i + 1}: {img_url}")
"""
Example Inputs and Expected Outputs:
Input:
Theme: floating city in the sky with glowing waterfalls
Expected Output:
A URL linking to an image of a futuristic city hovering in the clouds, with glowing waterfalls cascading into the air below.

Input:
Theme: enchanted forest with glowing trees and mysterious creatures
Expected Output:
A URL linking to an image of a magical forest where glowing trees illuminate the surroundings, with strange creatures peeking through the mist.

Input:
Theme: desert with giant crystals rising from the sand
Expected Output:
A URL linking to an image of a barren desert landscape with colossal, sparkling crystals emerging from the dunes.

Input:
Theme: volcanic island surrounded by black sand beaches
Expected Output:
A URL linking to an image of a dramatic volcanic island, with lava flows in the distance and black sand beaches along the coast.

Input:
Theme: ancient ruins of a forgotten civilization in a jungle
Expected Output:
A URL linking to an image of moss-covered stone ruins deep in a jungle, overgrown with vines and surrounded by thick vegetation.

This project encourages the exploration of highly imaginative concepts, improving your ability to create complex visual outputs from abstract or fantastical inputs.







"""